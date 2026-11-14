"""The MLOps agent: LLM tool-calling loop with permissions, evidence checks and tracing.

Guarantees enforced in code (not just in the prompt):
- Tools only run through the registry (validation, timeouts, structured logs).
- State-changing tools never run inside a chat turn. They become PendingActions that the user
  must approve explicitly (POST /actions/{id}/confirm or the CLI prompt).
- A "fact" is only kept if every tool it cites succeeded in this request; anything else is
  moved to uncertainties. Every failed tool call is reported as an uncertainty, so the answer
  can never silently present data that a tool did not return.
"""

import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent.llm import LLMClient, LLMError, LLMTurn, ToolCall
from agent.logging_config import get_logger, new_request_id, redact, set_request_id
from agent.schemas import (
    ChatResponse,
    ConfirmResponse,
    Fact,
    PendingAction,
    ToolCallRecord,
    ToolResult,
    ToolStatus,
)
from agent.tools import ToolRegistry

logger = get_logger("agent.orchestrator")

MAX_RESULT_CHARS = 12_000  # per tool result sent back to the model

SYSTEM_PROMPT = """\
You are the MLOps operations agent for one ML project. You answer questions about datasets
(DVC), experiments (MLflow), training pipelines (Apache Airflow 3) and the deployed prediction
API (Kubernetes) by calling the provided tools. Today is {today}.

EVIDENCE RULES (most important)
1. Every operational statement (versions, hashes, commits, run ids, metrics, states, replica
   counts, pod health, log lines) must come from a tool result in THIS conversation. Never use
   prior knowledge, never guess, never fill gaps with plausible values.
2. If a tool fails or is unavailable, say exactly which information is missing and why.
3. Call independent tools together in the same turn to save steps.
4. Cite the tool name(s) behind every fact.

PROJECT KNOWLEDGE
- MLflow experiments: "mlops-agent-training" (manual train.py runs) and
  "airflow-training-pipeline" (runs created by the Airflow DAG, tagged quality_gate=passed|failed).
  Every run has provenance tags: dataset_version, dvc_md5, dataset_git_commit, git_commit,
  data_matches_pointer.
- Dataset used by a run: read its provenance (mlflow_get_run or the run summaries), then
  confirm with dvc_find_version_by_hash using dvc_md5. Compare with dvc_current_version.
- Airflow DAG "training_pipeline": validate_dataset -> preprocess_data -> train_model ->
  evaluate_and_log_mlflow -> quality_gate. A failed validation or gate fails the run.
- Prediction API: Deployment and Service "prediction-api" in namespace "{namespace}".
  k8s_deployed_version returns the live /version, including the MLflow run id and dataset
  version of the model actually loaded. Use k8s_diagnose_service for availability questions.
- The highest metric does not make a model production-ready. Present rankings as evidence
  with their caveats, never as a promotion decision.

SAFETY
- State-changing tools ({write_tools}) are never executed by you. Calling one only creates a
  pending action that the user must approve. Only call them when the user explicitly asks for
  that exact operation. Never trigger training or retraining unless the user explicitly asks
  to trigger it; recommending it is different from doing it.
- Kubernetes access is limited to namespaces: {allowed}.

RETRAINING POLICY
Recommend retraining only with evidence: (a) the deployed model's training run is older than
{max_age} days, (b) the latest pipeline run failed its quality gate or failed in a way that
leaves no valid model, (c) model-degradation results supplied by the user, or (d) a newer DVC
dataset version exists than the one the deployed model was trained on. Otherwise state that
retraining is not currently indicated and which evidence would change that.

For broad requests ("best model, its dataset, deployment status, is retraining needed"):
1) MLflow: best/recent runs and metrics  2) provenance -> DVC version  3) Kubernetes: deployed
version and health  4) Airflow: recent runs of training_pipeline  5) correlate (is the deployed
model the best one? trained on the current dataset? did the last pipeline pass its gate?).

FINAL ANSWER FORMAT
When you have the evidence, reply with ONLY one JSON object and no tool calls:
{{"answer": "concise markdown summary for the engineer",
  "facts": [{{"statement": "...", "sources": ["tool_name", ...]}}],
  "uncertainties": ["what is unknown or could not be verified, and why"],
  "recommended_actions": ["next steps; state-changing ones need user approval"]}}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class PendingActionStore:
    """In-memory, single-use, expiring store of actions awaiting user approval."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl = ttl_seconds
        self._items: dict[str, tuple[float, PendingAction]] = {}
        self._lock = threading.Lock()

    def add(self, tool: str, arguments: dict[str, Any], request_id: str | None) -> PendingAction:
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        action = PendingAction(
            action_id=f"act-{uuid.uuid4().hex[:10]}",
            tool=tool,
            arguments=arguments,
            description=f"{tool}({args})",
            request_id=request_id,
        )
        with self._lock:
            self._purge()
            self._items[action.action_id] = (time.monotonic(), action)
        return action

    def pop(self, action_id: str) -> PendingAction | None:
        with self._lock:
            self._purge()
            entry = self._items.pop(action_id, None)
        return entry[1] if entry else None

    def list(self) -> list[PendingAction]:
        with self._lock:
            self._purge()
            return [a for _, a in self._items.values()]

    def _purge(self) -> None:
        now = time.monotonic()
        for key in [k for k, (t, _) in self._items.items() if now - t > self.ttl]:
            del self._items[key]


def _payload_for_model(result: ToolResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": result.status.value}
    if result.ok:
        payload["data"] = result.data
    else:
        payload["error"] = result.error.model_dump() if result.error else None
        payload["note"] = "This tool did not return data. Do not state values it would provide."
    text = json.dumps(payload, default=str)
    if len(text) > MAX_RESULT_CHARS:
        payload = {
            "status": result.status.value,
            "truncated": True,
            "data_preview": text[:MAX_RESULT_CHARS],
        }
    return payload


def parse_final(text: str | None) -> dict[str, Any]:
    """Extract the final JSON object; fall back to plain text if the model ignored the format."""
    if not text:
        return {"answer": ""}
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    match = _JSON_BLOCK.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "answer" in data:
                return data
        except json.JSONDecodeError:
            pass
    return {"answer": text.strip()}


def _facts(raw: Any) -> list[Fact]:
    facts = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("statement"):
            sources = item.get("sources") or []
            facts.append(Fact(statement=str(item["statement"]), sources=[str(s) for s in sources]))
        elif isinstance(item, str):
            facts.append(Fact(statement=item, sources=[]))
    return facts


def _strings(raw: Any) -> list[str]:
    return [str(x) for x in raw] if isinstance(raw, list) else []


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient | None,
        registry: ToolRegistry,
        *,
        pending: PendingActionStore | None = None,
        max_steps: int = 8,
        max_tool_calls: int = 20,
        tool_timeout: float = 20.0,
        model_max_age_days: int = 30,
        namespace: str = "mlops",
        allowed_namespaces: str = "mlops",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.pending = pending or PendingActionStore()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout
        self.model_max_age_days = model_max_age_days
        self.namespace = namespace
        self.allowed_namespaces = allowed_namespaces

    def system_prompt(self) -> str:
        write_tools = [s.name for s in self.registry.specs() if s.requires_confirmation]
        return SYSTEM_PROMPT.format(
            today=datetime.now(UTC).date().isoformat(),
            namespace=self.namespace,
            allowed=self.allowed_namespaces,
            write_tools=", ".join(write_tools) or "none",
            max_age=self.model_max_age_days,
        )

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        message: str,
        *,
        request_id: str | None = None,
        include_evidence: bool = False,
        prior_results: list[ToolResult] | None = None,
    ) -> ChatResponse:
        rid = request_id or new_request_id()
        set_request_id(rid)
        started = time.perf_counter()
        # Results of actions the user approved earlier count as evidence for this answer.
        results: list[ToolResult] = list(prior_results or [])
        records: list[ToolCallRecord] = []
        pending: list[PendingAction] = []
        notes: list[str] = []

        if self.llm is None:
            final = {"answer": "The language model is not configured (missing LLM API key)."}
        else:
            final = self._run_loop(message, rid, results, records, pending, notes)

        response = self._build_response(rid, final, results, records, pending, notes)
        if include_evidence:
            response.evidence = results
        logger.info(
            "chat finished",
            extra={
                "event": "chat",
                "request_id": rid,
                "status": "ok" if not notes else "degraded",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    def _run_loop(self, message, rid, results, records, pending, notes) -> dict[str, Any]:
        try:
            session = self.llm.start(self.system_prompt(), self.registry.specs())
            turn: LLMTurn = session.send(message)
            for _step in range(self.max_steps):
                if not turn.tool_calls:
                    return parse_final(turn.text)
                replies = []
                for call in turn.tool_calls:
                    if len(records) >= self.max_tool_calls:
                        payload = {"status": "rejected", "error": "tool-call budget exhausted"}
                        notes.append("Tool-call budget reached; the investigation may be partial.")
                    else:
                        payload = self._execute(call, rid, results, records, pending)
                    replies.append((call, payload))
                turn = session.send_tool_results(replies)
            notes.append("Step limit reached before the agent finished; the answer may be partial.")
            return parse_final(turn.text) if not turn.tool_calls else {"answer": ""}
        except LLMError as exc:
            logger.warning(
                "llm failure",
                extra={"event": "llm_call", "status": "error", "error_type": "LLMError"},
            )
            notes.append(f"The language model became unavailable: {redact(str(exc))}")
            return {"answer": ""}

    def _execute(self, call: ToolCall, rid, results, records, pending) -> dict[str, Any]:
        result = self.registry.execute(
            call.name, call.arguments, confirmed=False, request_id=rid, timeout=self.tool_timeout
        )
        results.append(result)
        records.append(
            ToolCallRecord(
                tool=result.tool,
                source=result.source,
                status=result.status,
                duration_ms=result.duration_ms,
                request_id=rid,
                arguments=call.arguments,
                error=result.error.message if result.error else None,
            )
        )
        if result.status == ToolStatus.CONFIRMATION_REQUIRED:
            action = self.pending.add(call.name, call.arguments, rid)
            pending.append(action)
            return {
                "status": "pending_user_approval",
                "action_id": action.action_id,
                "note": (
                    "NOT executed. The user must approve this action. Tell them what it will "
                    "do and that it is awaiting their confirmation. Do not claim it ran."
                ),
            }
        return _payload_for_model(result)

    def _build_response(self, rid, final, results, records, pending, notes) -> ChatResponse:
        # A proposed (pending) action is itself a verified fact: the system created it.
        succeeded = {
            r.tool for r in results if r.ok or r.status == ToolStatus.CONFIRMATION_REQUIRED
        }
        facts, uncertainties = [], []
        for fact in _facts(final.get("facts")):
            if fact.sources and set(fact.sources) <= succeeded:
                facts.append(fact)
            else:
                uncertainties.append(
                    f"Unverified (not backed by a successful tool result, removed from facts): "
                    f"{fact.statement}"
                )
        uncertainties = _strings(final.get("uncertainties")) + uncertainties

        reported: set[str] = set()
        for r in results:
            if r.ok or r.status == ToolStatus.CONFIRMATION_REQUIRED or r.tool in reported:
                continue
            reported.add(r.tool)
            source = r.source.value if r.source else "unknown"
            message = r.error.message if r.error else ""
            uncertainties.append(
                f"{r.tool} ({source}) returned '{r.status.value}': {message}. "
                f"No data from this call was used."
            )
        uncertainties.extend(notes)

        answer = str(final.get("answer") or "").strip()
        if not answer:
            answer = (
                "I could not complete the investigation. Only the facts listed below are "
                "backed by tool results; see the uncertainties for what is missing."
                if facts
                else "I could not obtain verified information for this request. "
                "No operational results are reported; see the uncertainties for the reasons."
            )
        actions = _strings(final.get("recommended_actions"))
        for action in pending:
            actions.append(
                f"Awaiting your approval: {action.description} (action id {action.action_id})"
            )
        return ChatResponse(
            request_id=rid,
            answer=answer,
            facts=facts,
            uncertainties=uncertainties,
            recommended_actions=actions,
            pending_actions=pending,
            tool_calls=records,
        )

    def continue_after_action(
        self, question: str, confirmation: ConfirmResponse, *, include_evidence: bool = False
    ) -> ChatResponse:
        """After the user approved and the system executed an action, finish the request."""
        outcome = confirmation.result
        if outcome is None:
            raise ValueError("Only executed actions can be continued")
        detail = (
            outcome.data if outcome.ok else (outcome.error.model_dump() if outcome.error else {})
        )
        message = (
            f"Original request: {question}\n\n"
            f"The user APPROVED the pending action and the system executed it:\n"
            f"{confirmation.tool}({json.dumps(confirmation.arguments)}) -> "
            f"status '{outcome.status.value}': {json.dumps(detail, default=str)[:4000]}\n\n"
            "Continue the original request with tools (for a triggered DAG run call "
            "airflow_wait_for_run, then airflow_failed_tasks if it failed; after a restart or scale "
            "call k8s_wait_for_rollout before checking health). Do not repeat the action. "
            f"You may cite '{confirmation.tool}' as a source for the execution result."
        )
        return self.chat(
            message,
            request_id=outcome.request_id,
            include_evidence=include_evidence,
            prior_results=[outcome],
        )

    # --------------------------------------------------------------- confirm
    def confirm(self, action_id: str, approve: bool) -> ConfirmResponse | None:
        action = self.pending.pop(action_id)
        if action is None:
            return None
        if not approve:
            logger.info(
                "action rejected by user",
                extra={"event": "confirmation", "tool_name": action.tool, "status": "rejected"},
            )
            return ConfirmResponse(
                action_id=action_id,
                tool=action.tool,
                arguments=action.arguments,
                approved=False,
                message="Action discarded; nothing changed.",
            )
        set_request_id(action.request_id or new_request_id())
        result = self.registry.execute(
            action.tool,
            action.arguments,
            confirmed=True,
            request_id=action.request_id,
            timeout=self.tool_timeout,
        )
        message = "Executed." if result.ok else f"Not completed: {result.status.value}"
        return ConfirmResponse(
            action_id=action_id,
            tool=action.tool,
            arguments=action.arguments,
            approved=True,
            result=result,
            message=message,
        )


def build_orchestrator() -> Orchestrator:
    """Wire the real Gemini client, the tool registry and settings together."""
    import os

    from agent.config import get_settings
    from agent.llm import GeminiClient
    from agent.tools import load_all_tools

    settings = get_settings()
    os.environ.setdefault("K8S_NAMESPACE", settings.k8s_namespace)
    os.environ.setdefault("K8S_ALLOWED_NAMESPACES", settings.k8s_allowed_namespaces)
    llm = None
    if settings.gemini_api_key is not None:
        llm = GeminiClient(
            settings.gemini_api_key.get_secret_value(),
            settings.llm_model,
            settings.llm_timeout_seconds,
        )
    return Orchestrator(
        llm,
        load_all_tools(),
        pending=PendingActionStore(settings.pending_action_ttl_seconds),
        max_steps=settings.agent_max_steps,
        max_tool_calls=settings.agent_max_tool_calls,
        tool_timeout=settings.tool_timeout_seconds,
        model_max_age_days=settings.model_max_age_days,
        namespace=settings.k8s_namespace,
        allowed_namespaces=settings.k8s_allowed_namespaces,
    )
