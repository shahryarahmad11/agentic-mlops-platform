"""Agent behaviour: tool selection, multi-tool reasoning, no invented results, approvals."""

import json

import httpx

from agent.llm import LLMError
from agent.orchestrator import Orchestrator, PendingActionStore, parse_final
from agent.schemas import ToolSource, ToolSpec, ToolStatus
from agent.tools import ToolRegistry
from tests.conftest import ScriptedLLM, calls, final

MD5 = "3acbb1e7aee10825dc89fbc3216816ad"


def fake_registry(mlflow_down: bool = False) -> tuple[ToolRegistry, list[str]]:
    reg, executed = ToolRegistry(), []

    def tool(name, source, **spec_kwargs):
        def wrap(func):
            def run(**kwargs):
                executed.append(name)
                return func(**kwargs)

            params = spec_kwargs.pop("parameters", {"type": "object", "properties": {}})
            reg.register(
                ToolSpec(
                    name=name, description=name, source=source, parameters=params, **spec_kwargs
                )
            )(run)
            return func

        return wrap

    @tool("mlflow_best_run", ToolSource.MLFLOW)
    def best():
        if mlflow_down:
            raise httpx.ConnectError("Connection refused")
        return {
            "best": {"run_id": "r1", "metrics": {"f1_macro": 0.985}, "provenance": {"dvc_md5": MD5}}
        }

    @tool("dvc_find_version_by_hash", ToolSource.DVC)
    def find():
        return {"found": True, "matches": [{"dataset_version": "data-v2", "commit": "c57fc80"}]}

    @tool("k8s_deployed_version", ToolSource.KUBERNETES)
    def deployed():
        return {"ready_replicas": 2, "live_version": {"model": {"mlflow_run_id": "r1"}}}

    @tool("airflow_get_dag_runs", ToolSource.AIRFLOW)
    def runs():
        return {"runs": [{"state": "success"}]}

    @tool(
        "airflow_trigger_dag",
        ToolSource.AIRFLOW,
        read_only=False,
        requires_confirmation=True,
        parameters={"type": "object", "properties": {"dag_id": {"type": "string"}}},
    )
    def trigger(dag_id="training_pipeline"):
        return {"dag_run_id": "manual__1", "state": "queued"}

    return reg, executed


def answer(facts, uncertainties=(), actions=(), text="summary") -> str:
    return json.dumps(
        {
            "answer": text,
            "facts": facts,
            "uncertainties": list(uncertainties),
            "recommended_actions": list(actions),
        }
    )


def test_agent_selects_and_executes_the_right_tool():
    reg, executed = fake_registry()
    llm = ScriptedLLM(
        calls(("mlflow_best_run", {})),
        final(answer([{"statement": "Best run r1, F1 0.985", "sources": ["mlflow_best_run"]}])),
    )
    response = Orchestrator(llm, reg).chat("Which run is best?")
    assert executed == ["mlflow_best_run"]
    assert response.tool_calls[0].status == ToolStatus.SUCCESS
    assert response.facts[0].statement == "Best run r1, F1 0.985"
    # The model received the real tool data, not a summary of it.
    assert llm.tool_results[0][0][1]["data"]["best"]["run_id"] == "r1"
    assert "mlflow_best_run" in llm.tool_names and "Today is" in llm.system_prompt


def test_mandatory_scenario_uses_all_four_technologies_and_never_retrains():
    reg, executed = fake_registry()
    llm = ScriptedLLM(
        calls(("mlflow_best_run", {}), ("airflow_get_dag_runs", {})),
        calls(("dvc_find_version_by_hash", {}), ("k8s_deployed_version", {})),
        # A misbehaving model tries to retrain on its own: it must only become a pending action.
        calls(("airflow_trigger_dag", {"dag_id": "training_pipeline"})),
        final(
            answer(
                [
                    {"statement": "Best run r1", "sources": ["mlflow_best_run"]},
                    {"statement": "Trained on data-v2", "sources": ["dvc_find_version_by_hash"]},
                    {"statement": "r1 is deployed, 2/2 ready", "sources": ["k8s_deployed_version"]},
                    {"statement": "Last pipeline succeeded", "sources": ["airflow_get_dag_runs"]},
                ],
                actions=["Retraining not indicated"],
            )
        ),
    )
    response = Orchestrator(llm, reg).chat("Find the best model ... is retraining necessary?")
    assert {c.source for c in response.tool_calls} == set(ToolSource)
    assert "airflow_trigger_dag" not in executed  # nothing was retrained
    assert len(response.facts) == 4
    assert response.pending_actions[0].tool == "airflow_trigger_dag"
    assert llm.tool_results[2][0][1]["status"] == "pending_user_approval"


def test_agent_does_not_invent_results_when_a_tool_is_unavailable():
    reg, _ = fake_registry(mlflow_down=True)
    # The model hallucinates a metric even though MLflow was unreachable.
    llm = ScriptedLLM(
        calls(("mlflow_best_run", {})),
        final(answer([{"statement": "Best run r1 with F1 0.985", "sources": ["mlflow_best_run"]}])),
    )
    response = Orchestrator(llm, reg).chat("Which run is best?")
    assert response.tool_calls[0].status == ToolStatus.UNAVAILABLE
    assert response.facts == []  # the invented fact was rejected
    joined = " ".join(response.uncertainties)
    assert "Unverified" in joined and "mlflow_best_run (mlflow) returned 'unavailable'" in joined
    sent = llm.tool_results[0][0][1]
    assert sent["status"] == "unavailable" and "data" not in sent


def test_facts_without_sources_are_not_presented_as_facts():
    reg, _ = fake_registry()
    llm = ScriptedLLM(final(answer([{"statement": "Model v3 is deployed", "sources": []}])))
    response = Orchestrator(llm, reg).chat("What is deployed?")
    assert response.facts == [] and "Model v3 is deployed" in response.uncertainties[0]


def test_pending_action_runs_only_after_explicit_approval():
    reg, executed = fake_registry()
    store = PendingActionStore()
    llm = ScriptedLLM(
        calls(("airflow_trigger_dag", {"dag_id": "training_pipeline"})),
        final(answer([], text="Awaiting approval")),
    )
    orch = Orchestrator(llm, reg, pending=store)
    response = orch.chat("Trigger the training pipeline")
    action = response.pending_actions[0]
    assert executed == [] and response.tool_calls[0].status == ToolStatus.CONFIRMATION_REQUIRED

    assert orch.confirm("act-unknown", True) is None
    confirmed = orch.confirm(action.action_id, approve=True)
    assert confirmed.result.ok and executed == ["airflow_trigger_dag"]
    assert orch.confirm(action.action_id, approve=True) is None  # single use


def test_rejected_action_changes_nothing():
    reg, executed = fake_registry()
    llm = ScriptedLLM(calls(("airflow_trigger_dag", {})), final(answer([])))
    orch = Orchestrator(llm, reg)
    action = orch.chat("trigger").pending_actions[0]
    assert orch.confirm(action.action_id, approve=False).approved is False
    assert executed == []


def test_llm_failure_returns_honest_degraded_answer():
    reg, _ = fake_registry()
    llm = ScriptedLLM(calls(("mlflow_best_run", {})), LLMError("LLM API error 503"))
    response = Orchestrator(llm, reg).chat("best run?")
    assert response.facts == []
    assert "could not" in response.answer.lower()
    assert any("language model became unavailable" in u for u in response.uncertainties)


def test_step_and_tool_budgets_stop_runaway_loops():
    reg, executed = fake_registry()
    llm = ScriptedLLM(*[calls(("airflow_get_dag_runs", {}))] * 5)
    response = Orchestrator(llm, reg, max_steps=3, max_tool_calls=2).chat("loop")
    assert len(executed) == 2
    assert any("budget" in u for u in response.uncertainties)


def test_parse_final_handles_fenced_json_and_plain_text():
    assert parse_final('```json\n{"answer": "hi", "facts": []}\n```')["answer"] == "hi"
    assert parse_final("plain words")["answer"] == "plain words"
    assert parse_final(None) == {"answer": ""}


def test_invalid_tool_request_from_model_is_reported():
    reg, _ = fake_registry()
    llm = ScriptedLLM(calls(("k8s_delete_namespace", {})), final(answer([])))
    response = Orchestrator(llm, reg).chat("delete everything")
    assert response.tool_calls[0].status == ToolStatus.INVALID_REQUEST


def test_after_approval_the_agent_continues_with_the_execution_result_as_evidence():
    reg, executed = fake_registry()
    llm = ScriptedLLM(
        calls(("airflow_trigger_dag", {"dag_id": "training_pipeline"})),
        final(answer([], text="Awaiting approval")),
        calls(("airflow_get_dag_runs", {})),
        final(
            answer(
                [
                    {"statement": "Run manual__1 was queued", "sources": ["airflow_trigger_dag"]},
                    {"statement": "Last run succeeded", "sources": ["airflow_get_dag_runs"]},
                ]
            )
        ),
    )
    orch = Orchestrator(llm, reg)
    first = orch.chat("Trigger the pipeline and report the outcome")
    confirmation = orch.confirm(first.pending_actions[0].action_id, approve=True)
    follow_up = orch.continue_after_action("Trigger the pipeline", confirmation)
    assert executed == ["airflow_trigger_dag", "airflow_get_dag_runs"]  # triggered exactly once
    assert len(follow_up.facts) == 2
