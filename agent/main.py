"""Command-line chat with the MLOps agent.

  python -m agent.main                              interactive session
  python -m agent.main "Which dataset version ..."  one question
  python -m agent.main "..." --save evidence/c1.json  also save answer + raw tool evidence

State-changing actions proposed by the agent are shown and run only if you type 'yes'.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from agent.config import ConfigError, get_settings
from agent.logging_config import configure_logging
from agent.orchestrator import Orchestrator, build_orchestrator
from agent.schemas import ChatResponse, ConfirmResponse

ICONS = {"success": "ok ", "confirmation_required": "?? "}


def render(response: ChatResponse) -> str:
    out = [f"\n=== Answer  [{response.request_id}]", response.answer.strip()]
    if response.facts:
        out.append("\n--- Facts (tool-backed)")
        out += [f"  * {f.statement}  [{', '.join(f.sources)}]" for f in response.facts]
    if response.uncertainties:
        out.append("\n--- Uncertainties")
        out += [f"  ? {u}" for u in response.uncertainties]
    if response.recommended_actions:
        out.append("\n--- Recommended actions")
        out += [f"  > {a}" for a in response.recommended_actions]
    out.append(f"\n--- Tool trace ({len(response.tool_calls)} calls)")
    for c in response.tool_calls:
        mark = ICONS.get(c.status.value, "!! ")
        args = json.dumps(c.arguments) if c.arguments else ""
        out.append(f"  {mark}{c.tool:<28} {c.status.value:<22} {c.duration_ms:>8.0f} ms  {args}")
    return "\n".join(out)


def handle_pending(orchestrator: Orchestrator, response: ChatResponse) -> list[ConfirmResponse]:
    outcomes = []
    for action in response.pending_actions:
        print(f"\n!!! The agent wants to run a STATE-CHANGING action:\n    {action.description}")
        answer = input("    Type 'yes' to approve, anything else to reject: ").strip().lower()
        result = orchestrator.confirm(action.action_id, approve=answer == "yes")
        if result is None:
            print("    Action expired.")
            continue
        print(f"    -> {result.message}")
        if result.result is not None:
            payload = result.result.data if result.result.ok else result.result.error.model_dump()
            print(json.dumps(payload, indent=2, default=str))
        outcomes.append(result)
    return outcomes


def ask(orchestrator: Orchestrator, question: str, save: Path | None) -> None:
    response = orchestrator.chat(question, include_evidence=save is not None)
    print(render(response))
    confirmations = handle_pending(orchestrator, response)
    follow_ups = []
    for confirmation in confirmations:
        if confirmation.approved and confirmation.result is not None:
            print("\n... approved action executed; the agent continues the request ...")
            follow_up = orchestrator.continue_after_action(
                question, confirmation, include_evidence=save is not None
            )
            print(render(follow_up))
            follow_ups.append(follow_up)
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        record = {"question": question, "response": response.model_dump(mode="json")}
        if confirmations:
            record["confirmations"] = [c.model_dump(mode="json") for c in confirmations]
        if follow_ups:
            record["follow_up_responses"] = [f.model_dump(mode="json") for f in follow_ups]
        save.write_text(json.dumps(record, indent=2, default=str))
        print(f"\n(saved to {save})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--save", type=Path, help="write answer + raw tool evidence as JSON")
    parser.add_argument("--verbose", action="store_true", help="show structured JSON logs")
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ConfigError as exc:
        sys.exit(str(exc))
    configure_logging("INFO" if args.verbose else "WARNING", environment=settings.app_env)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    orchestrator = build_orchestrator()
    if orchestrator.llm is None:
        sys.exit("GEMINI_API_KEY is not set")

    if args.question:
        ask(orchestrator, args.question, args.save)
        return
    print("MLOps agent. Ask a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(orchestrator, question, None)


if __name__ == "__main__":
    main()
