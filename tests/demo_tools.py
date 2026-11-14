"""Manual demo: run real tools through the registry. Usage: python tests/demo_tools.py mlflow"""

import json
import sys

from agent.logging_config import configure_logging, new_request_id
from agent.tools import load_all_tools

EXPERIMENT = "mlops-agent-training"

SCENARIOS: dict[str, list[tuple[str, str, dict]]] = {
    "mlflow": [
        ("1. Experiments", "mlflow_list_experiments", {}),
        ("2. Recent runs", "mlflow_list_runs", {"experiment_name": EXPERIMENT, "limit": 3}),
        (
            "3. One run (by name)",
            "mlflow_get_run",
            {"run_name": "rf-v2-n200-d10", "experiment_name": EXPERIMENT},
        ),
        ("4. Best run overall", "mlflow_best_run", {"experiment_name": EXPERIMENT}),
        (
            "5. Best run on data-v1 only",
            "mlflow_best_run",
            {"experiment_name": EXPERIMENT, "dataset_version": "data-v1"},
        ),
        ("6. Compare last 5 runs", "mlflow_compare_runs", {"experiment_name": EXPERIMENT}),
        (
            "7. Compare experiments",
            "mlflow_compare_experiments",
            {"experiment_names": [EXPERIMENT]},
        ),
        ("8. Registry", "mlflow_registry_info", {}),
        ("9. INVALID: unknown experiment", "mlflow_list_runs", {"experiment_name": "nope"}),
        (
            "10. INVALID: injection in run name",
            "mlflow_get_run",
            {"run_name": "x' OR 1=1", "experiment_name": EXPERIMENT},
        ),
        (
            "11. INVALID: bad metric",
            "mlflow_best_run",
            {"experiment_name": EXPERIMENT, "metric": "f1; DROP"},
        ),
        ("12. INVALID: bad run id", "mlflow_get_run", {"run_id": "xyz"}),
    ],
}


def main() -> None:
    group = sys.argv[1] if len(sys.argv) > 1 else ""
    if group not in SCENARIOS:
        raise SystemExit(f"Usage: python tests/demo_tools.py [{'|'.join(SCENARIOS)}]")
    configure_logging("WARNING", environment="test")
    registry = load_all_tools()
    new_request_id()
    only = sys.argv[2] if len(sys.argv) > 2 else None
    for title, name, args in SCENARIOS[group]:
        if only and not title.startswith(f"{only}."):
            continue
        result = registry.execute(name, args)
        print(f"\n### {title}\n{result.tool} -> {result.status.value} ({result.duration_ms} ms)")
        print(json.dumps(result.data if result.ok else result.error.model_dump(), indent=2))


if __name__ == "__main__":
    main()