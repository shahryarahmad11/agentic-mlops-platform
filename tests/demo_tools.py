"""Manual demo: run real tools through the registry.

Usage: python -m tests.demo_tools {mlflow|airflow|airflow-fail|k8s} [scenario number]
"""

import json
import sys
import time

from agent.logging_config import configure_logging, new_request_id
from agent.tools import load_all_tools

EXPERIMENT = "mlops-agent-training"
DAG = "training_pipeline"
APP = "prediction-api"
LAST_RUN = "$LAST_RUN"  # replaced by the run id of the most recent trigger
CONFIRMED = {"confirmed": True}

SCENARIOS: dict[str, list[tuple]] = {
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
            {"experiment_names": [EXPERIMENT, "airflow-training-pipeline"]},
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
    "airflow": [
        ("1. DAGs", "airflow_list_dags", {}),
        ("2. Recent runs", "airflow_get_dag_runs", {"dag_id": DAG}),
        ("3. Trigger WITHOUT confirmation", "airflow_trigger_dag", {"dag_id": DAG}),
        ("4. Trigger WITH confirmation", "airflow_trigger_dag", {"dag_id": DAG}, CONFIRMED),
        ("5. Wait for the run", "_wait", {"dag_id": DAG, "dag_run_id": LAST_RUN}),
        ("6. Run details", "airflow_get_run_details", {"dag_id": DAG, "dag_run_id": LAST_RUN}),
        ("7. Failed tasks", "airflow_failed_tasks", {"dag_id": DAG, "dag_run_id": LAST_RUN}),
        (
            "8. Gate task log",
            "airflow_task_logs",
            {"dag_id": DAG, "dag_run_id": LAST_RUN, "task_id": "quality_gate", "tail": 6},
        ),
        (
            "9. INVALID: DAG not approved",
            "airflow_trigger_dag",
            {"dag_id": "example_bash_operator"},
            CONFIRMED,
        ),
        (
            "10. INVALID: threshold out of range",
            "airflow_trigger_dag",
            {"dag_id": DAG, "min_f1": 5},
            CONFIRMED,
        ),
        (
            "11. INVALID: path traversal in run id",
            "airflow_get_run_details",
            {"dag_id": DAG, "dag_run_id": "../../x"},
        ),
        ("12. INVALID: unknown DAG", "airflow_get_dag_runs", {"dag_id": "no_such_dag"}),
    ],
    "airflow-fail": [
        (
            "1. Trigger with an impossible quality threshold",
            "airflow_trigger_dag",
            {"dag_id": DAG, "min_f1": 0.999},
            CONFIRMED,
        ),
        ("2. Wait for the run", "_wait", {"dag_id": DAG, "dag_run_id": LAST_RUN}),
        ("3. Run details", "airflow_get_run_details", {"dag_id": DAG, "dag_run_id": LAST_RUN}),
        (
            "4. Failed tasks + diagnostics",
            "airflow_failed_tasks",
            {"dag_id": DAG, "dag_run_id": LAST_RUN},
        ),
    ],
    "k8s": [
        ("1. Deployments", "k8s_list_deployments", {}),
        ("2. Pods", "k8s_list_pods", {"label_selector": "app.kubernetes.io/name=prediction-api"}),
        ("3. Services + endpoints", "k8s_list_services", {}),
        ("4. Deployed version (live /version)", "k8s_deployed_version", {}),
        ("5. Unhealthy pods + restarts", "k8s_unhealthy_pods", {}),
        ("6. Warning events", "k8s_get_events", {}),
        ("7. Full diagnosis", "k8s_diagnose_service", {}),
        ("8. Restart WITHOUT confirmation", "k8s_restart_deployment", {"deployment": APP}),
        (
            "9. FORBIDDEN: restart in kube-system (even confirmed)",
            "k8s_restart_deployment",
            {"deployment": "coredns", "namespace": "kube-system"},
            CONFIRMED,
        ),
        (
            "10. INVALID: scale to 50 replicas",
            "k8s_scale_deployment",
            {"deployment": APP, "replicas": 50},
            CONFIRMED,
        ),
        ("11. UNKNOWN: delete tool does not exist", "k8s_delete_deployment", {"deployment": APP}),
    ],
}


def wait_for_run(registry, dag_id: str, run_id: str, timeout: int = 300):
    deadline = time.monotonic() + timeout
    while True:
        result = registry.execute(
            "airflow_get_run_details", {"dag_id": dag_id, "dag_run_id": run_id}, timeout=20
        )
        if not result.ok or result.data["is_terminal"] or time.monotonic() > deadline:
            return result
        time.sleep(5)


def main() -> None:
    group = sys.argv[1] if len(sys.argv) > 1 else ""
    if group not in SCENARIOS:
        raise SystemExit(f"Usage: python -m tests.demo_tools [{'|'.join(SCENARIOS)}] [number]")
    only = sys.argv[2] if len(sys.argv) > 2 else None
    configure_logging("WARNING", environment="test")
    registry = load_all_tools()
    new_request_id()
    last_run = None
    for entry in SCENARIOS[group]:
        title, name, args = entry[:3]
        options = entry[3] if len(entry) > 3 else {}
        if only and not title.startswith(f"{only}."):
            continue
        args = {key: (last_run if value == LAST_RUN else value) for key, value in args.items()}
        if name == "_wait":
            result = wait_for_run(registry, args["dag_id"], args["dag_run_id"])
        else:
            result = registry.execute(name, args, **options)
        if name == "airflow_trigger_dag" and result.ok:
            last_run = result.data["dag_run_id"]
        print(f"\n### {title}\n{result.tool} -> {result.status.value} ({result.duration_ms} ms)")
        print(json.dumps(result.data if result.ok else result.error.model_dump(), indent=2))


if __name__ == "__main__":
    main()
