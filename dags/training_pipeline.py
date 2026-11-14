"""Training pipeline DAG: validate -> preprocess -> train -> evaluate/log -> quality gate.

Each stage runs training/pipeline_steps.py with the project's own Python. A failed validation
or quality gate fails its task, which fails the run, so any downstream promotion step must
depend on `quality_gate` and can never start after a failure.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

DAG_ID = "training_pipeline"
PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"
STEPS = "training/pipeline_steps.py"
ENV = {
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    "PYTHONUNBUFFERED": "1",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}


def step(task_id: str, args: str, **kwargs) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd '{PROJECT_DIR}' && '{PYTHON}' {STEPS} {args}",
        env=ENV,
        append_env=True,
        **kwargs,
    )


with DAG(
    dag_id=DAG_ID,
    description="Validate, preprocess, train, evaluate/log to MLflow, then apply a quality gate",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=10),
        "execution_timeout": timedelta(minutes=10),
    },
    params={
        "min_f1": Param(0.90, type="number", minimum=0, maximum=1),
        "n_estimators": Param(100, type="integer", minimum=1, maximum=1000),
        "max_depth": Param(5, type="integer", minimum=1, maximum=50),
    },
    tags=["mlops", "training"],
) as dag:
    # Deterministic failures (bad data, low quality) are not retried; infrastructure-prone
    # stages are retried twice.
    validate_dataset = step("validate_dataset", "validate", retries=0)
    preprocess_data = step("preprocess_data", "preprocess")
    train_model = step(
        "train_model",
        "train --n-estimators {{ params.n_estimators }} --max-depth {{ params.max_depth }}",
    )
    evaluate_and_log_mlflow = step(
        "evaluate_and_log_mlflow",
        "evaluate --airflow-dag-id " + DAG_ID + " --airflow-run-id '{{ run_id }}'",
    )
    quality_gate = step("quality_gate", "gate --min-f1 {{ params.min_f1 }}", retries=0)

    (
        validate_dataset
        >> preprocess_data
        >> train_model
        >> evaluate_and_log_mlflow
        >> quality_gate
    )