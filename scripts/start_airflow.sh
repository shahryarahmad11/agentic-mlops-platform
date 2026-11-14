#!/usr/bin/env bash
# Airflow 3 (standalone) from its own virtualenv, reading DAGs from this repository.
# Airflow is kept out of the project environment because its pins conflict with MLflow/DVC.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/airflow-venv/bin/activate"
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$REPO/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
exec airflow standalone
