#!/usr/bin/env bash
# MLflow tracking server (SQLite backend, local artifact store). Leave running.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000 --workers 1
