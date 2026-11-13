# Agentic MLOps Platform

AI-powered MLOps agent that turns natural-language requests into evidence-backed operations across **DVC**, **MLflow**, **Apache Airflow 3** and **Kubernetes**. Built with **FastAPI**, **Docker** and **GitHub Actions CI**, featuring permissioned tool calling, structured logging and automated tests.

> Operational claims made by the agent are always backed by real tool responses, never by the model's training knowledge.

## Planned Capabilities

| Module | Purpose |
|---|---|
| DVC tools | Dataset discovery, version/commit lookup, version comparison, change detection, reproduction guidance |
| MLflow tools | Experiment and run listing, metrics/params/artifacts, best-run selection, comparison, model registry |
| Airflow tools | DAG listing, run status, approved DAG triggering, failed-task diagnostics |
| Kubernetes tools | Pods/Deployments/Services inspection, logs, restart events, confirmed and namespace-restricted operations |
| Orchestrator | Multi-tool reasoning that separates known facts, uncertainties and recommended actions |

## Repository Structure

```
agentic-mlops-platform/
├── agent/            # Orchestrator, schemas and the four tool modules
├── api/              # FastAPI interface
├── dags/             # Airflow training pipeline
├── data/             # DVC-tracked datasets
├── training/         # Model training code
├── kubernetes/       # Deployment, Service, ConfigMap manifests
├── tests/            # pytest suite
├── screenshots/      # Evidence for the technical report
├── .github/workflows/ci.yml
├── Dockerfile
├── dvc.yaml
├── pyproject.toml
└── architecture.md
```

## Environment Setup (so far)

- OS: Ubuntu on WSL2
- Python 3.12 managed with [uv](https://github.com/astral-sh/uv) in an isolated `.venv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync                # install locked dependencies (pyproject.toml + uv.lock)
cp .env.example .env   # then edit .env locally; never commit it
```

## Security

No API keys, kubeconfigs or tokens are committed. Secrets are supplied via environment variables, and `.gitignore` blocks common credential files.

## Progress Log

### Push 1: Project foundation
- Created the public GitHub repository and linked it from WSL using GitHub CLI authentication
- Scaffolded the full project structure
- Added a security-focused `.gitignore`
- Created a Python 3.12 virtual environment with uv

### Push 2: Dependencies and secret handling
- Declared dependencies and dev tooling in `pyproject.toml` (FastAPI, MLflow, DVC, Kubernetes client, pytest, ruff) and locked them with `uv.lock`
- Python pinned to 3.12; Apache Airflow 3 is intentionally kept out of this environment and will run separately, since the agent talks to it over its REST API
- Added `.env.example` as the configuration contract; real values live in a git-ignored `.env`
- Chose Gemini (free tier) as the agent's LLM; its API key is supplied only through an environment variable

### Push 3: Configuration, logging and DVC (Task 1 foundation)
- Added `agent/config.py`: validated, fail-fast settings with clear errors (invalid environment, unsafe `DEBUG` in production, bad numeric values, missing LLM key) and the API key held as a masked secret
- Added `agent/logging_config.py`: structured JSON logs with request IDs and automatic redaction of keys and tokens
- Initialised DVC with a local remote (`../dvc-storage`) and telemetry disabled
- Added `training/generate_data.py` (reproducible synthetic dataset, `--rows` and `--seed` arguments)
- Tracked dataset v1 (500 rows) with DVC: only the `data/train.csv.dvc` pointer is in Git, and the CSV lives in the DVC cache and remote. The commit is tagged `data-v1`.

### Push 4: Dataset v2, historical restoration and MLflow server
- Created and tracked dataset v2 (1000 rows), tagged `data-v2`; `dvc status` detected the modification before it was tracked
- Demonstrated reproducible restoration: checking out `data-v1` and running `dvc pull` restores the 500-row dataset, and returning to `main` restores the 1000-row dataset (evidence in `screenshots/01_dvc_version_restore.png`)
- Verified the local MLflow tracking server (SQLite backend, local artifact store)

## Launching Services (so far)

MLflow tracking server (leave running in its own terminal):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### Push 5: Training script and MLflow experiment tracking (Task 2 foundation)
- Added `training/train.py`: trains a Random Forest on the DVC-tracked dataset and logs parameters, metrics (accuracy, macro-F1) and the model to MLflow
- Every run records provenance as MLflow tags: DVC content hash, dataset version tag, the Git commit that introduced that dataset version, current Git commit, a dirty-tree flag, and whether the CSV on disk still matches the DVC pointer
- Fixed model logging for MLflow 3.x by trusting only the scikit-learn `Tree` type when saving the model (no blanket trust)
- Trained three configurations on each dataset version (500 and 1000 rows), producing six tracked runs
- Evidence: `screenshots/02_mlflow_runs_comparison.png`, `screenshots/03_mlflow_run_provenance.png`
