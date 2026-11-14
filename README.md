# Agentic MLOps Platform

AI-powered MLOps agent that turns natural-language requests into evidence-backed operations across **DVC**, **MLflow**, **Apache Airflow 3** and **Kubernetes**. Built with **FastAPI**, **Docker** and **GitHub Actions CI**, featuring permissioned tool calling, structured logging and automated tests.

> Operational claims made by the agent are always backed by real tool responses, never by the model's training knowledge. This is enforced in code: a "fact" whose cited tool did not succeed is moved to *uncertainties*.

See [architecture.md](architecture.md) for the component diagram, request flow, the full tool/permission table and the safety controls.

## Capabilities

| Module | Tools | Purpose |
|---|---|---|
| DVC (`agent/tools/dvc_tools.py`) | 7 read-only | Datasets, current version + commit, history, compare versions, change detection, reproduction steps, hash to version lookup |
| MLflow (`agent/tools/mlflow_tools.py`) | 7 read-only | Experiments, runs, metrics/params/artifacts, best run (with caveats), run and experiment comparison, registry |
| Airflow (`agent/tools/airflow_tools.py`) | 6 read-only + 1 approval-gated | DAGs, run status, task states, wait for a run, failed tasks + log diagnostics; trigger only the approved DAG |
| Kubernetes (`agent/tools/kubernetes_tools.py`) | 8 read-only + 2 approval-gated | Deployments, Pods, logs, Services/endpoints, events, unhealthy Pods/restarts, deployed version, full diagnosis; restart/scale only in approved namespaces |
| Orchestrator (`agent/orchestrator.py`) | - | Gemini function calling, multi-tool reasoning, facts/uncertainties/actions, pending-action approvals, budgets |

## Repository Structure

```
agentic-mlops-platform/
├── agent/            # config, logging, schemas, LLM adapter, orchestrator, CLI, four tool modules
├── api/              # FastAPI interface for the agent
├── serving/          # FastAPI inference service deployed on Kubernetes
├── dags/             # Airflow training pipeline
├── data/             # DVC-tracked dataset (pointer in Git, data in the DVC remote)
├── training/         # data generation, train.py, pipeline stages, model export
├── kubernetes/       # Namespace, ConfigMap, Deployment, Service, agent RBAC
├── scripts/          # start services, deploy, inject failures, run demo conversations
├── tests/            # pytest suite (all external systems mocked)
├── evidence/         # saved agent conversations with raw tool evidence
├── screenshots/      # evidence for the technical report
├── .github/workflows/ci.yml
├── Dockerfile        # inference service image
├── dvc.yaml
├── pyproject.toml / uv.lock
└── architecture.md
```

## 1. Installation

Ubuntu (WSL2), Python 3.12 managed with [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync                      # locked dependencies from uv.lock
cp .env.example .env         # fill in GEMINI_API_KEY and AIRFLOW_PASSWORD locally; never commit .env
dvc pull                     # restore the dataset from the DVC remote
```

Airflow 3 lives in its own virtualenv (its pins conflict with MLflow/DVC):

```bash
uv venv --python 3.12 ~/airflow-venv && source ~/airflow-venv/bin/activate
uv pip install "apache-airflow==3.1.8" apache-airflow-providers-standard \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.8/constraints-3.12.txt"
```

Kubernetes runs locally with Docker + [kind](https://kind.sigs.k8s.io/) + kubectl.

## 2. Environment configuration

All settings come from environment variables / `.env` and are validated at startup (`agent/config.py`). See `.env.example` for the full list: LLM (`GEMINI_API_KEY`, `LLM_MODEL`), `MLFLOW_TRACKING_URI`, Airflow URL and credentials, `K8S_NAMESPACE` / `K8S_ALLOWED_NAMESPACES`, agent budgets and the retraining freshness policy `MODEL_MAX_AGE_DAYS`.

## 3. Launching each service

| Service | Command (own terminal) | URL |
|---|---|---|
| MLflow | `scripts/start_mlflow.sh` | http://127.0.0.1:5000 |
| Airflow 3 | `scripts/start_airflow.sh` | http://127.0.0.1:8080 |
| Kubernetes | `kind create cluster --name mlops` then the deploy steps below | - |
| Agent API | `uvicorn api.main:app --host 127.0.0.1 --port 8000` | http://127.0.0.1:8000/docs |

Deploy the prediction API (model exported from MLflow, with its provenance):

```bash
python training/export_model.py          # best run of the Airflow pipeline that passed the gate
scripts/k8s_deploy.sh 1.0.0              # docker build, kind load, kubectl apply, rollout status
scripts/agent_kubeconfig.sh              # optional: restricted ServiceAccount kubeconfig for the agent
```

## 4. Running the tests

```bash
uv run ruff check .
uv run pytest -q                          # 49 tests; no services or API key needed
```

CI (`.github/workflows/ci.yml`) runs lint + tests on every push and pull request, then builds and smoke-tests the inference image.

## 5. Talking to the agent

CLI:

```bash
python -m agent.main                                         # interactive
python -m agent.main "Why is my prediction API unavailable?"
python -m agent.main "..." --save evidence/my_question.json  # answer + raw tool evidence
scripts/run_conversations.sh                                 # the five demo scenarios
```

HTTP:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "Compare the last five training runs and recommend the best candidate based on F1 score.", "include_evidence": true}'
curl -s localhost:8000/actions                                         # pending approvals
curl -s -X POST localhost:8000/actions/<action_id>/confirm -H 'content-type: application/json' -d '{"approve": true}'
```

State-changing operations (trigger a DAG, restart/scale a Deployment) are never executed by the model: they come back as `pending_actions` and run only after explicit approval.

Direct tool demos (no LLM): `python -m tests.demo_tools {mlflow|airflow|airflow-fail|k8s}`.

## Failure injection

```bash
python -m tests.demo_tools airflow-fail          # quality gate fails (min_f1=0.999)
scripts/inject_failure.sh bad-model-path         # Pods never Ready, Service has no endpoints
scripts/inject_failure.sh bad-image              # ImagePullBackOff, rollout stalls
scripts/inject_failure.sh restore
```

## Security

No API keys, kubeconfigs or tokens are committed. Secrets are supplied via environment variables, redacted from logs and tool output, and `.gitignore` blocks common credential files. The agent's Kubernetes identity can be restricted to one namespace with `kubernetes/agent-rbac.yaml`.

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

### Push 5: Training script and MLflow experiment tracking (Task 2 foundation)
- Added `training/train.py`: trains a Random Forest on the DVC-tracked dataset and logs parameters, metrics (accuracy, macro-F1) and the model to MLflow
- Every run records provenance as MLflow tags: DVC content hash, dataset version tag, the Git commit that introduced that dataset version, current Git commit, a dirty-tree flag, and whether the CSV on disk still matches the DVC pointer
- Fixed model logging for MLflow 3.x by trusting only the scikit-learn `Tree` type when saving the model (no blanket trust)
- Trained three configurations on each dataset version (500 and 1000 rows), producing six tracked runs
- Evidence: `screenshots/02_mlflow_runs_comparison.png`, `screenshots/03_mlflow_run_provenance.png`


### Push 6: Tool contract, registry, DVC tools and MLflow tools (Tasks 1 and 2)
- Added `agent/schemas.py`: structured tool inputs and outputs (`ToolResult` with status, error, duration and request ID) and the chat response model that separates facts, uncertainties and recommended actions
- Added the tool registry in `agent/tools/__init__.py`: argument validation, per-call timeouts, exception mapping (unavailable, timeout, invalid, forbidden), confirmation enforcement for state-changing tools, and a structured log line for every call
- Added 7 read-only DVC tools (`agent/tools/dvc_tools.py`): list datasets, current version, version history, compare versions, detect changes, reproduction instructions, and hash-to-version lookup. Only allow-listed git/dvc subcommands run, without a shell, and refs and paths are validated
- Added 7 read-only MLflow tools (`agent/tools/mlflow_tools.py`): experiments, runs, run details (including logged models), best run, run comparison, experiment comparison, and registry info. Rankings always carry caveats and are never treated as a production decision
- Added `tests/demo_tools.py` to run real tools through the registry: `python -m tests.demo_tools mlflow`

### Push 7: Airflow training pipeline and Airflow tools (Task 3)
- Added `dags/training_pipeline.py`: validate -> preprocess -> train -> evaluate/log to MLflow -> quality gate, with retries for infrastructure-prone stages, no retries for deterministic failures, and run parameters (`min_f1`, `n_estimators`, `max_depth`)
- Added `training/pipeline_steps.py`: each stage prints one JSON line and fails the task on error; the gate tags the MLflow run `quality_gate=passed|failed`
- Added 7 Airflow tools using the Airflow 3 REST API with JWT auth; only `training_pipeline` may be triggered and only after user confirmation
- MLflow calls in the pipeline now fail fast (2 retries, 30 s timeout, socket preflight) and `log_model` uses explicit requirements, removing minutes of environment inference per run

### Push 8: Kubernetes deployment and Kubernetes tools (Task 4)
- Added the inference service (`serving/app.py`), `Dockerfile`, and manifests: Namespace, ConfigMap, Deployment (2 replicas, probes, requests/limits, non-root), ClusterIP Service, namespaced RBAC for the agent
- Added `training/export_model.py` so the deployed model carries its MLflow run id and DVC dataset provenance (`/version`)
- Added 10 Kubernetes tools; restart/scale need confirmation and every tool is restricted to approved namespaces
- Added failure injection scripts for diagnosis demos

### Push 9: Multi-tool agent, API and CLI (Integration)
- Added the Gemini adapter (`agent/llm.py`), the orchestrator with evidence guard and pending-action approvals, the FastAPI interface and the CLI

### Push 10: Tests and CI
- 49 pytest tests with mocked DVC/MLflow/Airflow/Kubernetes/LLM; GitHub Actions runs ruff, pytest and an inference-image smoke test
