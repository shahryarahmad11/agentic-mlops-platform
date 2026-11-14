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
| Kubernetes (`agent/tools/kubernetes_tools.py`) | 9 read-only + 2 approval-gated | Deployments, Pods, logs, Services/endpoints, events, unhealthy Pods/restarts, deployed version, wait for rollout, full diagnosis (incl. ConfigMaps and error log lines); restart/scale only in approved namespaces |
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
├── evidence/         # saved agent conversations (answer + raw tool output) and captured terminal output
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

Kubernetes runs locally with Docker + [kind](https://kind.sigs.k8s.io/) + kubectl:

```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER   # then open a new shell
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && sudo install kubectl /usr/local/bin/
curl -Lo kind https://kind.sigs.k8s.io/dl/v0.29.0/kind-linux-amd64 && sudo install kind /usr/local/bin/
kind create cluster --name mlops
```

On a small WSL machine (about 4 GB RAM) running MLflow, Airflow and kind together needs extra swap (a 6 GB swapfile was used).

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
scripts/agent_kubeconfig.sh              # restricted ServiceAccount kubeconfig for the agent
echo "KUBECONFIG=$HOME/.kube/mlops-agent.kubeconfig" >> .env
```

## 4. Running the tests

```bash
uv run ruff check .
uv run pytest -q                          # 50 tests; no services or API key needed
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

## Results (what was actually run)

| Scenario | Outcome | Evidence |
|---|---|---|
| Dataset provenance | Best run `rf-v2-n200-d10` traced to `data-v2` / commit `c57fc80`, with reproduction steps | `evidence/conversation_1_*.json`, `screenshots/15_*` |
| Model selection | Last 5 runs compared; `rf-v2-n200-d10` (F1 0.985) recommended with caveats (not production-ready, different test splits) | `conversation_2_*`, `screenshots/16_*` |
| Pipeline execution | Trigger proposed, approved by the user, run waited on: all 5 tasks succeeded, gate passed | `conversation_3_*`, `screenshots/04-06, 17_*` |
| Deployment diagnosis | Injected bad `MODEL_PATH`: agent found 0 ready endpoints, the model-load error and the ConfigMap value | `conversation_4_*`, `screenshots/18_*, 20a-b_*` |
| Multi-tool reasoning | MLflow + DVC + Kubernetes + Airflow correlated; retraining not indicated, nothing triggered | `conversation_5_*`, `screenshots/19_*` |
| Failure 2 | Dataset edited outside DVC: `validate_dataset` failed, agent combined Airflow logs + DVC status, fixed with `dvc checkout`, re-run passed | `failure2_*`, `screenshots/20c_*` |
| Quality gate | `min_f1=0.999` run failed at `quality_gate`, MLflow run tagged `quality_gate=failed` | `screenshots/07-09` |
| LLM unavailable | Gemini quota exhausted (HTTP 429): agent returned no facts, reported the reason | `llm_quota_exhausted_*`, `screenshots/20d_*` |

Conversations 1-3 used `gemini-3.5-flash`; after its free-tier daily limit (20 requests/day/model) was reached, the rest used `gemini-3.5-flash-lite` (set `LLM_MODEL` in `.env`).

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

### Push 11: End-to-end execution, failure investigations and evidence
- Installed Docker, kubectl and kind; deployed the prediction API (model `a4ade6fe`, gated pipeline run on `data-v2`) and connected the agent through the restricted `mlops-agent` ServiceAccount
- Diagnosed and fixed an unexpected pipeline slowdown (MLflow retry defaults + WSL mirrored networking + requirement inference): DAG runs went from 15+ minutes to under 90 seconds
- Ran the five demo conversations and two injected failures; saved answers and raw tool output in `evidence/` and screenshots in `screenshots/`
- Improvements found during the investigations: the diagnosis tool now returns ConfigMaps and error log lines first, the service no longer logs probe requests (image 1.0.1), and new wait tools (`airflow_wait_for_run`, `k8s_wait_for_rollout`) let the agent check outcomes only after they are final
- Docker build made robust to slow networks (pip timeouts, retries, cache mount); `dvc repro` stage recorded in `dvc.lock`; 50 tests
- Filled the remaining sections of the technical report

