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
