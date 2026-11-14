"""Export a model from MLflow into serving/model/ so it can be baked into the inference image.

Writes model.joblib plus metadata.json (MLflow run, metrics and DVC dataset provenance), which
the inference service reports on /version. That is how the agent links what is deployed on
Kubernetes back to MLflow and DVC.

Usage:
  python training/export_model.py --run-id <id>
  python training/export_model.py                       # best gated run of the Airflow pipeline
  python training/export_model.py --experiment mlops-agent-training --no-require-gate
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

OUT_DIR = Path("serving/model")
METRIC = "f1_macro"
PROVENANCE = (
    "dataset_version",
    "dvc_md5",
    "dataset_git_commit",
    "git_commit",
    "quality_gate",
    "airflow_run_id",
)


def pick_run(client: MlflowClient, experiment: str, require_gate: bool):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment not found: {experiment}")
    query = "attributes.status = 'FINISHED'"
    if require_gate:
        query += " and tags.quality_gate = 'passed'"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=query,
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No eligible runs in {experiment} (require_gate={require_gate})")
    return runs[0]


def model_uri(client: MlflowClient, run) -> str:
    models = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        max_results=1,
    )
    return models[0].model_uri if models else f"runs:/{run.info.run_id}/model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--experiment", default="airflow-training-pipeline")
    parser.add_argument("--no-require-gate", action="store_true")
    args = parser.parse_args()

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    run = (
        client.get_run(args.run_id)
        if args.run_id
        else pick_run(client, args.experiment, not args.no_require_gate)
    )
    source = model_uri(client, run)
    model = mlflow.sklearn.load_model(source)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.joblib")
    tags = run.data.tags
    metadata = {
        "mlflow_run_id": run.info.run_id,
        "mlflow_run_name": run.info.run_name,
        "mlflow_experiment_id": run.info.experiment_id,
        "model_uri": source,
        "metrics": {k: round(v, 6) for k, v in run.data.metrics.items()},
        "params": dict(run.data.params),
        **{key: tags.get(key) for key in PROVENANCE},
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
