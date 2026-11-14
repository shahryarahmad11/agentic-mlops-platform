"""Pipeline stages for the Airflow training DAG (run with the project's own Python).

Usage: python training/pipeline_steps.py {validate|preprocess|train|evaluate|gate} [options]
Every stage prints one JSON line and exits non-zero on failure, so Airflow marks the task failed.
"""

import argparse
import json
import os
import shutil
import socket
import sys
import time
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from train import DATA_PATH, DVC_POINTER, _file_md5, dataset_provenance

# Fail fast instead of MLflow's default 7 retries x 120 s when the server is down.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")

ARTIFACT_DIR = Path(os.getenv("PIPELINE_ARTIFACT_DIR", "pipeline_artifacts/current"))
EXPERIMENT = "airflow-training-pipeline"
FEATURES = [f"feature_{i}" for i in range(5)]
EXPECTED_COLUMNS = [*FEATURES, "target"]
MIN_ROWS = 100
MIN_CLASS_SHARE = 0.2


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _require_mlflow(stage: str) -> None:
    """Preflight: a clear failure in seconds if the tracking server is not reachable."""
    parsed = urlparse(_tracking_uri())
    if parsed.scheme not in ("http", "https"):
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((parsed.hostname, port), timeout=5).close()
    except OSError as exc:
        _fail(stage, f"MLflow tracking server not reachable at {_tracking_uri()}: {exc}")


# Explicit requirements skip MLflow's slow environment inference (minutes on small machines).
PIP_REQUIREMENTS = [f"{pkg}=={version(pkg)}" for pkg in ("mlflow", "scikit-learn", "pandas")]


def _emit(stage: str, status: str, **details: object) -> None:
    print(json.dumps({"stage": stage, "status": status, **details}, default=str), flush=True)


def _fail(stage: str, message: str, **details: object) -> None:
    _emit(stage, "failed", message=message, **details)
    sys.exit(1)


def _write(name: str, payload: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


def _read(name: str, stage: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.is_file():
        _fail(stage, f"missing input from the previous stage: {name}")
    return json.loads(path.read_text())


def validate(_args: argparse.Namespace) -> None:
    shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)  # no stale results from earlier runs
    if not DATA_PATH.is_file():
        _fail("validate", f"dataset not found: {DATA_PATH}")
    recorded = str(yaml.safe_load(DVC_POINTER.read_text())["outs"][0]["md5"])
    actual = _file_md5(DATA_PATH)
    frame = pd.read_csv(DATA_PATH)

    problems = []
    if actual != recorded:
        problems.append("dataset content does not match the hash recorded in the DVC pointer")
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if len(frame) < MIN_ROWS:
        problems.append(f"only {len(frame)} rows; at least {MIN_ROWS} required")
    if not missing:
        nulls = int(frame[EXPECTED_COLUMNS].isna().sum().sum())
        if nulls:
            problems.append(f"{nulls} missing values found")
        if not set(frame["target"].dropna().unique()) <= {0, 1}:
            problems.append("target must contain only 0 and 1")
        share = float(frame["target"].value_counts(normalize=True).min())
        if share < MIN_CLASS_SHARE:
            problems.append(f"smallest class is only {share:.0%} of the rows")

    _write(
        "validation.json",
        {"rows": len(frame), "recorded_md5": recorded, "actual_md5": actual, "problems": problems},
    )
    if problems:
        _fail("validate", "; ".join(problems), recorded_md5=recorded, actual_md5=actual)
    _emit("validate", "passed", rows=len(frame), md5=actual)


def preprocess(args: argparse.Namespace) -> None:
    _read("validation.json", "preprocess")  # proves validation ran in this pipeline run
    frame = pd.read_csv(DATA_PATH)
    before = len(frame)
    frame = frame.drop_duplicates().dropna().reset_index(drop=True)
    train_df, test_df = train_test_split(
        frame, test_size=args.test_size, random_state=args.seed, stratify=frame["target"]
    )
    train_df.to_csv(ARTIFACT_DIR / "train_split.csv", index=False)
    test_df.to_csv(ARTIFACT_DIR / "test_split.csv", index=False)
    summary = {
        "rows_before": before,
        "rows_after": len(frame),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": args.test_size,
        "seed": args.seed,
    }
    _write("preprocess.json", summary)
    _emit("preprocess", "passed", **summary)


def train(args: argparse.Namespace) -> None:
    _read("preprocess.json", "train")
    if args.n_estimators < 1 or args.max_depth < 1:
        _fail("train", "n-estimators and max-depth must be positive integers")
    frame = pd.read_csv(ARTIFACT_DIR / "train_split.csv")
    started = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed
    )
    model.fit(frame[FEATURES], frame["target"])
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    info = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "seed": args.seed,
        "train_rows": len(frame),
        "fit_seconds": round(time.perf_counter() - started, 2),
    }
    _write("train.json", info)
    _emit("train", "passed", **info)


def evaluate(args: argparse.Namespace) -> None:
    info = _read("train.json", "evaluate")
    prep = _read("preprocess.json", "evaluate")
    model = joblib.load(ARTIFACT_DIR / "model.joblib")  # produced by the previous stage
    test = pd.read_csv(ARTIFACT_DIR / "test_split.csv")
    x_test, y_test = test[FEATURES], test["target"]
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }

    _require_mlflow("evaluate")
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    tags = dataset_provenance() | {
        "airflow_dag_id": args.airflow_dag_id,
        "airflow_run_id": args.airflow_run_id,
        "pipeline": "training_pipeline",
    }
    run_name = f"airflow-rf-n{info['n_estimators']}-d{info['max_depth']}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": info["n_estimators"],
                "max_depth": info["max_depth"],
                "seed": info["seed"],
                "test_size": prep["test_size"],
                "n_rows": prep["rows_after"],
            }
        )
        mlflow.set_tags(tags)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_test.iloc[:2],
            pip_requirements=PIP_REQUIREMENTS,
            skops_trusted_types=["sklearn.tree._tree.Tree"],  # model trained by the prior stage
        )
    result = {
        "run_id": run.info.run_id,
        "experiment": EXPERIMENT,
        **metrics,
        "dataset_version": tags["dataset_version"],
        "data_matches_pointer": tags["data_matches_pointer"],
    }
    _write("evaluation.json", result)
    _emit("evaluate", "passed", **result)


def gate(args: argparse.Namespace) -> None:
    if not 0 <= args.min_f1 <= 1:
        _fail("gate", "min-f1 must be between 0 and 1")
    evaluation = _read("evaluation.json", "gate")
    reasons = []
    if evaluation["f1_macro"] < args.min_f1:
        reasons.append(
            f"f1_macro {evaluation['f1_macro']:.4f} is below the required minimum {args.min_f1}"
        )
    if evaluation["data_matches_pointer"] != "true":
        reasons.append("training data did not match its DVC pointer")
    passed = not reasons

    _require_mlflow("gate")
    client = MlflowClient(tracking_uri=_tracking_uri())
    client.set_tag(evaluation["run_id"], "quality_gate", "passed" if passed else "failed")
    client.set_tag(evaluation["run_id"], "quality_gate_min_f1", str(args.min_f1))
    _write("gate.json", {"passed": passed, "reasons": reasons, "min_f1": args.min_f1})
    if not passed:
        _fail(
            "gate",
            "; ".join(reasons),
            run_id=evaluation["run_id"],
            f1_macro=evaluation["f1_macro"],
            min_f1=args.min_f1,
        )
    _emit(
        "gate",
        "passed",
        run_id=evaluation["run_id"],
        f1_macro=evaluation["f1_macro"],
        min_f1=args.min_f1,
        note="Meets the quality gate; promotion would still need review",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    stages.add_parser("validate").set_defaults(func=validate)

    pre = stages.add_parser("preprocess")
    pre.add_argument("--seed", type=int, default=42)
    pre.add_argument("--test-size", type=float, default=0.2)
    pre.set_defaults(func=preprocess)

    fit = stages.add_parser("train")
    fit.add_argument("--n-estimators", type=int, default=100)
    fit.add_argument("--max-depth", type=int, default=5)
    fit.add_argument("--seed", type=int, default=42)
    fit.set_defaults(func=train)

    ev = stages.add_parser("evaluate")
    ev.add_argument("--airflow-dag-id", default="manual")
    ev.add_argument("--airflow-run-id", default="manual")
    ev.set_defaults(func=evaluate)

    gt = stages.add_parser("gate")
    gt.add_argument("--min-f1", type=float, default=0.90)
    gt.set_defaults(func=gate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()