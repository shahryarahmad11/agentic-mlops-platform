"""Train a classifier on the DVC-tracked dataset and log full provenance to MLflow."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/train.csv")
DVC_POINTER = Path("data/train.csv.dvc")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_md5(path: Path) -> str:
    # MD5 matches the hash DVC records; it is not used for security here.
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance() -> dict[str, str]:
    """Collect the recorded DVC reference and Git context for the dataset."""
    pointer = yaml.safe_load(DVC_POINTER.read_text())["outs"][0]
    # Oldest commit that introduced this exact DVC hash (correct for any checked-out version).
    introduced = _git(
        "log", "--reverse", f"-S{pointer['md5']}", "--format=%H", "--", str(DVC_POINTER)
    )
    pointer_commit = introduced.splitlines()[0] if introduced not in ("", "unknown") else ""
    tags = ""
    if pointer_commit not in ("", "unknown"):
        tags = _git("tag", "--points-at", pointer_commit, "--list", "data-v*")
    return {
        "dvc_file": str(DVC_POINTER),
        "dvc_md5": str(pointer["md5"]),
        "dataset_git_commit": pointer_commit or "unknown",
        "dataset_version": ",".join(tags.split()) or "untagged",
        "data_matches_pointer": str(_file_md5(DATA_PATH) == pointer["md5"]).lower(),
        "git_commit": _git("rev-parse", "HEAD"),
        # Data state is tracked separately via dvc_md5 / data_matches_pointer.
        "git_dirty": str(bool(_git("status", "--porcelain", "--", ".", ":(exclude)data"))).lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="mlops-agent-training")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.n_estimators < 1 or args.max_depth < 1:
        raise SystemExit("n-estimators and max-depth must be positive integers")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(args.experiment)

    frame = pd.read_csv(DATA_PATH)
    features, target = frame.drop(columns=["target"]), frame["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=args.seed, stratify=target
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "seed": args.seed,
                "test_size": 0.2,
                "n_rows": len(frame),
            }
        )
        mlflow.set_tags(dataset_provenance())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", input_example=x_test.iloc[:2])

    print(json.dumps({"run_id": run.info.run_id, "experiment": args.experiment, **metrics}))


if __name__ == "__main__":
    main()
