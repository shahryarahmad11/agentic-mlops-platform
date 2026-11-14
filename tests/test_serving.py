"""Inference service: probes and predictions (the app deployed on Kubernetes)."""

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier

from serving import app as serving


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    features = [f"feature_{i}" for i in range(5)]
    frame = pd.DataFrame([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], columns=features)
    joblib.dump(DecisionTreeClassifier().fit(frame, [0, 1]), tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({"mlflow_run_id": "r1"}))
    monkeypatch.setattr(serving, "METADATA_PATH", tmp_path / "metadata.json")
    return tmp_path


def test_ready_and_predicts_with_a_model(model_files, monkeypatch):
    monkeypatch.setattr(serving, "MODEL_PATH", model_files / "model.joblib")
    with TestClient(serving.app) as client:
        assert client.get("/readyz").status_code == 200
        assert client.get("/version").json()["model"]["mlflow_run_id"] == "r1"
        out = client.post("/predict", json={"instances": [[1, 1, 1, 1, 1]]}).json()
        assert out["predictions"] == [1]
        assert client.post("/predict", json={"instances": [[1, 2]]}).status_code == 422


def test_missing_model_is_alive_but_not_ready(model_files, monkeypatch):
    monkeypatch.setattr(serving, "MODEL_PATH", model_files / "missing.joblib")
    with TestClient(serving.app) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503 and "FileNotFoundError" in ready.json()["detail"]["reason"]
        assert client.post("/predict", json={"instances": [[0, 0, 0, 0, 0]]}).status_code == 503
