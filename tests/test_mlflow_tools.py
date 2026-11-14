"""MLflow tools with a mocked MlflowClient."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from agent.schemas import ToolStatus
from agent.tools import mlflow_tools as mt


def run(name, f1, version):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=f"id-{name}", run_name=name, status="FINISHED", start_time=0),
        data=SimpleNamespace(
            params={"n_estimators": "100"},
            metrics={"f1_macro": f1, "accuracy": f1},
            tags={
                "dataset_version": version,
                "dvc_md5": "a" * 32,
                "git_dirty": "false",
                "data_matches_pointer": "true",
                "quality_gate": "passed",
            },
        ),
    )


@pytest.fixture
def client(monkeypatch):
    fake = MagicMock()
    fake.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="1", name="mlops-agent-training"
    )
    fake.search_runs.return_value = [
        run("rf-v1", 0.92, "data-v1"),
        run("rf-v2-a", 0.985, "data-v2"),
        run("rf-v2-b", 0.983, "data-v2"),
    ]
    monkeypatch.setattr(mt, "_client", lambda: fake)
    return fake


def test_best_run_ranks_by_metric_with_caveats(registry, client):
    result = registry.execute("mlflow_best_run", {"experiment_name": "mlops-agent-training"})
    data = result.data
    assert result.ok and data["best"]["run_name"] == "rf-v2-a"
    assert data["best"]["provenance"]["quality_gate"] == "passed"
    caveats = " ".join(data["caveats"])
    assert "tie" in caveats  # 0.985 vs 0.983
    assert "not strictly comparable" in caveats  # mixed dataset versions
    assert "not a promotion decision" in data["decision"]


def test_best_run_filtered_by_dataset_version(registry, client):
    result = registry.execute(
        "mlflow_best_run", {"experiment_name": "mlops-agent-training", "dataset_version": "data-v1"}
    )
    assert result.data["best"]["run_name"] == "rf-v1"


def test_metric_injection_is_rejected(registry, client):
    result = registry.execute(
        "mlflow_best_run", {"experiment_name": "mlops-agent-training", "metric": "f1; DROP"}
    )
    assert result.status == ToolStatus.INVALID_REQUEST
    client.search_runs.assert_not_called()


def test_tracking_server_down_is_unavailable(registry, client):
    client.get_experiment_by_name.side_effect = requests.exceptions.ConnectionError("refused")
    result = registry.execute("mlflow_list_runs", {"experiment_name": "mlops-agent-training"})
    assert result.status == ToolStatus.UNAVAILABLE and result.data is None
