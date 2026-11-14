"""Airflow tools against a mocked Airflow 3 REST API (respx)."""

import json

import httpx
import pytest
import respx

from agent.schemas import ToolStatus

BASE = "http://airflow.test"
API = f"{BASE}/api/v2"


@pytest.fixture
def airflow(monkeypatch):
    monkeypatch.setenv("AIRFLOW_BASE_URL", BASE)
    monkeypatch.setenv("AIRFLOW_USERNAME", "agent")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "test-password")
    with respx.mock(assert_all_called=False) as mock:
        mock.post(f"{BASE}/auth/token").respond(json={"access_token": "jwt-token"})
        yield mock


def test_list_dags_marks_approved_dags(registry, airflow):
    route = airflow.get(f"{API}/dags").respond(
        json={
            "total_entries": 2,
            "dags": [
                {"dag_id": "training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]},
                {"dag_id": "example_bash_operator", "is_paused": True, "tags": []},
            ],
        }
    )
    result = registry.execute("airflow_list_dags")
    assert result.ok
    approved = {d["dag_id"]: d["approved_for_trigger"] for d in result.data["dags"]}
    assert approved == {"training_pipeline": True, "example_bash_operator": False}
    assert route.calls[0].request.headers["Authorization"] == "Bearer jwt-token"


def test_trigger_of_unapproved_dag_is_forbidden_and_sends_nothing(registry, airflow):
    post = airflow.post(f"{API}/dags/example_bash_operator/dagRuns").respond(json={})
    result = registry.execute(
        "airflow_trigger_dag", {"dag_id": "example_bash_operator"}, confirmed=True
    )
    assert result.status == ToolStatus.FORBIDDEN and not post.called


def test_confirmed_trigger_sends_validated_conf(registry, airflow):
    airflow.get(f"{API}/dags/training_pipeline").respond(json={"is_paused": False})
    post = airflow.post(f"{API}/dags/training_pipeline/dagRuns").respond(
        json={"dag_run_id": "manual__1", "state": "queued"}
    )
    result = registry.execute(
        "airflow_trigger_dag", {"dag_id": "training_pipeline", "min_f1": 0.95}, confirmed=True
    )
    assert result.ok and result.data["dag_run_id"] == "manual__1"
    assert json.loads(post.calls[0].request.content)["conf"] == {"min_f1": 0.95}


def test_out_of_range_parameters_are_rejected(registry, airflow):
    result = registry.execute(
        "airflow_trigger_dag", {"dag_id": "training_pipeline", "min_f1": 5}, confirmed=True
    )
    assert result.status == ToolStatus.INVALID_REQUEST


def test_failed_tasks_include_log_diagnostics(registry, airflow):
    run = f"{API}/dags/training_pipeline/dagRuns/run1"
    airflow.get(run).respond(json={"dag_run_id": "run1", "state": "failed"})
    airflow.get(f"{run}/taskInstances").respond(
        json={
            "task_instances": [
                {
                    "task_id": "validate_dataset",
                    "state": "success",
                    "try_number": 1,
                    "start_date": "1",
                },
                {"task_id": "quality_gate", "state": "failed", "try_number": 1, "start_date": "2"},
            ]
        }
    )
    airflow.get(f"{run}/taskInstances/quality_gate/logs/1").respond(
        json={
            "content": [
                {
                    "event": '{"stage": "gate", "status": "failed", "message": "f1_macro 0.97 is below 0.999"}'
                }
            ]
        }
    )
    result = registry.execute(
        "airflow_failed_tasks", {"dag_id": "training_pipeline", "dag_run_id": "run1"}
    )
    assert result.ok and result.data["first_failed_task"] == "quality_gate"
    assert "below 0.999" in result.data["failed_tasks"][0]["log_tail"][0]


def test_airflow_down_is_unavailable(registry, airflow):
    airflow.get(f"{API}/dags").mock(side_effect=httpx.ConnectError("refused"))
    result = registry.execute("airflow_list_dags")
    assert result.status == ToolStatus.UNAVAILABLE and result.error.retryable


def test_bad_credentials_are_forbidden(registry, airflow):
    airflow.post(f"{BASE}/auth/token").respond(status_code=401)
    assert registry.execute("airflow_list_dags").status == ToolStatus.FORBIDDEN


def test_path_traversal_in_run_id_is_rejected(registry, airflow):
    result = registry.execute(
        "airflow_get_run_details", {"dag_id": "training_pipeline", "dag_run_id": "../../x"}
    )
    assert result.status == ToolStatus.INVALID_REQUEST
