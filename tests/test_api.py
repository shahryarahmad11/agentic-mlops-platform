"""FastAPI interface with a fake LLM behind the orchestrator."""

import pytest
from fastapi.testclient import TestClient

from agent.orchestrator import Orchestrator
from api.main import app, get_orchestrator
from tests.conftest import ScriptedLLM, calls, final
from tests.test_orchestrator import answer, fake_registry


@pytest.fixture
def api():
    reg, executed = fake_registry()
    llm = ScriptedLLM(
        calls(("airflow_trigger_dag", {"dag_id": "training_pipeline"})),
        final(answer([], text="Trigger awaiting approval")),
    )
    orch = Orchestrator(llm, reg)
    app.dependency_overrides[get_orchestrator] = lambda: orch
    with TestClient(app) as test_client:
        yield test_client, executed
    app.dependency_overrides.clear()


def test_health_and_tool_permissions(api):
    client, _ = api
    assert client.get("/health").json()["status"] == "ok"
    tools = {t["name"]: t for t in client.get("/tools").json()}
    assert tools["airflow_trigger_dag"]["requires_confirmation"] is True
    assert tools["mlflow_best_run"]["read_only"] is True


def test_invalid_chat_request_is_rejected(api):
    client, _ = api
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_then_explicit_confirmation(api):
    client, executed = api
    response = client.post(
        "/chat", json={"message": "Trigger training"}, headers={"X-Request-ID": "req-test-1"}
    )
    body = response.json()
    assert response.headers["X-Request-ID"] == "req-test-1" and body["request_id"] == "req-test-1"
    action_id = body["pending_actions"][0]["action_id"]
    assert executed == []
    assert client.post("/actions/act-nope/confirm", json={"approve": True}).status_code == 404
    confirmed = client.post(f"/actions/{action_id}/confirm", json={"approve": True}).json()
    assert confirmed["result"]["status"] == "success" and executed == ["airflow_trigger_dag"]
