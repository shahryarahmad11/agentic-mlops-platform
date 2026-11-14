"""Tool registry: validation, timeouts, failure mapping, confirmation and structured logs."""

import json
import logging
import time

import httpx

from agent.logging_config import JsonFormatter
from agent.schemas import ToolSource, ToolSpec, ToolStatus
from agent.tools import ToolInputError, ToolPermissionError, ToolRegistry


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    params = {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "name": {"type": "string"}},
        "required": ["n"],
    }

    @reg.register(ToolSpec(name="echo", description="", source=ToolSource.DVC, parameters=params))
    def echo(n: int, name: str = "x"):
        return {"n": n, "name": name}

    @reg.register(ToolSpec(name="slow", description="", source=ToolSource.MLFLOW))
    def slow():
        time.sleep(2)

    @reg.register(ToolSpec(name="down", description="", source=ToolSource.AIRFLOW))
    def down():
        raise httpx.ConnectError("connection refused")

    @reg.register(ToolSpec(name="bad_input", description="", source=ToolSource.DVC))
    def bad_input():
        raise ToolInputError("nope")

    @reg.register(ToolSpec(name="denied", description="", source=ToolSource.KUBERNETES))
    def denied():
        raise ToolPermissionError("namespace not approved")

    @reg.register(ToolSpec(name="boom", description="", source=ToolSource.DVC))
    def boom():
        raise RuntimeError("password=hunter2 leaked in message")

    @reg.register(
        ToolSpec(
            name="change",
            description="",
            source=ToolSource.KUBERNETES,
            read_only=False,
            requires_confirmation=True,
        )
    )
    def change():
        return {"changed": True}

    return reg


def test_successful_execution_returns_structured_result():
    result = make_registry().execute("echo", {"n": 3}, request_id="req-1")
    assert result.ok and result.data == {"n": 3, "name": "x"}
    assert result.source == ToolSource.DVC and result.request_id == "req-1"
    assert result.duration_ms >= 0


def test_invalid_requests_are_rejected_before_running():
    reg = make_registry()
    assert reg.execute("nope").status == ToolStatus.INVALID_REQUEST
    assert reg.execute("echo", {}).error.message.startswith("Missing required")
    assert reg.execute("echo", {"n": "3"}).status == ToolStatus.INVALID_REQUEST
    assert reg.execute("echo", {"n": True}).status == ToolStatus.INVALID_REQUEST  # bool != int
    assert reg.execute("echo", {"n": 1, "evil": 1}).error.message.startswith("Unknown argument")
    assert reg.execute("bad_input").status == ToolStatus.INVALID_REQUEST


def test_timeout_is_reported_not_hung():
    started = time.perf_counter()
    result = make_registry().execute("slow", timeout=0.2)
    assert result.status == ToolStatus.TIMEOUT and result.error.retryable
    assert time.perf_counter() - started < 1.5


def test_service_failures_are_mapped():
    reg = make_registry()
    down = reg.execute("down")
    assert down.status == ToolStatus.UNAVAILABLE and down.data is None
    assert reg.execute("denied").status == ToolStatus.FORBIDDEN
    boom = reg.execute("boom")
    assert boom.status == ToolStatus.ERROR
    assert "hunter2" not in boom.error.message  # secrets redacted from error text


def test_state_changing_tool_requires_confirmation():
    reg = make_registry()
    blocked = reg.execute("change")
    assert blocked.status == ToolStatus.CONFIRMATION_REQUIRED and blocked.data is None
    assert reg.execute("change", confirmed=True).data == {"changed": True}


def test_structured_log_has_tool_status_duration_request_id_and_no_secrets(caplog):
    reg = make_registry()
    with caplog.at_level(logging.INFO, logger="agent.tools"):
        reg.execute("echo", {"n": 1}, request_id="req-abc")
        reg.execute("boom", request_id="req-abc")
    formatter = JsonFormatter("test")
    lines = [json.loads(formatter.format(r)) for r in caplog.records]
    assert {line["tool_name"] for line in lines} == {"echo", "boom"}
    for line in lines:
        assert line["request_id"] == "req-abc"
        assert line["status"] in {"success", "error"}
        assert isinstance(line["duration_ms"], float)
        assert line["event"] == "tool_call"
    assert "hunter2" not in json.dumps(lines)


def test_redaction_of_keys_and_tokens():
    record = logging.LogRecord(
        "x",
        logging.INFO,
        "",
        0,
        "key AIzaSyA1234567890abcdefghijklmnop token=abc Bearer eyJhbGciOi",
        None,
        None,
    )
    text = JsonFormatter().format(record)
    assert "AIzaSy" not in text and "eyJhbGciOi" not in text and "token=abc" not in text
