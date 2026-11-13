"""Structured JSON logging with request IDs and secret redaction."""

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import UTC, datetime

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

STRUCTURED_FIELDS = (
    "event", "environment", "request_id", "tool_name", "status", "duration_ms", "error_type",
)

_REDACTIONS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "***REDACTED***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer ***REDACTED***"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b(\s*[=:]\s*)\S+"),
        r"\1\2***REDACTED***",
    ),
]


def new_request_id() -> str:
    """Create a request ID and bind it to the current context."""
    rid = f"req-{uuid.uuid4().hex[:12]}"
    _request_id.set(rid)
    return rid


def set_request_id(value: str | None) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    def __init__(self, environment: str | None = None):
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if self.environment:
            payload["environment"] = self.environment
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            payload["request_id"] = request_id
        for field in STRUCTURED_FIELDS:
            if field in ("request_id",) or not hasattr(record, field):
                continue
            value = getattr(record, field)
            payload[field] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", environment: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(environment))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
