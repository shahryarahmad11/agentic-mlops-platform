"""Structured inputs and outputs shared by all tools, the agent and the API."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolSource(StrEnum):
    DVC = "dvc"
    MLFLOW = "mlflow"
    AIRFLOW = "airflow"
    KUBERNETES = "kubernetes"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"  # the system answered, but the operation failed
    UNAVAILABLE = "unavailable"  # the system could not be reached
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"  # bad or disallowed input, rejected before any call
    FORBIDDEN = "forbidden"  # operation or namespace not permitted
    CONFIRMATION_REQUIRED = "confirmation_required"  # state change awaiting explicit approval


class ToolError(BaseModel):
    type: str
    message: str
    retryable: bool = False


class ToolSpec(BaseModel):
    """Declares a tool and its permissions; the agent only ever sees registered specs."""

    name: str
    description: str
    source: ToolSource
    read_only: bool = True
    requires_confirmation: bool = False
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolResult(BaseModel):
    """What every tool returns. Operational claims must cite one of these."""

    tool: str
    source: ToolSource | None = None  # None only for unknown tool names
    status: ToolStatus
    request_id: str | None = None
    duration_ms: float = 0.0
    data: Any = None
    error: ToolError | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @classmethod
    def success(cls, tool: str, source: ToolSource | None, data: Any, **extra: Any) -> "ToolResult":
        return cls(tool=tool, source=source, status=ToolStatus.SUCCESS, data=data, **extra)

    @classmethod
    def failure(
        cls,
        tool: str,
        source: ToolSource | None,
        status: ToolStatus,
        error_type: str,
        message: str,
        retryable: bool = False,
        **extra: Any,
    ) -> "ToolResult":
        error = ToolError(type=error_type, message=message, retryable=retryable)
        return cls(tool=tool, source=source, status=status, error=error, **extra)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ToolCallRecord(BaseModel):
    """Compact trace entry returned with each answer (never contains secrets)."""

    tool: str
    source: ToolSource | None = None
    status: ToolStatus
    duration_ms: float
    request_id: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    facts: list[str] = Field(default_factory=list)  # each backed by a tool result
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
