"""Tool registry: one safe execution path for every tool the agent can call."""

import importlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from agent.logging_config import get_logger, get_request_id, redact
from agent.schemas import ToolResult, ToolSource, ToolSpec, ToolStatus

logger = get_logger("agent.tools")

DEFAULT_TIMEOUT_SECONDS = 15.0
TOOL_MODULES = ("dvc_tools", "mlflow_tools", "airflow_tools", "kubernetes_tools")
_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolInputError(ValueError):
    """Raised by a tool when its input is invalid or not on the allow-list."""


class ToolPermissionError(Exception):
    """Raised by a tool when the operation or namespace is not permitted."""


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    schema = spec.parameters
    props = schema.get("properties", {})
    unknown = sorted(set(arguments) - set(props))
    if unknown:
        return f"Unknown argument(s): {', '.join(unknown)}"
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return f"Missing required argument(s): {', '.join(missing)}"
    for key, value in arguments.items():
        kind = props[key].get("type")
        expected = _JSON_TYPES.get(kind)
        if expected is None:
            continue
        if (kind != "boolean" and isinstance(value, bool)) or not isinstance(value, expected):
            return f"Argument '{key}' must be of type {kind}"
    return None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if spec.name in self._tools:
                raise ValueError(f"Tool already registered: {spec.name}")
            self._tools[spec.name] = (spec, func)
            return func

        return decorator

    def specs(self, source: ToolSource | None = None) -> list[ToolSpec]:
        found = [spec for spec, _ in self._tools.values()]
        return [s for s in found if source is None or s.source == source]

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
        request_id: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> ToolResult:
        """Run a tool safely. Always returns a ToolResult and never raises."""
        started = time.perf_counter()
        rid = request_id or get_request_id()
        arguments = arguments or {}
        entry = self._tools.get(name)
        source = entry[0].source if entry else None

        def finish(result: ToolResult) -> ToolResult:
            result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            result.request_id = rid
            extra: dict[str, Any] = {
                "event": "tool_call",
                "tool_name": name,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "request_id": rid,
            }
            if result.error:
                extra["error_type"] = result.error.type
            logger.info("tool call finished", extra=extra)
            return result

        def fail(status: ToolStatus, error_type: str, message: str, retryable: bool = False):
            failure = ToolResult.failure(
                name, source, status, error_type, redact(message), retryable
            )
            return finish(failure)

        if entry is None:
            return fail(ToolStatus.INVALID_REQUEST, "UnknownTool", f"No such tool: {name}")
        spec, func = entry

        problem = _validate_arguments(spec, arguments)
        if problem:
            return fail(ToolStatus.INVALID_REQUEST, "InvalidArguments", problem)
        if spec.requires_confirmation and not confirmed:
            return fail(
                ToolStatus.CONFIRMATION_REQUIRED,
                "ConfirmationRequired",
                f"Tool '{name}' changes state and needs explicit user confirmation",
            )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            raw = executor.submit(func, **arguments).result(timeout=timeout)
        except (TimeoutError, httpx.TimeoutException):
            return fail(
                ToolStatus.TIMEOUT, "Timeout", f"Tool '{name}' exceeded {timeout}s", retryable=True
            )
        except ToolInputError as exc:
            return fail(ToolStatus.INVALID_REQUEST, "InvalidInput", str(exc))
        except ToolPermissionError as exc:
            return fail(ToolStatus.FORBIDDEN, "Forbidden", str(exc))
        except (ConnectionError, httpx.NetworkError) as exc:
            return fail(
                ToolStatus.UNAVAILABLE,
                "ServiceUnavailable",
                f"Cannot reach the service: {exc}",
                retryable=True,
            )
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            return fail(ToolStatus.ERROR, type(exc).__name__, str(exc))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        result = raw if isinstance(raw, ToolResult) else ToolResult.success(name, spec.source, raw)
        return finish(result)


registry = ToolRegistry()


def load_all_tools() -> ToolRegistry:
    """Import every tool module so that its @registry.register decorators run."""
    for module in TOOL_MODULES:
        importlib.import_module(f"agent.tools.{module}")
    return registry
