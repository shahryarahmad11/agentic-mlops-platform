"""Provider-neutral LLM interface plus the Gemini implementation.

The orchestrator only depends on LLMClient / ChatSession, so the provider is swappable and
tests use a scripted fake (no API key, no network).
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.schemas import ToolSpec


class LLMError(RuntimeError):
    """The language model could not be reached or returned an unusable answer."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class LLMTurn:
    """One model reply: either tool calls to run, or final text (or both)."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatSession(Protocol):
    def send(self, message: str) -> LLMTurn: ...

    def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> LLMTurn: ...


class LLMClient(Protocol):
    def start(self, system_prompt: str, tools: list[ToolSpec]) -> ChatSession: ...


# ----------------------------------------------------------------------- Gemini
class GeminiSession:
    def __init__(self, client: Any, model: str, config: Any) -> None:
        from google.genai import types

        self._types = types
        self._client = client
        self._model = model
        self._config = config
        self._contents: list[Any] = []

    def _generate(self) -> LLMTurn:
        from google.genai import errors

        try:
            response = self._client.models.generate_content(
                model=self._model, contents=self._contents, config=self._config
            )
        except errors.APIError as exc:
            raise LLMError(f"LLM API error {exc.code}: {exc.status or ''}".strip()) from exc
        except Exception as exc:  # network errors, timeouts
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc

        if not response.candidates or response.candidates[0].content is None:
            raise LLMError("LLM returned no content (possibly blocked or empty)")
        # Keep the model's own Content object: it carries Gemini thought signatures.
        self._contents.append(response.candidates[0].content)
        calls = [
            ToolCall(name=fc.name, arguments=dict(fc.args or {}), id=fc.id)
            for fc in (response.function_calls or [])
        ]
        text = None
        if not calls:
            parts = response.candidates[0].content.parts or []
            text = "".join(p.text for p in parts if getattr(p, "text", None) and not p.thought)
        return LLMTurn(text=text, tool_calls=calls)

    def send(self, message: str) -> LLMTurn:
        types = self._types
        self._contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )
        return self._generate()

    def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> LLMTurn:
        types = self._types
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=call.id, name=call.name, response=payload
                )
            )
            for call, payload in results
        ]
        self._contents.append(types.Content(role="user", parts=parts))
        return self._generate()


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(
                    attempts=3, initial_delay=2.0, http_status_codes=[429, 500, 503]
                ),
            ),
        )

    def start(self, system_prompt: str, tools: list[ToolSpec]) -> GeminiSession:
        types = self._types
        declarations = [
            types.FunctionDeclaration(
                name=spec.name,
                description=_describe(spec),
                parameters_json_schema=spec.parameters,
            )
            for spec in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return GeminiSession(self._client, self._model, config)


def _describe(spec: ToolSpec) -> str:
    flags = []
    if not spec.read_only:
        flags.append("STATE-CHANGING")
    if spec.requires_confirmation:
        flags.append("needs user approval: calling it only creates a pending action")
    suffix = f" [{'; '.join(flags)}]" if flags else ""
    return f"[{spec.source.value}] {spec.description}{suffix}"
