"""Shared fixtures. Every external system is mocked: CI needs no services and no API key."""

import os

os.environ["APP_ENV"] = "test"
os.environ["K8S_NAMESPACE"] = "mlops"
os.environ["K8S_ALLOWED_NAMESPACES"] = "mlops"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from typing import Any

import pytest

from agent.llm import LLMTurn, ToolCall
from agent.tools import load_all_tools

REGISTRY = load_all_tools()


@pytest.fixture
def registry():
    return REGISTRY


class ScriptedLLM:
    """Fake LLM: returns pre-scripted turns and records what the agent sent back."""

    def __init__(self, *turns: Any) -> None:
        self.turns = list(turns)
        self.tool_results: list[list[tuple[ToolCall, dict]]] = []
        self.system_prompt = ""
        self.tool_names: list[str] = []

    def start(self, system_prompt, tools):
        self.system_prompt = system_prompt
        self.tool_names = [t.name for t in tools]
        return self

    def send(self, message: str) -> LLMTurn:
        return self._next()

    def send_tool_results(self, results):
        self.tool_results.append(results)
        return self._next()

    def _next(self) -> LLMTurn:
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def calls(*specs: tuple[str, dict]) -> LLMTurn:
    return LLMTurn(
        tool_calls=[ToolCall(name=n, arguments=a, id=f"c{i}") for i, (n, a) in enumerate(specs)]
    )


def final(text: str) -> LLMTurn:
    return LLMTurn(text=text)
