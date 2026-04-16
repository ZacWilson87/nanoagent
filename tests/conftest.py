"""Shared fixtures for nanoagent tests."""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from nanoagent import AgentConfig, AgentEvent


@pytest.fixture
def null_config():
    """AgentConfig with null sink so tests don't spam stderr."""
    return AgentConfig(sink=lambda e: None, max_turns=5)


@pytest.fixture
def events():
    """Collects emitted AgentEvents into a list."""
    collected = []
    def sink(event: AgentEvent):
        collected.append(event)
    return collected, sink


def _make_tool_call(name: str, arguments: dict, call_id: str = "tc_001"):
    """Build a mock OpenAI tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = __import__("json").dumps(arguments)
    return tc


def _make_message(content: str | None = None, tool_calls: list | None = None):
    """Build a mock OpenAI message object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


def _make_response(content: str | None = None, tool_calls: list | None = None):
    """Build a mock OpenAI ChatCompletion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = _make_message(content, tool_calls)
    return resp


@pytest.fixture
def single_tool_response():
    """Canned response: one tool call (add), then final text answer."""
    tc = _make_tool_call("add", {"a": 1337, "b": 42}, "tc_001")
    call_1 = _make_response(tool_calls=[tc])
    call_2 = _make_response(content="The answer is 1379.")
    return [call_1, call_2]


@pytest.fixture
def no_tool_response():
    """Canned response: immediate text answer, no tool call."""
    return [_make_response(content="42")]


@pytest.fixture
def multi_tool_response():
    """Canned response: two tool calls in one turn, then final text."""
    tc1 = _make_tool_call("add", {"a": 1, "b": 2}, "tc_001")
    tc2 = _make_tool_call("add", {"a": 3, "b": 4}, "tc_002")
    call_1 = _make_response(tool_calls=[tc1, tc2])
    call_2 = _make_response(content="Done.")
    return [call_1, call_2]


@pytest.fixture
def mock_openai_client(single_tool_response):
    """Mock OpenAI client with single_tool_response sequence."""
    client = MagicMock()
    client.chat.completions.create.side_effect = single_tool_response
    return client
