"""Tests for Section 5: ReAct Loop."""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock
from nanoagent import (
    AgentError, AgentConfig, AgentEvent, Tool, ToolError,
    _react_loop, _execute_tool,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _tc(name: str, args: dict, call_id: str = "tc_1"):
    """Build a mock OpenAI tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc

def _response(content=None, tool_calls=None):
    """Build a mock OpenAI ChatCompletion."""
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.tool_calls = tool_calls
    return r

def _null_emit(e):
    pass

def _make_client(responses):
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    return client

def _add_tool():
    def add(a: int, b: int) -> int:
        return a + b
    return Tool("add", "Add two numbers", {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }, add)


# ── Tests ──────────────────────────────────────────────────────────────

def test_no_tool_immediate_return():
    """If the first response has no tool_calls, return text immediately."""
    client = _make_client([_response(content="The answer is 42.")])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "hi"}], [], config, _null_emit, client)
    assert result == "The answer is 42."
    assert client.chat.completions.create.call_count == 1


def test_single_tool_call():
    """Model calls one tool, then returns final text."""
    client = _make_client([
        _response(tool_calls=[_tc("add", {"a": 1337, "b": 42})]),
        _response(content="The result is 1379."),
    ])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)
    assert result == "The result is 1379."
    assert client.chat.completions.create.call_count == 2


def test_multi_tool_call_in_one_turn():
    """Model calls two tools in one response, then returns final text."""
    client = _make_client([
        _response(tool_calls=[_tc("add", {"a": 1, "b": 2}, "tc_1"), _tc("add", {"a": 3, "b": 4}, "tc_2")]),
        _response(content="Done."),
    ])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)
    assert result == "Done."
    assert client.chat.completions.create.call_count == 2


def test_tool_error_recovers():
    """ToolError from a tool is returned as a string; model can reason about it."""
    def broken() -> str:
        raise ToolError("database down")
    broken_tool = Tool("broken", "A broken tool", {
        "type": "object", "properties": {}, "required": []
    }, broken)
    client = _make_client([
        _response(tool_calls=[_tc("broken", {})]),
        _response(content="I couldn't complete that."),
    ])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [broken_tool], config, _null_emit, client)
    assert result == "I couldn't complete that."


def test_unexpected_exception_in_tool_does_not_crash():
    """Unexpected exceptions from tools are caught and returned to the model."""
    def exploding(x: str) -> str:
        raise ValueError("unexpected!")
    exploding_tool = Tool("exploding", "Explodes", {
        "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]
    }, exploding)
    client = _make_client([
        _response(tool_calls=[_tc("exploding", {"x": "boom"})]),
        _response(content="Handled."),
    ])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [exploding_tool], config, _null_emit, client)
    assert result == "Handled."


def test_max_turns_raises_agent_error():
    """Exceeding max_turns raises AgentError."""
    client = _make_client([_response(tool_calls=[_tc("add", {"a": 1, "b": 2})])] * 10)
    config = AgentConfig(sink=lambda e: None, max_turns=3)
    with pytest.raises(AgentError, match="Max turns"):
        _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)


def test_thinking_event_emitted_for_text_with_tool_calls():
    """Text content alongside tool_calls emits a 'thinking' event."""
    client = _make_client([
        _response(content="Let me think...", tool_calls=[_tc("add", {"a": 1, "b": 2})]),
        _response(content="Done."),
    ])
    events = []
    config = AgentConfig(sink=lambda e: None)
    _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config,
                lambda e: events.append(e), client)
    thinking_events = [e for e in events if e.type == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0].payload["text"] == "Let me think..."


def test_tool_not_found_returns_error_string():
    """_execute_tool returns a JSON error string when the tool name is unknown."""
    tc = MagicMock()
    tc.function = MagicMock()
    tc.function.name = "nonexistent"
    tc.function.arguments = "{}"
    result = _execute_tool(tc, [])
    data = json.loads(result)
    assert data["error"] == "ToolNotFound"


def test_execute_tool_returns_string():
    """_execute_tool always returns a string, even for non-string return values."""
    tc = _tc("add", {"a": 2, "b": 3})
    result = _execute_tool(tc, [_add_tool()])
    assert isinstance(result, str)
    assert result == "5"
