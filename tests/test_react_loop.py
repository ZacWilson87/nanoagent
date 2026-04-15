"""Tests for Section 5: ReAct Loop."""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from nanoagent import (
    AgentError, AgentConfig, AgentEvent, Tool, ToolError,
    _react_loop, _execute_tool,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _block(btype, **kwargs):
    b = MagicMock()
    b.type = btype
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b

def _response(blocks):
    r = MagicMock()
    r.content = blocks
    return r

def _null_emit(e):
    pass

def _make_client(responses):
    client = MagicMock()
    client.messages.create.side_effect = responses
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
    """If the first response has no tool_use, return text immediately."""
    text_block = _block("text", text="The answer is 42.")
    client = _make_client([_response([text_block])])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "hi"}], [], config, _null_emit, client)
    assert result == "The answer is 42."
    assert client.messages.create.call_count == 1


def test_single_tool_call():
    """Model calls one tool, then returns final text."""
    tu = _block("tool_use", name="add", id="t1", input={"a": 1337, "b": 42})
    text = _block("text", text="The result is 1379.")
    client = _make_client([_response([tu]), _response([text])])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)
    assert result == "The result is 1379."
    assert client.messages.create.call_count == 2


def test_multi_tool_call_in_one_turn():
    """Model calls two tools in one response, then returns final text."""
    tu1 = _block("tool_use", name="add", id="t1", input={"a": 1, "b": 2})
    tu2 = _block("tool_use", name="add", id="t2", input={"a": 3, "b": 4})
    final = _block("text", text="Done.")
    client = _make_client([_response([tu1, tu2]), _response([final])])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)
    assert result == "Done."
    # Messages were extended with both tool results
    assert client.messages.create.call_count == 2


def test_tool_error_recovers():
    """ToolError from a tool is returned as a string; model can reason about it."""
    def broken() -> str:
        raise ToolError("database down")
    broken_tool = Tool("broken", "A broken tool", {
        "type": "object", "properties": {}, "required": []
    }, broken)
    tu = _block("tool_use", name="broken", id="t1", input={})
    final = _block("text", text="I couldn't complete that.")
    client = _make_client([_response([tu]), _response([final])])
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
    tu = _block("tool_use", name="exploding", id="t1", input={"x": "boom"})
    final = _block("text", text="Handled.")
    client = _make_client([_response([tu]), _response([final])])
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop([{"role": "user", "content": "q"}], [exploding_tool], config, _null_emit, client)
    assert result == "Handled."


def test_max_turns_raises_agent_error():
    """Exceeding max_turns raises AgentError."""
    tu = _block("tool_use", name="add", id="t1", input={"a": 1, "b": 2})
    # Always return a tool_use — loop never terminates
    client = _make_client([_response([tu])] * 10)
    config = AgentConfig(sink=lambda e: None, max_turns=3)
    with pytest.raises(AgentError, match="Max turns"):
        _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config, _null_emit, client)


def test_thinking_event_emitted_for_text_blocks():
    """Text blocks that accompany tool_use emit 'thinking' events."""
    thought = _block("text", text="Let me think...")
    tu = _block("tool_use", name="add", id="t1", input={"a": 1, "b": 2})
    final = _block("text", text="Done.")
    client = _make_client([_response([thought, tu]), _response([final])])
    events = []
    config = AgentConfig(sink=lambda e: None)
    _react_loop([{"role": "user", "content": "q"}], [_add_tool()], config,
                lambda e: events.append(e), client)
    thinking_events = [e for e in events if e.type == "thinking"]
    assert len(thinking_events) == 1
    assert thinking_events[0].payload["text"] == "Let me think..."


def test_tool_not_found_returns_error_string():
    """_execute_tool returns a JSON error string when the tool name is unknown."""
    tu = _block("tool_use", name="nonexistent", id="t1", input={})
    result = _execute_tool(tu, [])
    data = json.loads(result)
    assert data["error"] == "ToolNotFound"


def test_execute_tool_returns_string():
    """_execute_tool always returns a string, even for non-string return values."""
    tu = _block("tool_use", name="add", id="t1", input={"a": 2, "b": 3})
    result = _execute_tool(tu, [_add_tool()])
    assert isinstance(result, str)
    assert result == "5"
