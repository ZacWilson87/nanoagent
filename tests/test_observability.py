"""Tests for Section 4: Observability."""
from __future__ import annotations
import json
import sys
import io
import pytest
from nanoagent import AgentEvent, default_sink


def test_agent_event_has_timestamp():
    event = AgentEvent(type="turn_start", payload={"message": "hi"})
    assert isinstance(event.ts, float)
    assert event.ts > 0


def test_agent_event_type_literal():
    """All seven event types are valid."""
    types = ["turn_start", "thinking", "tool_call", "tool_result",
             "turn_end", "context_trimmed", "error"]
    for t in types:
        ev = AgentEvent(type=t, payload={})
        assert ev.type == t


def test_default_sink_writes_json_to_stderr(capsys):
    """default_sink emits one JSON line per event on stderr."""
    event = AgentEvent(type="turn_start", payload={"message": "hello"})
    default_sink(event)
    captured = capsys.readouterr()
    assert captured.out == ""
    line = captured.err.strip()
    data = json.loads(line)
    assert data["type"] == "turn_start"
    assert data["message"] == "hello"
    assert "ts" in data


def test_default_sink_includes_all_payload_fields(capsys):
    """All payload keys appear at the top level in the JSON output."""
    event = AgentEvent(type="tool_call", payload={"tool": "add", "input": {"a": 1}})
    default_sink(event)
    captured = capsys.readouterr()
    data = json.loads(captured.err.strip())
    assert data["tool"] == "add"
    assert data["input"] == {"a": 1}


def test_null_sink_suppresses_output(capsys):
    """A lambda sink receives events without printing anything."""
    received = []
    null = lambda e: received.append(e)
    event = AgentEvent(type="turn_end", payload={"text": "done", "turns": 2})
    null(event)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert len(received) == 1
    assert received[0].type == "turn_end"


def test_custom_sink_receives_events():
    """A custom sink callable receives AgentEvent objects."""
    received = []
    def my_sink(event: AgentEvent):
        received.append(event)

    events_to_emit = [
        AgentEvent(type="turn_start", payload={"message": "q"}),
        AgentEvent(type="thinking", payload={"text": "reasoning..."}),
        AgentEvent(type="turn_end", payload={"text": "answer", "turns": 1}),
    ]
    for e in events_to_emit:
        my_sink(e)

    assert len(received) == 3
    assert received[0].type == "turn_start"
    assert received[1].type == "thinking"
    assert received[2].type == "turn_end"


def test_context_trimmed_event_has_removed_count():
    event = AgentEvent(type="context_trimmed", payload={"removed": 2})
    assert event.payload["removed"] == 2


def test_error_event():
    event = AgentEvent(type="error", payload={"message": "something broke"})
    assert event.type == "error"
