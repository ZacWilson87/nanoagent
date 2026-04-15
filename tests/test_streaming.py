"""Tests for Section 6: Streaming."""
from __future__ import annotations
import sys
import io
import pytest
from unittest.mock import MagicMock, patch, call
from nanoagent import _stream_response, AgentConfig, _react_loop, Tool


def _text_delta(text: str):
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.type = "text_delta"
    event.delta.text = text
    return event


def _other_event():
    event = MagicMock()
    event.type = "message_start"
    return event


def _final_message(blocks):
    msg = MagicMock()
    msg.content = blocks
    return msg


def _block(btype, **kwargs):
    b = MagicMock()
    b.type = btype
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


def _make_stream_context(events, final_message):
    """Return a context manager mock that yields events and a final message."""
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.__iter__ = MagicMock(return_value=iter(events))
    stream.get_final_message = MagicMock(return_value=final_message)
    return stream


def test_text_tokens_printed_to_stdout(capsys):
    """Text delta events are written to stdout immediately."""
    stream = _make_stream_context(
        [_text_delta("Hello"), _text_delta(", "), _text_delta("world!")],
        _final_message([_block("text", text="Hello, world!")])
    )
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    _stream_response(client, [], [], config)
    captured = capsys.readouterr()
    assert captured.out == "Hello, world!"


def test_non_text_events_ignored(capsys):
    """Non-text-delta events don't cause errors or spurious output."""
    stream = _make_stream_context(
        [_other_event(), _text_delta("Hi"), _other_event()],
        _final_message([_block("text", text="Hi")])
    )
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    _stream_response(client, [], [], config)
    captured = capsys.readouterr()
    assert captured.out == "Hi"


def test_stream_returns_final_message():
    """_stream_response returns the complete message from get_final_message()."""
    final = _final_message([_block("text", text="Done.")])
    stream = _make_stream_context([], final)
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    result = _stream_response(client, [], [], config)
    assert result is final


def test_system_prompt_passed_to_stream():
    """System prompt is included in the stream kwargs when non-empty."""
    final = _final_message([_block("text", text="Hi.")])
    stream = _make_stream_context([], final)
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(system="You are helpful.", sink=lambda e: None)
    _stream_response(client, [], [], config)
    call_kwargs = client.messages.stream.call_args[1]
    assert call_kwargs.get("system") == "You are helpful."


def test_no_system_prompt_not_passed_to_stream():
    """Empty system prompt is NOT included in stream kwargs."""
    final = _final_message([_block("text", text="Hi.")])
    stream = _make_stream_context([], final)
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(system="", sink=lambda e: None)
    _stream_response(client, [], [], config)
    call_kwargs = client.messages.stream.call_args[1]
    assert "system" not in call_kwargs


def test_react_loop_uses_stream_response_when_stream_true(capsys):
    """When stream=True, _react_loop calls _stream_response instead of messages.create."""
    text_block = _block("text", text="Answer.")
    final = _final_message([text_block])
    stream = _make_stream_context([_text_delta("Answer.")], final)
    client = MagicMock()
    client.messages.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop(
        [{"role": "user", "content": "hi"}], [], config, lambda e: None, client, stream=True
    )
    assert result == "Answer."
    assert client.messages.stream.called
    assert not client.messages.create.called
