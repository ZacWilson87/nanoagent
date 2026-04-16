"""Tests for Section 6: Streaming."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from nanoagent import _stream_response, AgentConfig, _react_loop, Tool


def _make_completion(content=None, tool_calls=None):
    """Build a mock ChatCompletion returned by get_final_completion()."""
    c = MagicMock()
    c.choices = [MagicMock()]
    c.choices[0].message = MagicMock()
    c.choices[0].message.content = content
    c.choices[0].message.tool_calls = tool_calls
    return c


def _make_stream_context(text_chunks: list[str], final_completion):
    """Return a context manager mock with text_stream and get_final_completion()."""
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.text_stream = iter(text_chunks)
    stream.get_final_completion = MagicMock(return_value=final_completion)
    return stream


def test_text_tokens_printed_to_stdout(capsys):
    """Text chunks from text_stream are written to stdout immediately."""
    stream = _make_stream_context(
        ["Hello", ", ", "world!"],
        _make_completion(content="Hello, world!")
    )
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    _stream_response(client, [], [], config)
    assert capsys.readouterr().out == "Hello, world!"


def test_empty_text_stream_prints_nothing(capsys):
    """When model only calls tools (no text), stdout is empty."""
    stream = _make_stream_context([], _make_completion())
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    _stream_response(client, [], [], config)
    assert capsys.readouterr().out == ""


def test_stream_returns_final_completion():
    """_stream_response returns the object from get_final_completion()."""
    final = _make_completion(content="Done.")
    stream = _make_stream_context([], final)
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    result = _stream_response(client, [], [], config)
    assert result is final


def test_tools_passed_when_non_empty():
    """tool_defs are included in stream kwargs when non-empty."""
    stream = _make_stream_context([], _make_completion(content="Hi."))
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    tool_defs = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]
    _stream_response(client, [], tool_defs, config)
    call_kwargs = client.chat.completions.stream.call_args[1]
    assert call_kwargs.get("tools") == tool_defs


def test_tools_omitted_when_empty():
    """tools kwarg is NOT sent when tool_defs is empty."""
    stream = _make_stream_context([], _make_completion(content="Hi."))
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    _stream_response(client, [], [], config)
    call_kwargs = client.chat.completions.stream.call_args[1]
    assert "tools" not in call_kwargs


def test_react_loop_uses_stream_when_stream_true(capsys):
    """When stream=True, _react_loop uses chat.completions.stream not .create."""
    final = _make_completion(content="Answer.")
    stream = _make_stream_context(["Answer."], final)
    client = MagicMock()
    client.chat.completions.stream.return_value = stream
    config = AgentConfig(sink=lambda e: None)
    result = _react_loop(
        [{"role": "user", "content": "hi"}], [], config, lambda e: None, client, stream=True
    )
    assert result == "Answer."
    assert client.chat.completions.stream.called
    assert not client.chat.completions.create.called
