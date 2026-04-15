"""Shared fixtures for nanoagent tests."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
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


def _make_content_block(btype: str, **kwargs):
    """Build a mock content block with .type attribute."""
    block = MagicMock()
    block.type = btype
    for k, v in kwargs.items():
        setattr(block, k, v)
    return block


def _make_response(content_blocks, stop_reason="end_turn"):
    """Build a mock Anthropic response."""
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    return resp


@pytest.fixture
def single_tool_response():
    """Canned response: one tool call (add), then final text answer."""
    tool_use_block = _make_content_block(
        "tool_use", name="add", id="tu_001", input={"a": 1337, "b": 42}
    )
    text_block = _make_content_block("text", text="The answer is 1379.")

    # First call returns tool_use, second returns final text
    call_1 = _make_response([tool_use_block])
    call_2 = _make_response([text_block])
    return [call_1, call_2]


@pytest.fixture
def no_tool_response():
    """Canned response: immediate text answer, no tool call."""
    text_block = _make_content_block("text", text="42")
    return [_make_response([text_block])]


@pytest.fixture
def multi_tool_response():
    """Canned response: two tool calls in one turn, then final text."""
    tu1 = _make_content_block("tool_use", name="add", id="tu_001", input={"a": 1, "b": 2})
    tu2 = _make_content_block("tool_use", name="add", id="tu_002", input={"a": 3, "b": 4})
    final = _make_content_block("text", text="Done.")
    return [_make_response([tu1, tu2]), _make_response([final])]


@pytest.fixture
def mock_anthropic_client(single_tool_response):
    """Mock Anthropic client with single_tool_response sequence."""
    client = MagicMock()
    client.messages.create.side_effect = single_tool_response
    return client
