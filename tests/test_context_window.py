"""Tests for Section 3: Context Manager."""
from __future__ import annotations
import pytest
from nanoagent import ContextManager, Turn


def _user(text="hello"):
    return Turn(role="user", content=text)

def _assistant(text="hi"):
    return Turn(role="assistant", content=text)


def test_push_and_len():
    ctx = ContextManager()
    ctx.push(_user())
    ctx.push(_assistant())
    assert len(ctx) == 2


def test_to_messages_format():
    ctx = ContextManager()
    ctx.push(_user("Hello"))
    ctx.push(_assistant("Hi"))
    msgs = ctx.to_messages()
    assert msgs == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_trim_removes_oldest_pair():
    ctx = ContextManager()
    ctx.push(_user("first"))       # turns[0] — never removed
    ctx.push(_assistant("reply1")) # turns[1] — removed on first trim
    ctx.push(_user("second"))      # turns[2] — removed on first trim
    ctx.push(_assistant("reply2")) # turns[3]
    ctx.push(_user("third"))       # turns[4]
    removed = ctx.trim()
    assert removed == 2
    assert len(ctx) == 3
    msgs = ctx.to_messages()
    # First message preserved
    assert msgs[0]["content"] == "first"
    # turns[1] and turns[2] gone; turns[3] and [4] remain
    assert msgs[1]["content"] == "reply2"
    assert msgs[2]["content"] == "third"


def test_trim_preserves_role_alternation():
    """After trimming, messages must still alternate user/assistant."""
    ctx = ContextManager()
    ctx.push(_user("q1"))
    ctx.push(_assistant("a1"))
    ctx.push(_user("q2"))
    ctx.push(_assistant("a2"))
    ctx.push(_user("q3"))
    ctx.push(_assistant("a3"))
    ctx.trim()
    msgs = ctx.to_messages()
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user"
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], f"Consecutive same roles at index {i}"


def test_trim_does_nothing_with_two_or_fewer():
    ctx = ContextManager()
    ctx.push(_user())
    ctx.push(_assistant())
    removed = ctx.trim()
    assert removed == 0
    assert len(ctx) == 2


def test_first_user_message_never_trimmed():
    ctx = ContextManager()
    ctx.push(_user("original task"))
    for i in range(10):
        ctx.push(_assistant(f"a{i}"))
        ctx.push(_user(f"q{i}"))
    for _ in range(5):
        ctx.trim()
    msgs = ctx.to_messages()
    assert msgs[0]["content"] == "original task"


def test_token_estimate_nonzero():
    ctx = ContextManager()
    ctx.push(_user("Hello world this is a message"))
    assert ctx.token_estimate() > 0


def test_roundtrip_push_to_messages():
    ctx = ContextManager()
    turns = [
        _user("What is 2+2?"),
        _assistant("4"),
        _user("And 3+3?"),
        _assistant("6"),
    ]
    for t in turns:
        ctx.push(t)
    msgs = ctx.to_messages()
    assert len(msgs) == 4
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
