"""Tests for Section 2: Tool Registry."""
from __future__ import annotations
import inspect
import pytest
from nanoagent import tool, tool_fn, Tool, _REGISTRY, _build_schema, TYPE_MAP


def test_tool_decorator_registers():
    """@tool registers the function in _REGISTRY."""
    @tool
    def my_func(x: str) -> str:
        """A test function."""
        return x
    assert "my_func" in _REGISTRY
    assert isinstance(_REGISTRY["my_func"], Tool)


def test_tool_decorator_name_and_description():
    """@tool sets name from function name and description from docstring."""
    @tool
    def greet(name: str) -> str:
        """Say hello to someone."""
        return f"Hello, {name}"
    assert _REGISTRY["greet"].name == "greet"
    assert _REGISTRY["greet"].description == "Say hello to someone."


def test_tool_decorator_input_schema():
    """@tool builds correct JSON Schema from type annotations."""
    @tool
    def compute(a: int, b: float, flag: bool) -> str:
        """Compute something."""
        return ""
    schema = _REGISTRY["compute"].input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "number"
    assert schema["properties"]["flag"]["type"] == "boolean"
    assert set(schema["required"]) == {"a", "b", "flag"}


def test_tool_decorator_callable():
    """The tool's fn attribute is the original callable."""
    @tool
    def square(n: int) -> int:
        """Square a number."""
        return n * n
    assert _REGISTRY["square"].fn(4) == 16


def test_tool_fn_explicit_registration():
    """tool_fn() registers with explicit name/description/schema."""
    @tool_fn("explicit_add", "Add two numbers", {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    })
    def _add(a: int, b: int) -> int:
        return a + b
    assert "explicit_add" in _REGISTRY
    assert _REGISTRY["explicit_add"].description == "Add two numbers"
    assert _REGISTRY["explicit_add"].fn(3, 4) == 7


def test_type_map_coverage():
    """TYPE_MAP covers all required types."""
    for t in [str, int, float, bool, list, dict]:
        assert t in TYPE_MAP


def test_unannotated_param_defaults_to_string():
    """Parameters without type annotations become 'string' in the schema."""
    def bare(x) -> str:
        """No annotations."""
        return str(x)
    schema = _build_schema(bare)
    assert schema["properties"]["x"]["type"] == "string"


def test_optional_param_not_in_required():
    """Parameters with defaults are not in required list."""
    def with_default(name: str, count: int = 1) -> str:
        """Has optional param."""
        return name * count
    schema = _build_schema(with_default)
    assert "name" in schema["required"]
    assert "count" not in schema["required"]


def test_tool_returns_tool_instance():
    """@tool returns a Tool instance, not the original function."""
    @tool
    def returns_tool(x: str) -> str:
        """Returns tool."""
        return x
    assert isinstance(returns_tool, Tool)
