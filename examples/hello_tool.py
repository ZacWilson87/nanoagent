"""hello_tool.py — The absolute minimum: one tool, one question.

Run with:  uv run examples/hello_tool.py
"""
from nanoagent import tool, run


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


print(run("What is 1337 + 42?"))
