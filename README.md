# nanoagent

> The irreducible AI agent reasoning engine.

`nanoagent.py` is a single file of ≤300 lines with one dependency
(`anthropic`) that implements a complete AI agent: the ReAct reasoning
loop, tool registration, context window management, streaming output,
and structured observability.

No LangChain. No LangGraph. No abstractions between you and the model.

Read it in 20 minutes. Understand agents completely.

## Quickstart

```python
from nanoagent import tool, run

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

print(run("What is 1337 + 42?"))
```

## Installation

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
```

Or with `uv`:

```bash
uv add anthropic
uv run examples/hello_tool.py
```

## Examples

| Example | Demonstrates |
|---------|-------------|
| `examples/hello_tool.py` | Minimal: one tool, one question |
| `examples/web_researcher.py` | Multi-tool chaining (search + fetch + summarize) |
| `examples/code_executor.py` | Model writes and runs its own Python code |
| `examples/multi_agent.py` | Two agents — one registered as a tool on the other |
| `examples/hitl_agent.py` | Human-in-the-loop via a blocking stdin tool |

## Why

Every engineer building with LangGraph, CrewAI, or AutoGen is working on
top of a reasoning pattern they've never seen implemented plainly.
nanoagent.py is that implementation.

Read [`docs/walkthrough.md`](docs/walkthrough.md) for a line-by-line guide.

## Architecture

```
nanoagent.py is structured in 8 sections:

  1. TYPES          — Tool, Turn, AgentEvent, AgentConfig (the vocabulary)
  2. TOOL REGISTRY  — @tool decorator, inspect-based JSON Schema generation
  3. CONTEXT MANAGER— Sliding window history, pair-trim invariant
  4. OBSERVABILITY  — Structured events to a pluggable sink
  5. REACT LOOP     — The heart: thought → tool call → observation → repeat
  6. STREAMING      — Live stdout token printing, buffer-and-reconstruct
  7. AGENT          — Thin wrapper: config + context + tools
  8. PUBLIC API     — 6 names: tool, tool_fn, AgentConfig, Agent, run, register_module
```

## Public API

```python
# Decorator: register a function as a tool
@tool
def my_tool(x: str) -> str: ...

# Explicit registration
@tool_fn("name", "description", input_schema)
def my_tool(x: str) -> str: ...

# Configuration
config = AgentConfig(model="claude-sonnet-4-20250514", max_turns=20, system="...")

# Multi-turn agent
agent = Agent(tools=[...], config=config)
response = agent.run("message")
agent.reset()

# One-shot convenience
response = run("message", tools=[...], system="...", stream=True)

# Register all tools from a module
count = register_module(my_module)
```

## Design Decisions

- **Single file**: The artifact is the education. `nanoagent.py` top-to-bottom is the tutorial.
- **≤300 lines**: Forces prioritization. Every line earns its place.
- **One dependency**: `anthropic` only. No hidden complexity from transitive deps.
- **ReAct pattern**: Thought → action → observation. The simplest complete reasoning loop.
- **Pluggable sink**: Production observability without changing the core.
- **Pair-trim**: Preserves the Anthropic API's role-alternation invariant automatically.

## Tests

```bash
pytest tests/
```

40 tests across 5 files. No real API calls — mocked throughout.

## License

MIT
