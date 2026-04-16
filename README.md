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

When building AI solutions you start to notice the same patterns — in the
tools you use and in the code you write yourself. The pattern is simple.
But frameworks bury it under abstractions, configs, and layers of indirection
that obscure more than they clarify.

nanoagent is what happens when you strip all of that away. One file. One
dependency. No noise. The pattern, plainly.

The ≤300-line constraint is a forcing function: every line that doesn't earn
its place gets cut. What's left is the thing itself, readable end-to-end in
20 minutes.

## Understanding the Code

**[docs/walkthrough.md](docs/walkthrough.md)** is the companion to reading
`nanoagent.py`. It covers:

- The ReAct pattern — what it is and why it works
- Each of the 8 sections in depth, with the reasoning behind every decision
- A step-by-step execution trace of a real example
- What nanoagent deliberately doesn't do, and why

If you want to actually understand how agents work, read the walkthrough
alongside the source. It's the point of the project.

## Learning Path

**1. Read the source first (20 min)**

Open `nanoagent.py` and read it top-to-bottom once, then open
`docs/walkthrough.md` alongside it. The walkthrough explains *why* each
section is shaped the way it is — not just what it does.

**2. Run the examples with events visible**

The JSON events print to stderr. Separate streams to see them clearly:

```bash
python examples/web_researcher.py 2>/tmp/events.json
cat /tmp/events.json
```

Or watch live in a split terminal:

```bash
python examples/web_researcher.py 1>/dev/null   # events only
```

**3. Write your own tool (the real lesson)**

This is where it clicks:

```python
from nanoagent import tool, run

@tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())

@tool
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

print(run("Reverse the phrase 'hello world' and then count its words"))
```

Watch the model decide which tools to call and in what order — without
you telling it.

**4. Swap the observability sink**

The sink is pluggable. Try capturing events instead of printing them:

```python
import json
from nanoagent import tool, run, AgentConfig, Agent, AgentEvent

log = []

def capturing_sink(event: AgentEvent) -> None:
    log.append({"ts": event.ts, "type": event.type, **event.payload})

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = Agent(config=AgentConfig(sink=capturing_sink))
agent.run("What is 99 + 1?")

print(json.dumps(log, indent=2))  # full structured trace
```

**5. Break the context window intentionally**

Have a long multi-turn conversation and watch `context_trimmed` events
appear. Then look at `ContextManager.trim()` (line 115) to see exactly
what gets dropped and why.

**6. Read `multi_agent.py` last**

Once you understand a single agent, the multi-agent example shows how one
agent becomes a tool on another — the architecture pattern behind every
production multi-agent system.

---

The key insight: all the complexity in LangChain, LangGraph, and CrewAI is
decorators around the same ~50 lines in section 5. Once you can read
`_react_loop()` at line 157 and explain each step out loud, you understand
agents.

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
