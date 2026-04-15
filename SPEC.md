# nanoagent — Specification

## Philosophy

nanoagent.py is a single file of ≤300 lines of pure Python with one
dependency (`anthropic`) that implements a complete AI agent reasoning
engine: the ReAct loop, tool registration, context window management,
multi-turn memory, streaming output, and structured observability.

No LangChain. No LangGraph. No CrewAI. No AutoGen. No abstractions between
you and the model.

---

## Decisions (do not relitigate)

- Language: Python 3.11+
- Only allowed dependency in `nanoagent.py`: `anthropic` (the SDK)
- Model: `claude-sonnet-4-20250514` hardcoded as default, overridable
- Reasoning pattern: ReAct (Reason + Act) — thought → tool call → observation → repeat
- Tool definition: plain Python functions decorated with `@tool`
- Streaming: yes, via Anthropic streaming API — output tokens as they arrive
- Context management: sliding window with configurable `max_turns` (default 20)
- Memory: in-process only
- Observability: structured event emission to a pluggable sink (default: stderr JSON lines)
- Examples: separate files in `/examples` — never inline in `nanoagent.py`
- Tests: `pytest` only
- Packaging: `pyproject.toml` with `uv`
- License: MIT

---

## Repository Structure

```
nanoagent/
├── nanoagent.py              # THE ENGINE — ≤300 lines, one dep, the artifact
├── examples/
│   ├── hello_tool.py
│   ├── web_researcher.py
│   ├── code_executor.py
│   ├── multi_agent.py
│   └── hitl_agent.py
├── tests/
│   ├── test_tool_registry.py
│   ├── test_react_loop.py
│   ├── test_context_window.py
│   ├── test_streaming.py
│   └── test_observability.py
├── docs/
│   └── walkthrough.md
├── pyproject.toml
├── AGENTS.md
├── SPEC.md
└── README.md
```

---

## Core Data Model

```python
@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable

@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: list[dict]

@dataclass
class AgentEvent:
    type: Literal[
        "turn_start", "thinking", "tool_call", "tool_result",
        "turn_end", "context_trimmed", "error",
    ]
    payload: dict
    ts: float

@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 20
    max_tokens: int = 4096
    system: str = ""
    sink: Callable | None = None
```

---

## Section Structure

```
# ── 1. TYPES
# ── 2. TOOL REGISTRY
# ── 3. CONTEXT MANAGER
# ── 4. OBSERVABILITY
# ── 5. REACT LOOP
# ── 6. STREAMING
# ── 7. AGENT
# ── 8. PUBLIC API
```

---

## Section 2 — Tool Registry

TYPE_MAP:
```python
TYPE_MAP = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}
```

`@tool` decorator reads signature + docstring, builds `input_schema`, registers in `_REGISTRY`.
`tool_fn(name, description, input_schema)` is the explicit alternative.

---

## Section 3 — Context Manager

```python
class ContextManager:
    def __init__(self, max_turns: int = 20)
    def push(self, turn: Turn) -> None
    def to_messages(self) -> list[dict]
    def trim(self) -> int
    def token_estimate(self) -> int
```

Trim: remove oldest 2 turns (user+assistant pair). Never remove first user message.
Emit `context_trimmed` event when trim occurs.

---

## Section 5 — ReAct Loop

Loop up to `max_turns`. Each iteration:
1. Call model
2. If no tool_use blocks → extract text, emit `turn_end`, return
3. Emit `thinking` for text blocks
4. Execute each tool, collect results
5. Append assistant response + tool results to messages

`_execute_tool` never raises — returns `{"error": ..., "message": ...}` on failure.
Raise `AgentError` if max_turns exceeded.

---

## Section 6 — Streaming

Default `stream=True`. Print text tokens to stdout as they arrive.
Buffer tool_use blocks until complete. Reconstruct full response object.

---

## Section 8 — Public API

```python
def tool(fn: Callable) -> Tool: ...
def tool_fn(name: str, description: str, input_schema: dict) -> Callable: ...
AgentConfig
Agent
def run(message: str, tools=None, system="", stream=True) -> str: ...
def register_module(module) -> int: ...
```

---

## Error Handling

```python
class AgentError(Exception): ...   # unrecoverable: max turns, API failure
class ToolError(Exception): ...    # expected tool failure, model can reason about it
```

---

## Implementation Phases

- Phase 0: Skeleton — section comments, pyproject.toml
- Phase 1: Types + Tool Registry — tests pass
- Phase 2: Context Manager — tests pass
- Phase 3: Observability — tests pass
- Phase 4: ReAct Loop (non-streaming) — tests pass
- Phase 5: Streaming — tests pass
- Phase 6: Agent Class + Public API — tests pass
- Phase 7: Examples
- Phase 8: Walkthrough doc, README, line count ≤300 check
