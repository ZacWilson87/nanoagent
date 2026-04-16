# nanoagent.py — The irreducible AI agent reasoning engine
# ≤300 lines. One dependency (openai). Read top-to-bottom in 20 minutes.
from __future__ import annotations
import inspect, json, sys, time, typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import openai

# ── 1. TYPES ──────────────────────────────────────────────────────────

class AgentError(Exception):
    """Raised when the agent cannot recover — max turns, API failure, etc."""

class ToolError(Exception):
    """Raised by a tool to signal an expected failure the agent can reason about."""

@dataclass
class Tool:
    """A callable registered with the agent, carrying its own JSON Schema."""
    name: str
    description: str
    input_schema: dict    # JSON Schema sent to the API
    fn: Callable          # the actual Python function

@dataclass
class Turn:
    """One message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str          # plain text

@dataclass
class AgentEvent:
    """Structured event emitted at every state transition — the audit trail."""
    type: Literal[
        "turn_start",      # new user message submitted
        "thinking",        # model produced a text block
        "tool_call",       # model requested a tool
        "tool_result",     # tool executed, result ready
        "turn_end",        # model reached final answer (no tool calls)
        "context_trimmed", # context window was pruned
        "error",           # unrecoverable error
    ]
    payload: dict
    ts: float = field(default_factory=time.time)

@dataclass
class AgentConfig:
    """Knobs with sane defaults. Override what you need."""
    model: str = "gpt-4o"
    max_turns: int = 20        # max ReAct iterations before forced stop
    max_tokens: int = 4096     # max tokens per model call
    system: str = ""           # system prompt
    sink: Callable | None = None   # observability sink (None = stderr JSON)
    base_url: str | None = None    # e.g. "https://api.groq.com/openai/v1"
    api_key: str | None = None     # falls back to OPENAI_API_KEY env var

# ── 2. TOOL REGISTRY ──────────────────────────────────────────────────
# Convert Python functions → OpenAI tool definitions via inspect + docstrings.

TYPE_MAP: dict[type, str] = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}
_REGISTRY: dict[str, Tool] = {}

def _build_schema(fn: Callable) -> dict:
    """Build a JSON Schema from a function's type annotations."""
    # get_type_hints() resolves forward references (needed for PEP 563 annotations)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        # DECISION: unannotated params default to "string" — conservative, never crashes
        props[name] = {"type": TYPE_MAP.get(hints.get(name), "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}

def tool(fn: Callable) -> Tool:
    """Decorator: register a Python function as an agent tool.

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b
    """
    t = Tool(fn.__name__, (fn.__doc__ or "").strip(), _build_schema(fn), fn)
    _REGISTRY[t.name] = t
    return t

def tool_fn(name: str, description: str, input_schema: dict) -> Callable:
    """Explicit registration (alternative to @tool) — useful when wrapping external APIs."""
    def decorator(fn: Callable) -> Tool:
        t = Tool(name, description, input_schema, fn)
        _REGISTRY[name] = t
        return t
    return decorator

# ── 3. CONTEXT MANAGER ────────────────────────────────────────────────
# Sliding-window history. OpenAI requires alternating user/assistant roles,
# so we always trim in pairs to preserve that invariant.

class ContextManager:
    """Sliding-window conversation history. Pair-trims to keep role alternation."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def push(self, turn: Turn) -> None:
        """Append a turn."""
        self._turns.append(turn)

    def to_messages(self) -> list[dict]:
        """Convert to the list[dict] format the OpenAI API expects."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def trim(self) -> int:
        """Remove the oldest user+assistant pair. Never removes the first user message.

        Returns count of turns removed. Callers check len > max_turns before calling.
        """
        if len(self._turns) <= 2:
            return 0
        # Preserve _turns[0] (original task), drop _turns[1] and [2]
        self._turns = [self._turns[0]] + self._turns[3:]
        return 2

    def token_estimate(self) -> int:
        """Rough token count: total chars / 4."""
        return sum(len(t.content) for t in self._turns) // 4

    def __len__(self) -> int:
        return len(self._turns)

# ── 4. OBSERVABILITY ──────────────────────────────────────────────────
# Every state transition emits a structured AgentEvent to a sink.
# Swap the sink without touching any other code.

def default_sink(event: AgentEvent) -> None:
    """Write one JSON line per event to stderr."""
    print(json.dumps({"ts": event.ts, "type": event.type, **event.payload}), file=sys.stderr)

# ── 5. REACT LOOP ─────────────────────────────────────────────────────
# The heart of the file. This is what every agent framework does under the hood.
# Thought → tool call → observation → repeat until no tool_calls in response.

def _execute_tool(tc: Any, tools: list[Tool]) -> str:
    """Execute a tool call. Never raises — errors become model-visible strings."""
    matched = next((t for t in tools if t.name == tc.function.name), None)
    if matched is None:
        return json.dumps({"error": "ToolNotFound", "message": f"No tool named '{tc.function.name}'"})
    try:
        return str(matched.fn(**json.loads(tc.function.arguments)))
    except ToolError as e:
        return json.dumps({"error": "ToolError", "message": str(e)})
    except Exception as e:
        return json.dumps({"error": type(e).__name__, "message": str(e)})

def _react_loop(
    messages: list[dict],
    tools: list[Tool],
    config: AgentConfig,
    emit: Callable[[AgentEvent], None],
    client: openai.OpenAI,
    stream: bool = False,
) -> str:
    """Core ReAct reasoning loop. Returns final text or raises AgentError."""
    tool_defs = [{"type": "function", "function": {
        "name": t.name, "description": t.description, "parameters": t.input_schema,
    }} for t in tools]
    # System prompt goes as first message — OpenAI takes it in the messages list, not a kwarg
    msgs = list(messages) if not config.system else [{"role": "system", "content": config.system}, *messages]
    for turn_num in range(config.max_turns):
        if stream:
            response = _stream_response(client, msgs, tool_defs, config)
        else:
            kw: dict[str, Any] = dict(model=config.model, max_tokens=config.max_tokens, messages=msgs)
            if tool_defs:
                kw["tools"] = tool_defs
            response = client.chat.completions.create(**kw)
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            text = msg.content or ""
            emit(AgentEvent(type="turn_end", payload={"text": text, "turns": turn_num}))
            return text

        if msg.content:
            emit(AgentEvent(type="thinking", payload={"text": msg.content}))

        results = []
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            emit(AgentEvent(type="tool_call", payload={"tool": tc.function.name, "input": args}))
            result = _execute_tool(tc, tools)
            emit(AgentEvent(type="tool_result", payload={"tool": tc.function.name, "result": result}))
            results.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        msgs.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })
        msgs.extend(results)

    raise AgentError(f"Max turns ({config.max_turns}) exceeded without resolution")

# ── 6. STREAMING ──────────────────────────────────────────────────────
# Print text tokens to stdout as they arrive. Tool call JSON is buffered
# inside the SDK and fully assembled before get_final_completion() returns.

def _stream_response(
    client: openai.OpenAI,
    messages: list[dict],
    tool_defs: list[dict],
    config: AgentConfig,
) -> Any:
    """Stream model response, printing text tokens immediately. Returns full response."""
    kw: dict[str, Any] = dict(model=config.model, max_tokens=config.max_tokens, messages=messages)
    if tool_defs:
        kw["tools"] = tool_defs
    with client.chat.completions.stream(**kw) as stream:
        for text in stream.text_stream:
            sys.stdout.write(text)
            sys.stdout.flush()
        return stream.get_final_completion()

# ── 7. AGENT ──────────────────────────────────────────────────────────
# Thin wrapper: holds config + context + tools, delegates reasoning to _react_loop.
# Multi-turn state lives in ContextManager. The loop itself is stateless.

class Agent:
    """The agent. Thin wrapper around _react_loop with persistent context."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        config: AgentConfig | None = None,
        system: str = "",
    ) -> None:
        self.config = config or AgentConfig(system=system)
        if system and not config:
            self.config.system = system
        self.tools = tools if tools is not None else list(_REGISTRY.values())
        self.context = ContextManager(self.config.max_turns)
        self.emit = self.config.sink or default_sink
        self._client = openai.OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def run(self, message: str, stream: bool = True) -> str:
        """Submit a message, get a final response. Maintains conversation history."""
        self.emit(AgentEvent(type="turn_start", payload={"message": message}))
        self.context.push(Turn(role="user", content=message))
        while len(self.context) > self.config.max_turns:
            count = self.context.trim()
            self.emit(AgentEvent(type="context_trimmed", payload={"removed": count}))
        messages = self.context.to_messages()
        response = _react_loop(messages, self.tools, self.config, self.emit, self._client, stream)
        self.context.push(Turn(role="assistant", content=response))
        return response

    def run_fresh(self, message: str, stream: bool = False) -> str:
        """Submit a message with cleared history. Stateless single-shot call."""
        self.reset()
        return self.run(message, stream=stream)

    def reset(self) -> None:
        """Clear conversation history."""
        self.context = ContextManager(self.config.max_turns)

# ── 8. PUBLIC API ─────────────────────────────────────────────────────
# These 6 names are the entire public surface. Everything else is private (_prefixed).
#   from nanoagent import tool, run           # for simple scripts
#   from nanoagent import Agent, AgentConfig  # for production use

def run(
    message: str,
    tools: list[Tool] | None = None,
    system: str = "",
    stream: bool = True,
) -> str:
    """Convenience: one-shot query. Creates a fresh Agent, runs once, returns response."""
    return Agent(tools=tools, system=system).run(message, stream=stream)

def register_module(module: Any) -> int:
    """Register all @tool-decorated objects from a module. Returns count registered."""
    count = 0
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Tool):
            _REGISTRY[obj.name] = obj
            count += 1
    return count
