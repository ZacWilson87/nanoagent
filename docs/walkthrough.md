# nanoagent.py — A Walkthrough

> 294 lines. One dependency. A complete AI agent.

`nanoagent.py` implements a complete AI agent reasoning engine: the ReAct
loop, tool registration, context management, streaming output, and structured
observability. You can read it top-to-bottom in about 20 minutes. This
document walks you through it section by section.

---

## The Core Idea

The fundamental question of an AI agent is: how do you give a language model
the ability to *act* in the world, not just respond to prompts?

The answer is the **ReAct pattern** (Reason + Act). You tell the model about
a set of tools — functions it can call — and structure the conversation so
the model can interleave reasoning with action. The loop looks like this:

1. You send the model a user message plus tool definitions.
2. The model either responds with text (it's done) or requests one or more
   tool calls.
3. If it requested tools, you execute them and send the results back.
4. Repeat from step 2 until the model stops requesting tools.

That's it. The model does the reasoning. You do the execution. The loop
terminates naturally when the model is satisfied with what it knows.

What makes this powerful is that you don't need to tell the model *when* to
use tools or *which* tool to use for a given situation. The model figures this
out from the tool descriptions and the context. All you need to provide are
well-named tools with clear docstrings.

What makes this safe is that the model never directly executes code — it
*requests* tool calls, and your code executes them with whatever sandboxing
you choose to apply. The model sees only the string results you return.

---

## Section 1: Types

nanoagent.py opens with four dataclasses and two exception types. Reading
these is sufficient to understand the vocabulary of the entire codebase.

**`AgentError`** — raised when the agent cannot continue: the model exceeded
the maximum turn limit, the API failed unrecoverably, etc. These are
programmer-visible errors, not model-visible errors.

**`ToolError`** — raised *by tools* to signal an expected failure. When a
tool raises `ToolError`, the error message goes back to the model as a tool
result. The model can reason about it, try a different approach, or ask for
clarification. This is the intended mechanism for expected failures like
"rate limit hit" or "user not found."

**`Tool`** — a named callable with a description and a JSON Schema that
describes its inputs. This is exactly the shape the Anthropic API expects.
The `fn` attribute is the actual Python function to call.

**`Turn`** — one message in the conversation history. Just a role and a list
of content blocks. The content blocks are the raw Anthropic format — no
translation layer needed.

**`AgentEvent`** — every meaningful state transition emits one of these.
Seven types, each with a payload and a timestamp. This is the audit trail.

**`AgentConfig`** — five knobs with sane defaults. You'll rarely need to
change more than `system` (your system prompt) and `sink` (where events go).

---

## Section 2: Tool Registry

The goal of the tool registry is to let you write:

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F, sunny in {city}"
```

...and have it automatically produce the JSON Schema the Anthropic API needs:

```json
{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}
```

The implementation reads the function signature with `inspect.signature()`
and resolves type annotations with `typing.get_type_hints()` (necessary
because `from __future__ import annotations` makes all annotations strings
until resolved). A simple lookup dict maps Python types to JSON Schema types:

```python
TYPE_MAP = {str: "string", int: "integer", float: "number", ...}
```

The `@tool` decorator wraps the function in a `Tool` dataclass and
registers it in a module-level `_REGISTRY` dict. The function's docstring
becomes the tool description — this is the most important thing you'll write.
The model chooses tools based on their descriptions.

`tool_fn()` is the explicit alternative for cases where you're wrapping an
external API function and want to provide the name and description separately.

---

## Section 3: Context Manager

Language models are stateless. Every API call starts fresh. Maintaining a
multi-turn conversation means your code must track the history and send it
with every request.

`ContextManager` wraps a list of `Turn` objects and handles two problems:

**Conversion** — `to_messages()` converts your `Turn` list into the
`list[dict]` format the Anthropic API expects. This is mechanical but
easy to get wrong.

**Context overflow** — models have finite context windows. If a conversation
runs long enough, you'll exceed the token limit. The sliding window solves
this by dropping old turns when the history grows too large.

The sliding window has one critical constraint: the Anthropic API requires
messages to alternate between user and assistant roles. You can't drop a
single turn — you must always drop in **pairs** (one user + one assistant).
Otherwise, you get two consecutive user messages, which the API rejects.

The `trim()` method implements this: it always removes `turns[1]` and
`turns[2]`, preserving `turns[0]` (the original user message containing the
task) and all subsequent turns. This keeps the original goal in context even
as intermediate steps age out.

---

## Section 4: Observability

Most agent frameworks emit events as log statements. The problem with log
statements is that they're strings — you can't programmatically react to them,
filter them by type, or route different events to different destinations.

nanoagent.py emits **structured events** to a **pluggable sink**. Every state
transition — turn start, model thinking, tool call, tool result, final answer,
context trim — emits an `AgentEvent` with a typed `type` field and a
structured `payload` dict.

The default sink writes one JSON line per event to stderr:

```python
def default_sink(event: AgentEvent) -> None:
    print(json.dumps({"ts": event.ts, "type": event.type, **event.payload}),
          file=sys.stderr)
```

In development, you can grep this output: `python script.py 2>&1 | grep tool_call`.

In production, swap the sink for anything — a structured logger, a metrics
counter, a distributed trace:

```python
def my_sink(event: AgentEvent) -> None:
    if event.type == "tool_call":
        metrics.increment("tool_calls", tags={"tool": event.payload["tool"]})
    logger.info("agent_event", extra=event.payload)

agent = Agent(config=AgentConfig(sink=my_sink))
```

The null sink (`lambda e: None`) suppresses all output — useful in tests.

---

## Section 5: The ReAct Loop

This is the heart of the file. Read it carefully.

`_react_loop()` takes a list of messages, a list of tools, a config, an emit
function, and a client. It returns a string (the final answer) or raises
`AgentError`.

Here's what one iteration of the loop does:

**Step 1** — Call the model. Pass all current messages, all tool definitions,
the model name, and token limit. The API returns a response with one or more
content blocks.

**Step 2** — Inspect the response. Are there any `tool_use` blocks? If not,
the model is done. Extract the text block, emit a `turn_end` event, and
return. The loop terminates here.

**Step 3** — If there are tool_use blocks, the model wants to act. First,
emit any text blocks as `thinking` events — this is the model's reasoning,
visible in your event stream.

**Step 4** — Execute each tool the model requested. Call `_execute_tool()`,
which looks up the tool by name, calls `fn(**input)`, and returns the string
result. Critically, `_execute_tool` **never raises** — any exception becomes
a structured error string that the model can reason about on the next turn.

**Step 5** — Append the assistant's response (with its tool_use blocks) and
the tool results (as a user message) to the message list. Then loop back to
Step 1.

Why does the tool result go back as a *user* message? Because the Anthropic
API models the conversation as a series of user/assistant turns. The tool
results are conceptually "the user reporting what happened when the tool ran."
Weird, but it's how the API works, and the model handles it correctly.

The loop runs at most `max_turns` times. If you hit the limit, `AgentError`
is raised. In practice, well-designed tasks with 3-5 tools rarely need more
than 10 turns. Set max_turns higher for exploratory research tasks, lower for
simple single-tool calls.

---

## Section 6: Streaming

Without streaming, you call the API and wait — potentially 10-30 seconds —
for the full response before seeing any output. With streaming, tokens arrive
as the model generates them, so users see progress immediately.

`_stream_response()` opens a streaming connection to the API and iterates
over events. When a `text_delta` event arrives, it writes the token directly
to stdout:

```python
sys.stdout.write(event.delta.text)
sys.stdout.flush()
```

The tricky part is tool_use blocks. Unlike text, tool inputs arrive as partial
JSON fragments that you can't execute until the complete JSON is assembled.
The Anthropic streaming SDK handles this internally — by the time you call
`stream.get_final_message()`, the tool_use blocks are fully assembled and
ready for the ReAct loop to process.

This is why `_stream_response()` returns a complete response object rather
than yielding events: the ReAct loop needs to inspect all content blocks
uniformly, whether or not streaming was used. The streaming/non-streaming
distinction is an implementation detail hidden behind the same response shape.

---

## Section 7: Putting It Together

Let's trace `hello_tool.py` execution step by step:

```python
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

print(run("What is 1337 + 42?"))
```

1. `@tool` decorates `add`: reads its signature, builds the JSON Schema,
   wraps it in `Tool("add", "Add two numbers together.", {...}, add)`,
   stores it in `_REGISTRY["add"]`.

2. `run("What is 1337 + 42?")` creates a fresh `Agent` with all tools from
   `_REGISTRY` and calls `agent.run(message, stream=True)`.

3. `Agent.run()` emits `turn_start`, pushes the user message to context,
   builds the message list, and calls `_react_loop()`.

4. `_react_loop()` calls the API with the user message and the `add` tool
   definition. The model responds with a `tool_use` block:
   `{name: "add", input: {a: 1337, b: 42}}`.

5. Since there's a tool_use block, the loop doesn't terminate. It calls
   `_execute_tool()` → `add(a=1337, b=42)` → `1379`. Emits `tool_call` and
   `tool_result` events.

6. Appends the assistant's response + tool result to the message list. Loops.

7. API call #2: the model now knows the result is 1379 and says "The answer
   is 1379." No tool_use blocks. The loop extracts the text, emits `turn_end`,
   returns the text.

8. Since streaming is on, the text "The answer is 1379." was already printed
   to stdout token-by-token. `print()` in hello_tool.py prints the same
   string again — you'd suppress one in production.

Total: 2 API calls, ~200 lines of framework, 0 abstractions you can't read.

---

## What This Doesn't Do (And Why)

**Persistence** — nanoagent.py makes no assumptions about storage.
`ContextManager` holds history in process memory. This is intentional.
Persistence is a deployment concern, not a reasoning concern. In production,
serialize `context.to_messages()` to a database between sessions.

**Parallel tool execution** — tools execute sequentially. When the model
returns multiple tool_use blocks, nanoagent.py runs them one by one. Parallel
execution would require `asyncio` or `threading`, adding ~30 lines and
hiding a non-trivial concurrency model. The sequential version is easier to
understand and debug. Most tasks don't benefit from parallel tool execution.

**Sandboxing** — `code_executor.py` runs code in a subprocess with a
timeout, but that's basic isolation. Production code execution needs
seccomp profiles, network namespace isolation, filesystem restrictions, or
a dedicated sandboxing service (gVisor, Firecracker). nanoagent.py provides
the hook; you supply the sandbox.

**Multi-modal** — nanoagent.py handles text only. Adding image inputs means
changing the user turn content structure to include `{"type": "image", ...}`
blocks. The ReAct loop doesn't care about content types — it's the tools and
the message structure that would need updating.

**Retry with backoff** — API calls can fail transiently. nanoagent.py doesn't
retry. In production, wrap `client.messages.create()` with `tenacity` or
equivalent.

These aren't gaps — they're deliberate exclusions. The constraint is
intentional. 294 lines that explain the algorithm beats 3000 lines that
handle every edge case.

---

## Where To Go From Here

nanoagent.py is the foundation. Here's the ecosystem narrative:

**Durable execution** — For agents that must survive crashes, restarts, and
long-running tasks, look at [Temporal](https://temporal.io/) for workflow
orchestration or [LangGraph](https://github.com/langchain-ai/langgraph) for
stateful graph-based agents.

**Observability** — The pluggable sink in nanoagent.py maps naturally to
[OpenTelemetry](https://opentelemetry.io/) spans. Each `AgentEvent` becomes
a span attribute. Route events to Jaeger, Honeycomb, or Datadog.

**Context management** — The sliding window trim strategy in nanoagent.py
is the simplest approach. More sophisticated strategies include:
summarization (compress old turns via a second model call),
semantic chunking (keep the most relevant old turns, not just the most recent),
and retrieval-augmented memory (store old turns in a vector DB, retrieve on demand).

**Production tooling** — For real deployments, wrap nanoagent's tools with
retry logic (`tenacity`), input validation (`pydantic`), and rate limiting.
The `@tool` decorator pattern composes cleanly with other decorators.

nanoagent.py isn't trying to be a production agent framework. It's trying to
make production agent frameworks legible. Once you understand this
implementation, picking up LangGraph, CrewAI, or AutoGen is a matter of
recognizing familiar patterns under unfamiliar abstractions.
