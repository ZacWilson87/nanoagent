---
agentmd: "1.0"
type: agents
scope: project
name: "nanoagent"
stack:
  - python
  - anthropic-sdk
conventions:
  - "Commits follow Conventional Commits (feat/fix/docs/chore/refactor)"
  - "nanoagent.py public API: only names listed in Section 8 are public"
  - "All private functions prefixed with _"
  - "No comments that restate what code does — only explain why"
  - "Every public function/class has a docstring"
  - "Examples must be self-contained and runnable with uv run"
  - "Zero print() statements inside nanoagent.py — use the sink"
rules:
  - rules/line-count.rule.md
  - rules/no-extra-deps.rule.md
  - rules/no-print-in-core.rule.md
agent_instructions: |
  The hard constraints are: ≤300 lines in nanoagent.py, only `anthropic`
  as a dependency in that file, all events go through the sink, not print().
  Check these before every commit. The walkthrough doc is a first-class
  deliverable — write it with the same care as the code.
---
