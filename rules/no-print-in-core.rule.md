---
agentmd: "1.0"
type: rule
id: no-print-in-core
severity: error
description: "No print() calls in nanoagent.py — use the observability sink"
applies_to: ["nanoagent.py"]
exceptions: ["nanoagent.py#default_sink"]   # the sink itself prints
check:
  backend: grep
  pattern: "^(?!.*default_sink).*\\bprint\\("
  message: "print() in nanoagent.py — use emit(AgentEvent(...)) instead"
---
