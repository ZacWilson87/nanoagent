---
agentmd: "1.0"
type: rule
id: nanoagent-line-count
severity: error
immutable: true
description: "nanoagent.py must be ≤300 lines"
applies_to: ["nanoagent.py"]
check:
  backend: grep
  pattern: "."           # count non-empty lines
  message: "nanoagent.py exceeds 300 lines — refactor before committing"
---
