---
agentmd: "1.0"
type: rule
id: no-extra-deps-in-core
severity: error
immutable: true
description: "nanoagent.py may only import openai and stdlib modules"
applies_to: ["nanoagent.py"]
check:
  backend: ast
  language: python
  pattern: |
    (import_from_statement
      module_name: (dotted_name) @mod
      (#not-match? @mod "^(openai|json|sys|os|inspect|typing|dataclasses|functools|time|subprocess|threading)"))
  message: "Non-stdlib, non-openai import detected in nanoagent.py"
---
