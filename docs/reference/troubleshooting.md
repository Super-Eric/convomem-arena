---
title: "Troubleshooting"
description: "Common errors and how to fix them."
---

# Troubleshooting

| Symptom | Fix |
|---|---|
| `taskset 'convomem-arena' not found` | Install the package into the environment running the CLI: `uv pip install -e .` |
| Auth error from the judge or the model | Set `PRIME_API_KEY`, or point `--client.*` / `--env.taskset.task.judge.*` at another endpoint and key variable |
| `judge returned neither RIGHT nor WRONG` | The judge model isn't following the format; use a stronger or instruction-tuned judge |
| Tool server times out at startup | Raise `--env.taskset.task.tools.timeout.startup`; check that the package is installed for the server's Python |
| Context-length errors in `context` mode | Lower `context-sizes`, or switch to `memory-mode tools` |
| Rollouts end with `turn_limit` | Raise `--env.taskset.task.max-turns`, or inspect the trace for search loops |
| Slow first run | First-time Hub downloads; later runs use the HF cache (`HF_HOME`) |
