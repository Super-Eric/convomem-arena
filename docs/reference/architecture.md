---
title: "Architecture"
description: "Package layout, data flow, why models.py is separate, and how to plug extra rewards from config."
---

# Architecture

```
convomem-arena/
├── pyproject.toml              # package "convomem-arena"; deps: verifiers, huggingface-hub
├── README.md                   # intro, quick starts, license
├── mkdocs.yml                  # this documentation site
├── docs/                       # documentation pages
├── uv.lock
├── convomem_arena/
│   ├── __init__.py             # exports ConvoMemTaskset (how verifiers discovers the taskset)
│   ├── taskset.py              # configs, ConvoMemTask (setup/stop/reward/metric), ConvoMemTaskset (load/build), prompts
│   ├── data.py                 # HubSource (lazy HF download), EvidenceStream (seeded round-robin), mix(), formatting
│   ├── models.py               # ConvoMemData (TaskData) and MemoryState (State), shared by taskset and tool server
│   ├── judges.py               # official ConvoMem judge prompts, RIGHT/WRONG parser, category → judge mapping
│   ├── judge_client.py         # call those judges outside a rollout (any OpenAI-compatible endpoint)
│   ├── judge_check/            # judge robustness checker: variants, perturbation generator, runner, report
│   ├── data_check/             # data quality checker: deterministic checks on sampled raw items
│   ├── text.py                 # shared word / key-term helpers for the checkers
│   └── servers/
│       └── memory.py           # MemoryToolset (MCP): BM25 index, search/read/list tools; runnable as a module
└── tests/
    ├── test_convomem.py        # offline tests (no network, no model)
    ├── test_judge_check.py     # judge checker tests with a fake generator and fake judges
    └── test_data_check.py      # data checker tests on hand-built items
```

## Data flow

```
HubSource ── list_repo_tree / hf_hub_download (cached) ──▶ raw JSON
    │
EvidenceStream.category_items()   seeded file order → items, round-robin over evidence counts
    │
round_robin(categories)           interleave categories
    │
ConvoMemTaskset.build(item, filler, size)
    ├─ mix()                      evidence + same-persona filler, seeded positions
    ├─ prompt by memory_mode      context: full transcript | tools/files: question only
    └─ ConvoMemData(...)          hidden: answer, evidence_messages, evidence_positions, conversations
```

## Why `models.py` is separate

The memory server runs as its own process (`python -m convomem_arena.servers.memory`). It receives the rollout's `TaskData` by importing its class, and it shares `MemoryState` with the task. Keeping both models in a module that imports neither the taskset nor the server avoids circular imports.

## Plugging extra rewards

verifiers lets you add or replace hooks from config without touching code. For example, to try evidence recall as a small shaping reward, write a function:

```python
# my_hooks.py
def recall_bonus(task, trace) -> float:
    found = set(task.evidence_positions) & set(trace.state.retrieved)
    return len(found) / max(len(task.evidence_positions), 1)
```

```toml
[env.taskset.task.rewards]
recall_bonus = { fn = "my_hooks.py:recall_bonus", weight = 0.1 }
```
