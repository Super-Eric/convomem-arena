---
title: "Reading results"
description: "What vf-run writes to traces.jsonl, and how to compute per-category accuracy."
---

# Reading results

`vf-run` writes `<output_dir>/<run-name>/traces.jsonl` with one episode per line. Useful fields:

| Path | Content |
|---|---|
| `task.data.id` | Provenance ID (folder / count / persona / item / ctx) |
| `task.data.category`, `.evidence_count`, `.context_size`, `.num_conversations` | Slicing keys |
| `task.data.question`, `.answer`, `.evidence_positions` | Ground truth |
| `traces[0].rewards.correct.score` | 1.0 / 0.0 |
| `traces[0].metrics` | `memory_tool_calls`, `evidence_recall`, `conversations_retrieved` (tools mode) |
| `traces[0].info.convomem` | Category, evidence count, context size, mode |
| `traces[0].info.judge_calls` | Exact judge request and response |
| `traces[0].stop_condition` | e.g. `agent_completed`, `turn_limit` |

Per-category accuracy:

```python
import json, collections, sys

acc = collections.defaultdict(list)
for line in open(sys.argv[1]):
    ep = json.loads(line)
    d, t = ep["task"]["data"], ep["traces"][0]
    acc[(d["category"], d["context_size"])].append(t["rewards"]["correct"]["score"])

for (cat, size), xs in sorted(acc.items()):
    print(f"{cat:<22} ctx={size:<4} acc={sum(xs) / len(xs):.3f} n={len(xs)}")
```
