---
title: "How a task is built"
description: "Mixing evidence into same-persona filler with ConvoMem's official algorithm, task IDs, and lazy ordering."
---

# How a task is built

For every evidence item and every requested context size, the taskset builds one task. Mixing follows ConvoMem's official `StandardTestCasesGenerator`:

1. **Keep all evidence.** Every evidence conversation of the item is included, even when `context_size` is smaller than the evidence count. The context size is a target for dilution, not a hard cap.
2. **Sample filler from the same persona.** Draw `context_size − evidence_count` conversations from that persona's 400 filler conversations, so the distractors are on-topic for the same person.
3. **Place the evidence at random sorted positions.** Evidence conversations go at random positions in their **original order**, which matters for `changing_facts`, where the order defines which value is current.
4. **Seed everything.** The RNG is `random.Random(f"{seed}:{task_id}")`, so a given seed and task ID always produce the same history.

The task ID encodes the full provenance and doubles as the task key:

```
user_evidence/1_evidence/af47b47b-…_Procurement_Manager/17/ctx30
└─ folder ──┘└─ count ─┘└──────── persona file ────────┘└item┘└size┘
```

## Ordering and laziness

`load()` is a **generator**, and dataset files download from the Hub only when first needed (they're cached in the standard HF cache). Tasks are interleaved:

- round-robin **across categories** (in `categories` order), and within each category
- round-robin **across evidence counts**, then
- persona files and items in a **seeded shuffled order**, then
- one task per **context size** for each item.

So `-n 6 --env.taskset.context-sizes 10` gives one task from each category and downloads only a few MB. `-n 12` with two context sizes gives the same six items at both sizes. That makes context-size comparisons paired.
