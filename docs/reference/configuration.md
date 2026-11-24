---
title: "Configuration"
description: "Every taskset and task setting, the common verifiers run flags, and the TOML form."
---

# Configuration

All fields are CLI flags (snake_case → kebab-case) and can also be set in TOML under `[env.taskset]`, using the snake_case field names.

## Taskset (`--env.taskset.*`): what gets loaded

| Flag | Type | Default | Description |
|---|---|---|---|
| `categories` | list | all six | Subset of `user_facts assistant_facts changing_facts abstention preferences implicit_connections` |
| `evidence-counts` | list[int] \| None | all | Keep only items with these evidence counts |
| `context-sizes` | list[int] | `[1, 10, 30]` | Conversations per history; one task per size per item. The paper uses `1 2 3 4 5 6 10 20 30 50 70 100 150 200 300` |
| `seed` | int | `0` | Seeds item order and every mix |
| `repo-id` | str | `Salesforce/ConvoMem` | Dataset repo |
| `revision` | str | `e3e9b39…` | Dataset revision (pinned for reproducibility) |

## Task (`--env.taskset.task.*`): how each task runs and is scored

| Flag | Type | Default | Description |
|---|---|---|---|
| `memory-mode` | `context` \| `tools` \| `files` | `context` | How the history reaches the agent |
| `max-turns` | int ≥ 1 | `40` | Model calls per rollout before `turn_limit` stops it |
| `fact-scoring` | `judge` \| `exact_match` | `judge` | How the three fact categories are rewarded: the official judge, or the deterministic [exact match](../concepts/scoring.md#exact-match-for-fact-categories) with no judge call, a cheap sanity check only; use `judge` for full-scale evaluation and leaderboards |
| `judge.model` | str | verifiers default | Judge model |
| `judge.base-url` | str | Prime Inference | OpenAI-compatible judge endpoint |
| `judge.api-key-var` | str | `PRIME_API_KEY` | Name of the env var that holds the judge key |
| `tools.runtime.type` | str | `subprocess` | Where the memory MCP server runs (`tools` mode) |

## Common run flags (from verifiers)

| Flag | Meaning |
|---|---|
| `-n` / `-r` | Number of tasks / rollouts per task |
| `-m` | Agent model |
| `--client.base-url`, `--client.api-key-var` | Agent model endpoint |
| `--env.agent.harness.id` | `null`, `bash`, `claude_code`, `codex`, … |
| `--env.agent.runtime.type` | `subprocess`, `docker`, `prime` (default) |
| `--select.include.idx 0:100`, `-s` (shuffle), `--select.limit` | Task selection |
| `-o` | Output directory |

Example TOML (`convomem.toml`, used as `uv run vf-run convomem-arena @ convomem.toml -m <model>`):

```toml
[env.taskset]
categories = ["user_facts", "changing_facts", "abstention"]
context_sizes = [10, 50]
seed = 0

[env.taskset.task]
memory_mode = "tools"
max_turns = 30

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
```

`vf-validate` nests the same settings under `[taskset]` / `[taskset.task]` (its flags are `--taskset.*`) and takes the runtime as `--runtime.type`.
