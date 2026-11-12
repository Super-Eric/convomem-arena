# convomem-arena

**A conversational-memory environment for agentic systems, built on [verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 for the Prime Intellect Environments Hub.**

convomem-arena turns [Salesforce ConvoMem](https://huggingface.co/datasets/Salesforce/ConvoMem) ([paper](https://arxiv.org/abs/2511.10523), [reference code](https://github.com/SalesforceAIResearch/ConvoMem)) into tasks for evaluation and RL. Each task asks a question about a user's earlier conversations with the assistant: a fact the user stated, something the assistant recommended, a detail that later changed, a preference to respect, or something that was never said at all. The agent has to find the answer in a history that also contains many unrelated conversations from the same person.

The same tasks can reach the agent in three ways. That lets you measure **long-context recall** and **agentic memory use** on identical data:

| `memory_mode` | What the agent gets | Tools | Tests |
|---|---|---|---|
| `context` (default) | The full history in the prompt, using ConvoMem's official answer prompt | none | Long-context recall (the paper's baseline) |
| `tools` | Only the question. The history sits behind a per-rollout MCP memory server | `memory_search`, `memory_read_conversation`, `memory_list_conversations` | Agentic retrieval: deciding what to look up, and when |
| `files` | Only the question. The history is written to `memory/conversation_NNN.txt` in the agent's workspace | Whatever the harness provides (e.g. `bash`) | Memory as a filesystem, with general-purpose tools |

Answers are graded by **ConvoMem's official RIGHT/WRONG LLM judges**, using the matching judge for each category. In `tools` mode the environment also measures **whether the agent actually retrieved the evidence**.

---

## Contents

1. [Quick start](#quick-start)
2. [How this fits into verifiers, prime-envs, and the Hub](#how-this-fits-into-verifiers-prime-envs-and-the-hub)
3. [The dataset](#the-dataset)
4. [How a task is built](#how-a-task-is-built)
5. [Memory modes in detail](#memory-modes-in-detail)
6. [Scoring](#scoring)
7. [Configuration reference](#configuration-reference)
8. [Running evaluations](#running-evaluations)
9. [Reading results](#reading-results)
10. [Package layout and architecture](#package-layout-and-architecture)
11. [Development and testing](#development-and-testing)
12. [Publishing to the Environments Hub](#publishing-to-the-environments-hub)
13. [Design decisions](#design-decisions)
14. [Limitations and caveats](#limitations-and-caveats)
15. [Troubleshooting](#troubleshooting)
16. [Citation and license](#citation-and-license)

---

## Quick start

Requirements: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# install (from this directory)
uv venv --python 3.12
uv pip install -e ".[dev]"

# 1. check that tasks load and set up, with no model calls and no API key
uv run vf-validate convomem-arena -n 6 --only-setup --runtime.type subprocess

# 2. run a few real episodes (needs a model + judge endpoint; defaults use PRIME_API_KEY)
export PRIME_API_KEY=...
uv run vf-run convomem-arena -n 6 -r 1 -m <model> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess
```

`-n 6` already covers all six categories, because tasks are interleaved across categories ([why](#ordering-and-laziness)).

**No API key? Run it fully local against Ollama instead**, using `qwen3:30b-a3b` as both
the agent and the judge model (verified working end to end — 6/6, mean reward 1.000 on
the smoke test below):

```bash
# Ollama must be serving qwen3:30b-a3b at 127.0.0.1:11434 first:
#   ollama pull qwen3:30b-a3b && ollama run qwen3:30b-a3b "reply with one word: ok"

export OLLAMA_FAKE_KEY=unused   # value is ignored by Ollama; vf-run just needs the var set

uv run vf-run convomem-arena -n 6 -r 1 -m qwen3:30b-a3b \
  --client.base-url http://127.0.0.1:11434/v1 --client.api-key-var OLLAMA_FAKE_KEY \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.context-sizes 10 \
  --env.taskset.task.judge.base-url http://127.0.0.1:11434/v1 \
  --env.taskset.task.judge.api-key-var OLLAMA_FAKE_KEY \
  --env.taskset.task.judge.model qwen3:30b-a3b
```

A 30B model judging its own transcript is weaker on the rubric-graded categories
(`preferences`, `implicit_connections`) than a frontier judge — fine for a smoke test,
be skeptical of absolute scores there.

---

## How this fits into verifiers, prime-envs, and the Hub

| Piece | Role | In this project |
|---|---|---|
| **verifiers** (`import verifiers.v1 as vf`) | Framework: tasks, tools, judges, rewards, and the runner that puts an agent on a task and records a trace | Every class here subclasses a `vf.*` base |
| **prime-envs** | Prime Intellect's repo of first-party environments; the source of conventions (layout, prompting rules, README style) | This package follows those conventions and can drop in as `environments/<group>/convomem_arena` |
| **Environments Hub** | Registry where environment packages are published and installed by ID (`owner/name@version`) | `prime env push` publishes this folder as `convomem-arena` |

> **verifiers v0 vs. v1.** Older tutorials and many Hub environments (listed as *Legacy*) use the v0 API: `load_environment()`, `vf.SingleTurnEnv`, `vf.ToolEnv`, `vf-eval`. The current release (`verifiers>=0.4.1`) ships only **v1**: `Taskset` / `Task` / `TaskData`, `@vf.reward`, `vf.Toolset`, and the `vf-validate` / `vf-run` CLIs. This package is v1 only.

### verifiers v1 concepts, mapped to this code

| verifiers concept | What it is | Here |
|---|---|---|
| `vf.TaskData` | One immutable row: the `prompt` plus hidden fields for scoring | `ConvoMemData` in `models.py` |
| `vf.Task` | Behavior for one row: `setup`, stop conditions, toolsets, rewards, metrics | `ConvoMemTask` in `taskset.py` |
| `vf.Taskset` | Loader that yields tasks from `load()` | `ConvoMemTaskset` in `taskset.py` |
| `vf.TasksetConfig` | Load-time knobs → `--env.taskset.*` flags | `ConvoMemConfig` |
| `vf.TaskConfig` | Per-task runtime and scoring knobs → `--env.taskset.task.*` flags | `ConvoMemTaskConfig` |
| `vf.State` | Mutable per-rollout state shared between tool servers and scoring | `MemoryState` in `models.py` |
| `vf.Toolset` + `@vf.tool` | Custom tools served over MCP | `MemoryToolset` in `servers/memory.py` |
| `vf.Judge` | LLM grader: a prompt template plus a `parse` function | `ConvoMemJudge` and its subclasses in `judges.py` |
| `@vf.reward(weight=…)` | Score that adds `score × weight` to `trace.reward` | `correct` |
| `@vf.metric` | Logged measurement that never affects the reward | `memory` (tool calls, evidence recall) |
| `@vf.stop` | Predicate that ends the rollout early | `turn_limit` |
| Harness | Agent program that talks to the model (`null`, `bash`, `claude_code`, …), chosen at run time | Not fixed by this taskset |
| Runtime | Where the harness runs (`subprocess`, `docker`, `prime`), chosen at run time | Not fixed by this taskset |
| Trace | The full record of a rollout, saved to `traces.jsonl` | See [Reading results](#reading-results) |

The taskset ID maps to the module: `convomem-arena` → `import convomem_arena` → the class listed in `__all__` (`ConvoMemTaskset`). That's the only thing `convomem_arena/__init__.py` does.

### Rollout lifecycle

```
ConvoMemTaskset.load() ──yields──▶ ConvoMemTask(ConvoMemData, ConvoMemTaskConfig)
                                          │
  1. Task.setup(trace, runtime)           │  records trace.info["convomem"];
                                          │  files mode: writes memory/conversation_NNN.txt
  2. Task.toolsets(config)                │  tools mode: launches one MemoryToolset per rollout,
                                          │  which loads the history in setup_task(task_data)
  3. Harness runs the model ◀────────────┘  prompt = TaskData.prompt; MCP tools if any;
       │                                     @vf.stop turn_limit checked before each model call
  4. Scoring
       ├─ @vf.metric memory    → memory_tool_calls, evidence_recall, conversations_retrieved
       └─ @vf.reward correct   → ConvoMem judge for the category → RIGHT = 1.0, WRONG = 0.0
                                          │
  5. Trace written to traces.jsonl ◀──────┘
```

---

## The dataset

**Source:** `Salesforce/ConvoMem` on Hugging Face, pinned to revision `e3e9b39115b02346824c70d349350de738f8be41`. 75,336 question–answer items across 100 synthetic personas (CRM-flavored professions: Account Executive, Data Architect, Telemarketer, …).

The environment reads two folders:

```
core_benchmark/
├── evidence_questions/<category>_evidence/<N>_evidence/<persona-uuid>_<Title>.json   # questions + evidence conversations
└── filler_conversations/<persona-uuid>_<Title>.json                                  # 400 unrelated conversations per persona
```

Each evidence item looks like this, abridged from a real record:

```json
{
  "question": "For the new software vendor contract, what was the precise liability cap in USD that I established as my absolute non-negotiable limit?",
  "answer": "You set your absolute non-negotiable liability cap for the new vendor at $2,500,000.",
  "message_evidences": [{"speaker": "User", "text": "...my absolute walk-away is a $2.5M cap..."}],
  "conversations": [{"id": "…", "messages": [{"speaker": "User", "text": "…"}, {"speaker": "Assistant", "text": "…"}]}],
  "personId": "af47b47b-…", "category": "Professional Life", "scenario_description": "…"
}
```

### Categories

| Category (`--env.taskset.categories`) | Folder | What it tests | Evidence counts | Example |
|---|---|---|---|---|
| `user_facts` | `user_evidence` | Recall of facts the user stated, possibly combined across messages | 1–6 | "What liability cap did I set?" → "$2,500,000" |
| `assistant_facts` | `assistant_facts_evidence` | Recall of what the assistant itself said or recommended | 1–6 | "What KPI did you suggest for the advocacy program?" → "Advocate Sourced Revenue" |
| `changing_facts` | `changing_evidence` | Tracking updates; only the latest value is correct | 2–6 | "When is the CRM update scheduled?" → the last rescheduled date |
| `abstention` | `abstention_evidence` | Recognizing that the answer was never given | 1–3 | "What is the projected contract value…?" → should say it doesn't know |
| `preferences` | `preference_evidence` | Recommendations that respect stated preferences; the answer is a **rubric** | 1–2 | "Recommend a lightweight PM tool" → rubric: simple, visual tools |
| `implicit_connections` | `implicit_connection_evidence` | Connecting indirect context; the answer is a **rubric** | 1–3 | "How should we streamline proposals?" → rubric: agile, low-bureaucracy |

The **evidence count** is the number of conversations the evidence is spread across. Multi-evidence items require combining information from several conversations.

Speaker labels in the raw data mix `User`/`user` and `Assistant`/`assistant`. They're normalized to `User` / `Assistant`.

---

## How a task is built

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

### Ordering and laziness

`load()` is a **generator**, and dataset files download from the Hub only when first needed (they're cached in the standard HF cache). Tasks are interleaved:

- round-robin **across categories** (in `categories` order), and within each category
- round-robin **across evidence counts**, then
- persona files and items in a **seeded shuffled order**, then
- one task per **context size** for each item.

So `-n 6 --env.taskset.context-sizes 10` gives one task from each category and downloads only a few MB. `-n 12` with two context sizes gives the same six items at both sizes. That makes context-size comparisons paired.

---

## Memory modes in detail

### `context`: the long-context baseline

The prompt is ConvoMem's official answer prompt (`EvaluationUtils.getModelAnswerPrompt`), with the history laid out exactly as `buildConversationContext` does it:

```
Answer the question based on the conversations below. If the User in the conversation refers to themselves ("I", "me", "my"), they are the person being asked about. Be direct and factual.

Conversation 1:
User: I'm trying to decide on a hike for this weekend…
Assistant: …

Conversation 2:
…

Question: <question>

Answer:
```

A conversation averages roughly **6,000 characters (~1.5k tokens)**. Rough prompt sizes:

| `context_size` | ≈ prompt tokens |
|---|---|
| 1 | 1.5k |
| 10 | 15k |
| 30 | 45k |
| 100 | 150k |
| 300 | 450k |

In this mode `TaskData.conversations` is left empty, because the history is already in the prompt and storing it twice would double memory.

### `tools`: agentic memory over MCP

The prompt contains only the question and the number of stored conversations:

```
Answer the question based on your past conversations with the User. The 20 conversations, in chronological order, are available through the memory tools. If the User in the conversations refers to themselves ("I", "me", "my"), they are the person being asked about. Be direct and factual.

Question: <question>
```

A **per-rollout** MCP server (`MemoryToolset`, `TOOL_PREFIX = "memory"`) receives the task's history in `setup_task` and exposes:

| Tool | Signature | Returns |
|---|---|---|
| `memory_search` | `(query: str, limit: int = 8)` | Top messages by Okapi BM25 (k1=1.5, b=0.75, light stopword list), each with `conversation`, `message`, `speaker`, full `text`. `limit` is clamped to 1–20 |
| `memory_read_conversation` | `(conversation: int, start: int = 1, count: int = 40)` | Consecutive messages of one conversation (1-based numbering) plus `total_messages`. `count` is clamped to 1–100 |
| `memory_list_conversations` | `(offset: int = 0, limit: int = 50)` | `total_conversations` and, per conversation, `num_messages` and the first 200 characters of its opening user message |

Every search or read records which conversations it surfaced on the shared `MemoryState`, which scoring reads back as retrieval metrics. Listing counts as a tool call but not as retrieval.

This is the mode for long histories (100–300 conversations) that don't fit in context, which is the regime the ConvoMem paper studies with memory systems.

### `files`: memory as a filesystem

`Task.setup` writes one file per conversation (`memory/conversation_001.txt` is the oldest) relative to the runtime's working directory. The prompt points the agent there and asks for the answer as the final message. There are no custom tools: the agent uses whatever the harness gives it (`bash`'s `grep`/`cat`, Claude Code's file tools, …). That follows prime-envs' preference for general harness tools over custom ones. **Run this mode on `docker` or `prime`** (see [Limitations](#limitations-and-caveats)).

---

## Scoring

### Reward

| Name | Weight | Value |
|---|---|---|
| `correct` | 1.0 | `1.0` if the category's ConvoMem judge says RIGHT, else `0.0`. An empty final reply scores `0.0` without calling the judge |

### Judges

The judge prompts are copied **verbatim** from the reference implementation (`questions/evidence/generation/AnsweringEvaluation.scala`). Each category uses the same evaluator the paper does:

| Category | Reference evaluator | Class | Judge sees |
|---|---|---|---|
| `user_facts` (1 evidence) | `UserFactsAnsweringEvaluation` (single) | `UserFactsSingleJudge` | question, evidence message, gold answer, response |
| `user_facts` (2+ evidence) | `UserFactsAnsweringEvaluation` (multi) | `UserFactsMultiJudge` | question, numbered evidence messages, gold answer, response |
| `assistant_facts`, `changing_facts` | `DefaultAnsweringEvaluation` | `ConvoMemJudge` | question, gold answer, response |
| `abstention` | `AbstentionAnsweringEvaluation` | `AbstentionJudge` | question, response (a refusal or "I don't know" is RIGHT) |
| `preferences`, `implicit_connections` | `RubricBasedAnsweringEvaluation` | `RubricAnswerJudge` | question, rubric, response (must satisfy **all** criteria) |

**Verdict parsing** follows the reference: a reply containing only RIGHT → pass; WRONG, or both RIGHT and WRONG → fail; neither → **raise**. Raising rather than returning 0.0 means a broken judge shows up as an error instead of being recorded as a model mistake.

The judge endpoint defaults to verifiers' `JudgeConfig` (Prime Inference, key from `PRIME_API_KEY`). The paper used GPT-4o-mini and Gemini Flash during data validation. Pick a judge and keep it fixed when comparing runs.

### Metrics (recorded in `tools` mode only)

| Name | Meaning |
|---|---|
| `memory_tool_calls` | Number of memory tool calls in the rollout |
| `evidence_recall` | Fraction of evidence conversations returned by at least one `memory_search` or `memory_read_conversation` call |
| `conversations_retrieved` | Distinct conversations the agent retrieved |

These are **metrics, not rewards**, so they're logged for analysis and never trained on. Rewarding recall directly would encourage the agent to dump the whole history. To experiment with it as a shaping signal, see [plugged rewards](#plugging-extra-rewards).

---

## Configuration reference

All fields are CLI flags (snake_case → kebab-case) and can also be set in TOML under `[env.taskset]`, using the snake_case field names.

### Taskset (`--env.taskset.*`): what gets loaded

| Flag | Type | Default | Description |
|---|---|---|---|
| `categories` | list | all six | Subset of `user_facts assistant_facts changing_facts abstention preferences implicit_connections` |
| `evidence-counts` | list[int] \| None | all | Keep only items with these evidence counts |
| `context-sizes` | list[int] | `[1, 10, 30]` | Conversations per history; one task per size per item. The paper uses `1 2 3 4 5 6 10 20 30 50 70 100 150 200 300` |
| `seed` | int | `0` | Seeds item order and every mix |
| `repo-id` | str | `Salesforce/ConvoMem` | Dataset repo |
| `revision` | str | `e3e9b39…` | Dataset revision (pinned for reproducibility) |

### Task (`--env.taskset.task.*`): how each task runs and is scored

| Flag | Type | Default | Description |
|---|---|---|---|
| `memory-mode` | `context` \| `tools` \| `files` | `context` | How the history reaches the agent |
| `max-turns` | int ≥ 1 | `40` | Model calls per rollout before `turn_limit` stops it |
| `judge.model` | str | verifiers default | Judge model |
| `judge.base-url` | str | Prime Inference | OpenAI-compatible judge endpoint |
| `judge.api-key-var` | str | `PRIME_API_KEY` | Name of the env var that holds the judge key |
| `tools.runtime.type` | str | `subprocess` | Where the memory MCP server runs (`tools` mode) |

### Common run flags (from verifiers)

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

---

## Running evaluations

```bash
# Long-context baseline at two history lengths (paired: same items at both sizes)
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.context-sizes 10 30

# Agentic memory on histories too long for context
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.task.memory-mode tools --env.taskset.context-sizes 100 300

# Memory as files, solved by a coding harness in an isolated container
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.taskset.task.memory-mode files --env.agent.harness.id bash --env.agent.runtime.type docker

# One category only, harder multi-evidence items
uv run vf-run convomem-arena -n 20 -m <model> --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.categories changing_facts --env.taskset.evidence-counts 4 5 6

# Custom judge endpoint (any OpenAI-compatible API)
uv run vf-run convomem-arena -n 6 -m <model> ... \
  --env.taskset.task.judge.base-url https://api.openai.com/v1 \
  --env.taskset.task.judge.api-key-var OPENAI_API_KEY \
  --env.taskset.task.judge.model gpt-4.1-mini
```

For large-scale evaluation and RL, use prime-rl (`uv run eval convomem-arena …` from a prime-rl checkout with this package installed).

### Suggested experiments

- **Context-length curve:** `context` mode at sizes `1 10 30 50 100`, accuracy per category. The paper reports that long context degrades as histories grow.
- **Context vs. tools:** same items and sizes, `memory-mode context` vs. `tools`. Where does retrieval overtake the full history?
- **Retrieval vs. reasoning failures:** in `tools` mode, compare `correct` with `evidence_recall`. Low recall with a wrong answer is a search failure. High recall with a wrong answer is a reasoning or synthesis failure, which is common on `changing_facts`.
- **Abstention calibration:** does more history (more distractors) make the agent hallucinate more on `abstention`?

---

## Reading results

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
    print(f"{cat:<22} ctx={size:<4} acc={sum(xs)/len(xs):.3f} n={len(xs)}")
```

---

## Package layout and architecture

```
convomem-arena/
├── pyproject.toml              # package "convomem-arena"; deps: verifiers, huggingface-hub
├── README.md
├── uv.lock
├── convomem_arena/
│   ├── __init__.py             # exports ConvoMemTaskset (how verifiers discovers the taskset)
│   ├── taskset.py              # configs, ConvoMemTask (setup/stop/reward/metric), ConvoMemTaskset (load/build), prompts
│   ├── data.py                 # HubSource (lazy HF download), EvidenceStream (seeded round-robin), mix(), formatting
│   ├── models.py               # ConvoMemData (TaskData) and MemoryState (State), shared by taskset and tool server
│   ├── judges.py               # official ConvoMem judge prompts, RIGHT/WRONG parser, category → judge mapping
│   └── servers/
│       └── memory.py           # MemoryToolset (MCP): BM25 index, search/read/list tools; runnable as a module
└── tests/
    └── test_convomem.py        # offline tests (no network, no model)
```

### Data flow

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

### Why `models.py` is separate

The memory server runs as its own process (`python -m convomem_arena.servers.memory`). It receives the rollout's `TaskData` by importing its class, and it shares `MemoryState` with the task. Keeping both models in a module that imports neither the taskset nor the server avoids circular imports.

### Plugging extra rewards

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

---

## Development and testing

```bash
uv pip install -e ".[dev]" ruff

uv run pytest tests                                  # 17 offline tests
uv run ruff check convomem_arena tests
uv run ruff format --check convomem_arena tests

# setup check against real Hub data, all three modes
for m in context tools files; do
  uv run vf-validate convomem-arena -n 6 --only-setup --runtime.type subprocess \
    --taskset.task.memory-mode $m --taskset.context-sizes 10
done
```

The tests cover: speaker normalization; mixing (evidence kept, order preserved, determinism, contexts smaller than the evidence count); the reference transcript layout; RIGHT/WRONG parsing; the category → judge mapping; evidence substitution in judge prompts; BM25 search, paging and bounds; round-robin ordering; prompt construction per mode; and seed reproducibility. Hub access is replaced by an in-memory `FakeSource`.

**Testing the whole pipeline without an API key:** the agent client and the judge both take a `base_url`. Pointing both at a small scripted OpenAI-compatible server (one that calls `memory_search`, answers from the result, and returns "RIGHT" to judge prompts) exercises tool-server launch, state sync, metrics and judge dispatch:

```bash
uv run vf-run convomem-arena -n 6 -r 1 -m fake \
  --client.base-url http://127.0.0.1:8765/v1 --client.api-key-var FAKE_KEY \
  --env.taskset.task.judge.base-url http://127.0.0.1:8765/v1 --env.taskset.task.judge.api-key-var FAKE_KEY \
  --env.agent.harness.id null --env.agent.runtime.type subprocess --env.taskset.task.memory-mode tools
```

---

## Publishing to the Environments Hub

Following the Prime Intellect docs (install the Prime CLI as described at <https://docs.primeintellect.ai>; these commands haven't been run from this repo yet):

```bash
prime login
prime env push                 # from this directory; add --visibility=PRIVATE to keep it private
prime env push --auto-bump     # later releases: bump the version automatically
```

Others can then install it with `prime env install <owner>/convomem-arena` and run it by ID. verifiers strips `owner/` and `@version` and imports `convomem_arena`.

Before pushing:
- bump `version` in `pyproject.toml` (or use `--auto-bump`)
- update the changelog below
- run the tests and a small `vf-run`, and inspect `traces.jsonl`
- remember the dataset's **non-commercial** license (below)

To contribute to `prime-envs` instead, place this folder at `environments/<group>/convomem_arena`, install it with `uv pip install -e environments/<group>/convomem_arena` from the repo root, and optionally add a `[tool.prime-envs.ci]` table to `pyproject.toml` to tune CI.

---

## Design decisions

| Decision | Reason |
|---|---|
| **verifiers v1**, not v0 | The current release ships only v1; v0 environments are listed as Legacy on the Hub |
| **Mix on the fly** instead of using the 25.7 GB `pre_mixed_testcases` | Same official algorithm, any context size, and a few MB per small run instead of ~40 MB files |
| **Same-persona filler, evidence order preserved** | Matches the reference generator; ordering defines the answer for `changing_facts` |
| **Official answer and judge prompts, verbatim** | Comparable with the paper; the instructions match the reward |
| **Judge failures raise** | Infrastructure errors must not be scored as wrong answers |
| **Per-task toolset** (not shared) | Each rollout has a different history; per-rollout servers isolate state |
| **History out of the prompt in tools/files mode** | The agent must actually use memory; the history lives in hidden `TaskData` fields |
| **Recall as a metric, not a reward** | Observing retrieval without incentivizing history dumps |
| **Generic tool descriptions** | prime-envs guidance: describe what a tool does, not how to use it for this task |
| **`files` mode** | prime-envs prefers general harness tools over custom ones; this lets any coding agent use its own |
| **Lazy generator plus round-robin** | `-n` stays cheap and still covers every category |
| **No role prompt** ("You are a memory assistant…") | prime-envs guidance for RL environments |

---

## Limitations and caveats

- **Answer leakage on `subprocess` with shell harnesses.** With `bash` (or any shell-capable harness) on the `subprocess` runtime, the agent's shell runs on the evaluator host, which also holds the Hugging Face cache with gold answers. Use the `null` harness for `context` and `tools` modes, and `docker` or `prime` for `files` mode or any shell harness.
- **Not the paper's exact samples.** The mixing algorithm and judges are official, but samples are drawn here rather than taken from `pre_mixed_testcases`, so results are comparable in protocol but not item-for-item.
- **The judge model affects scores.** Rubric categories are especially sensitive to the choice of judge. Keep the judge fixed across compared runs.
- **`--select.shuffle` materializes the whole selection,** which downloads every needed evidence and filler file (up to ~1.2 GB evidence + ~560 MB filler, and a lot of RAM for large context sizes). Narrow the selection with `categories`, `evidence-counts` and `context-sizes` first.
- **Uneven folder sizes.** Some category/evidence-count folders are small (e.g. `user_facts` with 4 evidence conversations). Round-robin draws from them until they run out.
- **BM25 is lexical.** `memory_search` won't match paraphrases. That's part of the challenge, and the agent can fall back to listing and reading conversations.
- **Untested here:** `files` mode has passed setup validation, but no full model run in that mode has been recorded yet.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `taskset 'convomem-arena' not found` | Install the package into the environment running the CLI: `uv pip install -e .` |
| Auth error from the judge or the model | Set `PRIME_API_KEY`, or point `--client.*` / `--env.taskset.task.judge.*` at another endpoint and key variable |
| `judge returned neither RIGHT nor WRONG` | The judge model isn't following the format; use a stronger or instruction-tuned judge |
| Tool server times out at startup | Raise `--env.taskset.task.tools.timeout.startup`; check that the package is installed for the server's Python |
| Context-length errors in `context` mode | Lower `context-sizes`, or switch to `memory-mode tools` |
| Rollouts end with `turn_limit` | Raise `--env.taskset.task.max-turns`, or inspect the trace for search loops |
| Slow first run | First-time Hub downloads; later runs use the HF cache (`HF_HOME`) |

---

## Citation and license

If you use this environment, please cite ConvoMem:

> ConvoMem, Salesforce AI Research, arXiv:2511.10523 (2025). Use the BibTeX from <https://arxiv.org/abs/2511.10523>.

- **Dataset:** Salesforce/ConvoMem, **CC-BY-NC-4.0** (non-commercial use only).
- **Judge prompts:** adapted verbatim from SalesforceAIResearch/ConvoMem; see that repository's license.

---

## Changelog

- 2026-11-09: Initial release. verifiers v1 taskset with `context` / `tools` / `files` memory modes, official ConvoMem mixing and judges, and retrieval metrics.
