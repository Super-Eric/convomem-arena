---
title: "Quick start: evaluate a personal assistant"
description: "Make a personal assistant live through a ConvoMem persona's history, let it build its own memory, then ask it the questions. Vellum is the worked example."
---

# Quick start: evaluate a personal assistant

The [model quick start](quickstart.md) hands the history to the model. A personal assistant is never handed its history: it hears each conversation once, decides what to remember, files it away, and has to find it again later. To test that, the assistant has to **live through** the conversations first.

`convomem_arena.vellum_eval` does this for [Vellum](https://github.com/vellum-ai/vellum-assistant), an open-source personal assistant with its own memory system. The same four stages apply to any assistant with a memory you can fill and query.

## How it works

1. **Plan.** Pick *N* questions per category for one persona, and merge their evidence conversations (plus filler from the same persona) into **one time-ordered history**, one day apart and ending yesterday. One memory then answers every question.
2. **Build.** Import the history into a clean Vellum instance, let Vellum's own *memory retrospective* review each conversation and call `remember`, force a consolidation pass that files the facts into concept pages, and save the result as a `vellum backup` snapshot. This is the slow part, so it is cached.
3. **Ask.** With memory writing switched off, ask each question in a new conversation. The reply is read from the turn's MLflow trace, and the injected memory from Vellum's memory-v3 pool log.
4. **Score.** Grade with ConvoMem's official judge prompts (any OpenAI-compatible endpoint; default local Ollama), plus two rough diagnostics for fact categories: is the answer anywhere in memory (capture), and was it in what got injected (retrieval).

## Run it

```bash
python -m convomem_arena.vellum_eval.run \
  --source /path/to/vellum-assistant \
  --instance vellum-rsi-eval --persona Small_Business_Owner \
  --per-category 2 --history-size 16
```

`--source` is a vellum-assistant checkout (with `cli/` and `assistant/`), and `--instance` a hatched instance. To run the stages separately, use `--stages build`, then `--stages ask,score --run-dir outputs/vellum-eval/<run-id>`. Each run writes `plan.json`, `build.json`, `answers.jsonl`, `scores.jsonl` and a `report.md` with accuracy per category.

## Before you start

!!! warning "Use a dedicated instance"
    The build restores snapshots over the instance's workspace. Never point it at an assistant you use day to day.

- **A freshly hatched instance** with `mlflowTracing.enabled`, connections and profiles for every call site, and a Claude Code token stored if call sites use the `claude-code` provider. A snapshot restore wipes that token's value, so set it again after each restore.
- **A clean shell environment.** The harness strips `VELLUM_WORKSPACE_DIR` and `*_IPC_SOCKET_DIR` inherited from your shell; a profile that exports another instance's values would otherwise make daemons bind that instance's socket.
- **One persona per instance for now.** `vellum backup` bundles exclude the Qdrant vector index, and a restore keeps the live one, so restoring persona B over persona A leaves A's vectors behind. Restoring a persona's own snapshot is fine.
- **Time.** Retrospectives take about 3–5 minutes per conversation on a local 30B model, so a 17-conversation history takes about an hour to build.

## Reading the result

A low score has four possible causes, and the diagnostics help separate them: the fact was never **captured**, **consolidation** dropped it, **retrieval** didn't inject it, or the model had it and still **answered** wrong. Below ten concept pages, Vellum's memory v3 runs a *lean profile* that skips its selector, so a small history never tests retrieval; the report says so when that happens.

!!! tip "Only want to test Vellum's model?"
    To run the ordinary benchmark with a Vellum assistant answering from a history in its prompt, see [Use a Vellum assistant as the model](../guides/vellum-as-model.md).
