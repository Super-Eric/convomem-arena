---
title: "Design decisions"
description: "The choices behind convomem-arena and the reason for each."
---

# Design decisions

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
