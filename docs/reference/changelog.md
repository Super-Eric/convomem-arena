---
title: "Changelog"
description: "What changed, and when."
---

# Changelog

- 2026-11-23: Exact match hardened after [issue #11](https://github.com/rahmansunny071/convomem-arena/issues/11): denied, hedged and retracted answers, and lists of candidate names, now score 0; answers and replies are read the same way (accents, `a.m.`, letters glued to digits, hyphenated and possessive names, numbers with separators), so every gold answer matches itself.
- 2026-11-18: Optional deterministic scoring for the fact categories (`fact-scoring exact_match`), and an `exact_match` metric recorded on every fact-category rollout.
- 2026-11-17: Data quality checker (`python -m convomem_arena.data_check`): structural checks, answer grounding, changing-fact order, filler contamination and duplicates on a persona-spread sample.
- 2026-11-16: Judge robustness checker (`python -m convomem_arena.judge_check`): one-input-at-a-time paraphrase and meaning-change variants, repeated verdicts, per-judge report. The standalone judge client moved to `convomem_arena.judge_client`.

- 2026-11-09: Initial release. verifiers v1 taskset with `context` / `tools` / `files` memory modes, official ConvoMem mixing and judges, and retrieval metrics.
