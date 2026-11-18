---
title: "Limitations and caveats"
description: "Known limits: answer leakage with shell harnesses, sample comparability, judge sensitivity and more."
---

# Limitations and caveats

- **Answer leakage on `subprocess` with shell harnesses.** With `bash` (or any shell-capable harness) on the `subprocess` runtime, the agent's shell runs on the evaluator host, which also holds the Hugging Face cache with gold answers. Use the `null` harness for `context` and `tools` modes, and `docker` or `prime` for `files` mode or any shell harness.
- **Not the paper's exact samples.** The mixing algorithm and judges are official, but samples are drawn here rather than taken from `pre_mixed_testcases`, so results are comparable in protocol but not item-for-item.
- **The judge model affects scores.** Rubric categories are especially sensitive to the choice of judge. Keep the judge fixed across compared runs.
- **`--select.shuffle` materializes the whole selection,** which downloads every needed evidence and filler file (up to ~1.2 GB evidence + ~560 MB filler, and a lot of RAM for large context sizes). Narrow the selection with `categories`, `evidence-counts` and `context-sizes` first.
- **Uneven folder sizes.** Some category/evidence-count folders are small (e.g. `user_facts` with 4 evidence conversations). Round-robin draws from them until they run out.
- **BM25 is lexical.** `memory_search` won't match paraphrases. That's part of the challenge, and the agent can fall back to listing and reading conversations.
- **Untested here:** `files` mode has passed setup validation, but no full model run in that mode has been recorded yet.
