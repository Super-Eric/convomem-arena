---
title: "Judge robustness"
description: "Change one judge input at a time and check whether the verdict stays or flips as it should."
---

# Judge robustness

Every score here is only as good as its judge, so `convomem_arena.judge_check` tests a judge model with the same rendered prompts and parser the environment uses. It changes **one input at a time** and checks whether the verdict behaves:

| Variant | Changes | A robust judge |
|---|---|---|
| `question_paraphrase` | question reworded, same meaning | keeps its verdict |
| `evidence_paraphrase` | evidence reworded, same facts | keeps its verdict |
| `question_negated` | question reversed or negated | flips RIGHT → WRONG (abstention: keeps RIGHT, a refusal still refuses) |
| `evidence_changed` | evidence now contradicts the answer | flips RIGHT → WRONG |
| `reference_changed` | gold answer replaced with a different one | flips RIGHT → WRONG |
| `response_wrong` | response replaced with a plausible wrong one | flips RIGHT → WRONG |

The baseline response is the gold answer itself (for the two rubric categories, a generated reply that satisfies the rubric), so the baseline should be RIGHT. A variant is only run against a judge that can see the input it changes: only the user-facts prompts show evidence, and the abstention prompt shows no gold answer. Each check is repeated, and the majority verdict is used; repeats that disagree are reported as the judge's noise floor.

```bash
# 1. sample items and generate perturbations with a generator model (cached; review review.md)
python -m convomem_arena.judge_check perturb --per-category 4 --generator-think \
  --out outputs/judge-check/run1

# 2. grade baseline and variants with each judge, 3 repeats each (resumable)
python -m convomem_arena.judge_check judge --run-dir outputs/judge-check/run1 \
  --judge-model qwen3:30b-a3b --judge-model gpt-oss:20b --repeats 3

# 3. score and write report.md
python -m convomem_arena.judge_check report --run-dir outputs/judge-check/run1
```

**First results (2026-11-17, 24 items, 4 per category, 3 repeats, temperature 0):**

| Judge | Correct response judged RIGHT | Paraphrase invariance | Wrong response / changed reference → WRONG | Negated question → WRONG | Contradicted evidence → WRONG | Unstable groups |
|---|---|---|---|---|---|---|
| `qwen3:30b-a3b` | 23/24 | 30/30 | 42/42 | 0/19 | 0/4 | 4/122 |
| `gpt-oss:20b` | 22/24 | 29/30 | 40/40 | 0/18 | 0/4 | 0/122 |

Both judges catch a wrong response or a changed reference every time and ignore rewording, but neither notices when the question no longer asks what the response answers, or when the evidence contradicts it: they grade **response against reference**, not response against question or evidence. So a ConvoMem score from these judges says "the reply states the gold fact", nothing more. The misses on correct responses are all rubric categories, where the prompt demands every criterion. qwen's 4 unstable groups (3 on negated questions) show it is noisiest exactly where it is uncertain. Small samples (only 4 items exercise evidence) make these indications, not measurements.

Endpoints default to local Ollama; any OpenAI-compatible endpoint works with `--judge-base-url` and `--judge-api-key-var`. Perturbations are machine-generated and checked only mechanically (present, actually different, numbers kept in paraphrases), so read `review.md` before trusting a report: a bad paraphrase makes a robust judge look fragile.
