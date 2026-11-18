---
title: "Scoring"
description: "The reward, ConvoMem's official per-category judges, and the retrieval metrics recorded in tools mode."
---

# Scoring

## Reward

| Name | Weight | Value |
|---|---|---|
| `correct` | 1.0 | `1.0` if the category's ConvoMem judge says RIGHT, else `0.0`. An empty final reply scores `0.0` without calling the judge. With `fact-scoring exact_match`, the three fact categories use the deterministic check below instead |

## Exact match for fact categories

The official judges make scores comparable with the ConvoMem paper, but each one is a model call: it costs time and money, needs an endpoint, and can vary between runs. For `user_facts`, `assistant_facts` and `changing_facts` there is a deterministic alternative.

ConvoMem's gold answers are full sentences, so a literal comparison almost never matches. Instead a reply **matches when it contains the answer's details**: the names and numbers the question doesn't already contain. *"The concert has been postponed to April 10th of next year"* needs `april` and `10`. Answers with no names or numbers fall back to their content words. Two guards keep it from being gamed when used as a training reward:

- **No dumping:** a reply longer than max(100, 5 × the answer's words) scores 0.
- **No stale numbers** (`changing_facts`): a reply that also states a number from an earlier evidence message, one the latest message and the answer no longer contain, scores 0. *"October 5th or April 10th"* fails; *"postponed from the original October date to April 10th"* passes.

It is recorded as the `exact_match` metric on every fact-category rollout, whatever the reward uses, so the two can be compared on any run. To make it the reward and skip the judge for those categories:

```bash
vf-run convomem-arena ... --env.taskset.task.fact-scoring exact_match
```

The other three categories always use their judge: abstention and the two rubric categories have no string to match.

**How well it agrees with the judge.** On 36 replies the judge had graded (24 from local runs and 12 from a Vellum memory evaluation), exact match gave the same verdict every time; the guards were tuned on the first 24, and the 12 were held out. On judge-robustness cases with known answers it accepted all 12 correct replies, including under a reworded question, and rejected 11 of 12 planted wrong replies. The miss shows its limit: in *"a Caprese salad, a tomato basil pasta sauce and pesto"* only "Caprese" is a name, so a reply that gets the other two dishes wrong still passes. Requiring most content words as well catches that case but rejects correct paraphrases, so it is left out.

## Judges

The judge prompts are copied **verbatim** from the reference implementation (`questions/evidence/generation/AnsweringEvaluation.scala`). Each category uses the same evaluator the paper does:

| Category | Reference evaluator | Class | Judge sees |
|---|---|---|---|
| `user_facts` (1 evidence) | `UserFactsAnsweringEvaluation` (single) | `UserFactsSingleJudge` | question, evidence message, gold answer, response |
| `user_facts` (2+ evidence) | `UserFactsAnsweringEvaluation` (multi) | `UserFactsMultiJudge` | question, numbered evidence messages, gold answer, response |
| `assistant_facts`, `changing_facts` | `DefaultAnsweringEvaluation` | `ConvoMemJudge` | question, gold answer, response |
| `abstention` | `AbstentionAnsweringEvaluation` | `AbstentionJudge` | question, response (a refusal or "I don't know" is RIGHT) |
| `preferences`, `implicit_connections` | `RubricBasedAnsweringEvaluation` | `RubricAnswerJudge` | question, rubric, response (must satisfy **all** criteria) |

**Verdict parsing** follows the reference: a reply containing only RIGHT → pass; WRONG, or both RIGHT and WRONG → fail; neither → **raise**. Raising rather than returning 0.0 means a broken judge shows up as an error instead of being recorded as a model mistake.

The judge endpoint defaults to verifiers' `JudgeConfig` (Prime Inference, key from `PRIME_API_KEY`). The paper used GPT-4o-mini and Gemini Flash during data validation. Pick a judge and keep it fixed when comparing runs, and check how far you can trust it with the [judge robustness checker](../quality/judge-robustness.md).

## Metrics (recorded in `tools` mode only)

| Name | Meaning |
|---|---|
| `memory_tool_calls` | Number of memory tool calls in the rollout |
| `evidence_recall` | Fraction of evidence conversations returned by at least one `memory_search` or `memory_read_conversation` call |
| `conversations_retrieved` | Distinct conversations the agent retrieved |

These are **metrics, not rewards**, so they're logged for analysis and never trained on. Rewarding recall directly would encourage the agent to dump the whole history. To experiment with it as a shaping signal, see [plugged rewards](../reference/architecture.md#plugging-extra-rewards).
