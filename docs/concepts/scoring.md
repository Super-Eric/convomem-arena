---
title: "Scoring"
description: "The reward, ConvoMem's official per-category judges, and the retrieval metrics recorded in tools mode."
---

# Scoring

## Reward

| Name | Weight | Value |
|---|---|---|
| `correct` | 1.0 | `1.0` if the category's ConvoMem judge says RIGHT, else `0.0`. An empty final reply scores `0.0` without calling the judge |

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
