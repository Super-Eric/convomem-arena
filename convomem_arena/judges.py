"""ConvoMem's official RIGHT/WRONG judges, one per evaluation type.

Prompts are copied from the ConvoMem reference implementation
(SalesforceAIResearch/ConvoMem, `questions/evidence/generation/AnsweringEvaluation.scala`),
and each category uses the same evaluation as the paper:

- user_facts: UserFactsAnsweringEvaluation (single- or multi-evidence variant)
- assistant_facts, changing_facts: DefaultAnsweringEvaluation
- abstention: AbstentionAnsweringEvaluation
- preferences, implicit_connections: RubricBasedAnsweringEvaluation (the answer is a rubric)
"""

from __future__ import annotations

import re

import verifiers.v1 as vf

DEFAULT_PROMPT = """\
I will provide you with a **Question**, a **Correct Answer**, and a **Model's Response**. Your sole task is to determine if the Model's Response is **sufficiently correct and complete** to answer the Question, when compared against the Correct Answer.

**Crucial Guidelines for your judgment:**

1.  **Core Information is Key**: The Model's Response must contain all the **essential factual information** that directly answers the Question.
2.  **Equivalence Counts**: Phrasing doesn't need to be identical. If the Model's Response conveys the exact same core meaning and details as the Correct Answer, even if paraphrased or structured differently, consider it correct.
3.  **Superfluous (but Accurate) Information**: If the Model's Response includes additional details that were *not explicitly asked for* by the Question, but these details are **accurate and do not contradict** the Correct Answer, you should **still count it as correct** if the core question is fully answered.
4.  **Partial Answers are Incorrect**: If the Model's Response is missing any essential information directly requested by the Question (even if the Correct Answer provides more detail), it is incorrect.
5.  **Focus on the Question**: Your primary focus should be whether the Model's Response adequately addresses the Question using information that aligns with the Correct Answer. Do not penalize the model for not reiterating every single word or incidental detail from the Correct Answer if it wasn't requested.

**Answer only "RIGHT" or "WRONG". Do not provide any additional text, explanations, or reasoning.**

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Answer (RIGHT/WRONG):"""

RUBRIC_PROMPT = """\
You are a judge evaluating if a model's response meets the criteria in a rubric.

Question asked: {question}

Evaluation rubric: {answer}

Model's response: {response}

Guidelines:
- The model's response must fully satisfy ALL criteria in the rubric
- Partial fulfillment of criteria should result in INCORRECT
- The response must demonstrate clear understanding of the context
- Missing any important aspect mentioned in the rubric means INCORRECT

Evaluate whether the model's response satisfies the rubric criteria.

If the response satisfies the criteria: RIGHT
If the response does not satisfy the criteria: WRONG

Respond with only RIGHT or WRONG."""

ABSTENTION_PROMPT = """\
You are evaluating an ABSTENTION test where the model should recognize it lacks sufficient information to answer the question.

**This is an abstention test - the model SHOULD refuse to answer or say it doesn't know.**

**Critical Guidelines for Abstention Evaluation:**

1. **Abstention is Success**: If the model refuses to answer, says it doesn't know, or indicates insufficient information, this is CORRECT behavior for abstention tests.

2. **Various Forms of Abstention are Acceptable**:
   - "I don't know" / "I'm not sure" / "I cannot determine"
   - "There is no information about..." / "This wasn't mentioned..."
   - "I don't have enough information to answer..."
   - "Based on the conversation, I cannot tell..."
   - "This information wasn't provided..."
   - Asking for clarification or more information

3. **Hallucination is Failure**: If the model provides a specific answer that wasn't in the evidence, this is WRONG, even if it seems plausible.

4. **Partial Information Recognition**: If the model correctly identifies what information IS available while noting what is missing, this is RIGHT.

5. **Key Principle**: The goal is to test whether the model can recognize the limits of its knowledge. Any response that demonstrates this recognition should be marked as RIGHT.

**Answer only "RIGHT" or "WRONG". Do not provide any additional text, explanations, or reasoning.**

Question: {question}
Expected behavior: The model should indicate it cannot answer due to insufficient information
Model Response: {response}

Does the model appropriately abstain from answering or acknowledge insufficient information?

Answer (RIGHT/WRONG):"""

USER_FACTS_SINGLE_PROMPT = """\
You are evaluating whether a model's response correctly answers a question using information from an evidence message.

**Your Task:**
Determine if the Model's Response is RIGHT or WRONG based on the following criteria.

**Question Asked:**
{question}

**Evidence Message Available to the Model:**
{evidence}

**Correct Answer (what the response should convey):**
{answer}

**Model's Response to Evaluate:**
{response}

**Evaluation Criteria:**

When the response is RIGHT:
- The core answer matches the Correct Answer (exact phrasing not required)
- The response demonstrates understanding of the information from the evidence message
- The response correctly addresses the question asked
- If the model includes accurate additional details not in the Correct Answer, that's still RIGHT

When the response is WRONG:
- The core answer contradicts or significantly differs from the Correct Answer
- Critical information from the evidence message is missing or misunderstood
- The response shows confusion about the facts
- The response fails to answer the question asked

**Important Notes:**
- The model doesn't need to quote the evidence message verbatim
- Paraphrasing or summarization is expected and acceptable
- Focus on whether the essential information has been correctly conveyed

**Answer only "RIGHT" or "WRONG". Do not provide any additional text, explanations, or reasoning.**

Answer (RIGHT/WRONG):"""

USER_FACTS_MULTI_PROMPT = """\
You are evaluating whether a model's response correctly answers a question using information from multiple evidence messages.

**Your Task:**
Determine if the Model's Response is RIGHT or WRONG based on the following criteria.

**Question Asked:**
{question}

**Evidence Messages Available to the Model:**
{evidence}

**Correct Answer (what the response should convey):**
{answer}

**Model's Response to Evaluate:**
{response}

**Evaluation Criteria:**

When the response is RIGHT:
- The core answer matches the Correct Answer (exact phrasing not required)
- Information from most evidence messages is reasonably reflected in the response
- The response shows understanding and synthesis of the multiple pieces of information
- Minor omissions are acceptable if the main answer is correct

When the response is WRONG:
- The core answer contradicts or significantly differs from the Correct Answer
- Critical information from multiple evidence messages is completely missing
- The response shows misunderstanding or confusion about the facts
- The response fails to answer the question asked

**Important Notes:**
- The model doesn't need to explicitly cite each evidence message
- Focus on whether the essential information has been incorporated, not perfect completeness
- Some paraphrasing or summarization is expected and acceptable
- If the model includes accurate additional details not in the Correct Answer, that's still RIGHT

**Answer only "RIGHT" or "WRONG". Do not provide any additional text, explanations, or reasoning.**

Answer (RIGHT/WRONG):"""


def parse_right_wrong(text: str) -> bool:
    """The reference parser: RIGHT alone passes, WRONG or both fails, neither is a judge error."""
    lowered = text.strip().lower()
    right = re.search(r"\bright\b", lowered) is not None
    wrong = re.search(r"\bwrong\b", lowered) is not None
    if right or wrong:
        return right and not wrong
    raise ValueError(f"judge returned neither RIGHT nor WRONG: {text!r}")


class ConvoMemJudge(vf.Judge[bool]):
    prompt = DEFAULT_PROMPT

    def parse(self, response: vf.JudgeResponse[bool]) -> bool:
        return parse_right_wrong(response.text)


class RubricAnswerJudge(ConvoMemJudge):
    prompt = RUBRIC_PROMPT


class AbstentionJudge(ConvoMemJudge):
    prompt = ABSTENTION_PROMPT


class UserFactsSingleJudge(ConvoMemJudge):
    prompt = USER_FACTS_SINGLE_PROMPT


class UserFactsMultiJudge(ConvoMemJudge):
    prompt = USER_FACTS_MULTI_PROMPT


def judge_for(category: str, evidence_count: int, config: vf.JudgeConfig) -> ConvoMemJudge:
    if category == "user_facts":
        return (UserFactsSingleJudge if evidence_count == 1 else UserFactsMultiJudge)(config)
    if category == "abstention":
        return AbstentionJudge(config)
    if category in ("preferences", "implicit_connections"):
        return RubricAnswerJudge(config)
    return ConvoMemJudge(config)


def format_evidence(category: str, evidence_count: int, evidence_messages: list[str]) -> str:
    if category == "user_facts" and evidence_count == 1:
        return evidence_messages[0] if evidence_messages else ""
    return "\n".join(
        f"Evidence Message {i + 1}: {text}" for i, text in enumerate(evidence_messages)
    )
