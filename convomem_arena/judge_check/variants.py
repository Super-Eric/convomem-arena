"""What the robustness checker changes, and what a robust judge should do about it.

A judge sees up to four inputs: the question, the evidence, the reference answer and
the response being graded. Each variant changes exactly one of them, either keeping
its meaning (a robust judge's verdict must not move) or changing it (a robust
judge's verdict must move, from RIGHT to WRONG).

Not every judge sees every input. The five official ConvoMem prompts render:

| judge kind          | question | evidence | reference | response |
|---------------------|----------|----------|-----------|----------|
| user_facts_single   | yes      | yes      | yes       | yes      |
| user_facts_multi    | yes      | yes      | yes       | yes      |
| default             | yes      | -        | yes       | yes      |
| rubric              | yes      | -        | yes*      | yes      |
| abstention          | yes      | -        | -         | yes      |

(*the reference is a grading rubric, not an answer.) A variant that changes an input
its judge never sees cannot change the verdict, so it is skipped, not counted.
"""

from __future__ import annotations

from typing import Literal

Expect = Literal["same", "flip"]

JUDGE_KINDS = ("user_facts_single", "user_facts_multi", "default", "rubric", "abstention")


def judge_kind(category: str, evidence_count: int) -> str:
    """Which official prompt grades this category (mirrors `judges.judge_for`)."""
    if category == "user_facts":
        return "user_facts_single" if evidence_count == 1 else "user_facts_multi"
    if category == "abstention":
        return "abstention"
    if category in ("preferences", "implicit_connections"):
        return "rubric"
    return "default"


SEES = {
    "user_facts_single": {"question", "evidence", "reference", "response"},
    "user_facts_multi": {"question", "evidence", "reference", "response"},
    "default": {"question", "reference", "response"},
    "rubric": {"question", "reference", "response"},
    "abstention": {"question", "response"},
}

# variant -> (the input it changes, a one-line description)
VARIANTS: dict[str, tuple[str, str]] = {
    "question_paraphrase": ("question", "same question, reworded"),
    "question_negated": ("question", "question reversed or negated"),
    "evidence_paraphrase": ("evidence", "same evidence, reworded"),
    "evidence_changed": ("evidence", "evidence changed so it contradicts the answer"),
    "reference_changed": ("reference", "reference answer replaced with a different one"),
    "response_wrong": ("response", "response replaced with a plausible wrong one"),
}


def expectation(variant: str, kind: str) -> Expect | None:
    """What a robust judge does under `variant`; None when the judge can't see it.

    Rewording keeps meaning, so the verdict must stay. Changing the reference, the
    response or the evidence's facts breaks the match with a correct response, so a
    RIGHT verdict must flip to WRONG. Negating the question means the response no
    longer answers it, so it must flip too, except for abstention: a reply that
    declines still declines, so that judge must stay at RIGHT.
    """
    changed_input, _ = VARIANTS[variant]
    if changed_input not in SEES[kind]:
        return None
    if variant.endswith("_paraphrase"):
        return "same"
    if variant == "question_negated" and kind == "abstention":
        return "same"
    return "flip"
