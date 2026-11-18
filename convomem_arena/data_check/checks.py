"""Deterministic quality checks on raw ConvoMem items. No model calls.

Two severities:

- **error**: the item is structurally inconsistent (an evidence message that is not in
  its conversations, counts that disagree with the folder, a missing field). These are
  facts about the file, not judgements.
- **review**: a heuristic says something may be wrong (the answer's key words are not
  in the evidence, or reappear in unrelated filler). Each needs a human look; a
  flag is a pointer, not a verdict.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from convomem_arena.data import normalize_speaker
from convomem_arena.text import (
    content_words,
    coverage,
    is_number,
    key_terms,
    norm,
    present,
    specifics,
)

FACT_CATEGORIES = ("user_facts", "assistant_facts", "changing_facts")
EXPECTED_SPEAKER = {"user_facts": "User", "assistant_facts": "Assistant"}

CHECKS: dict[str, tuple[str, str]] = {
    # name: (severity, what it means)
    "empty_field": ("error", "question or answer is empty"),
    "evidence_count_mismatch": (
        "error",
        "number of evidence messages differs from the folder's count",
    ),
    "flagged_conversation_count": (
        "error",
        "conversations marked containsEvidence differ from the count",
    ),
    "evidence_not_in_conversations": (
        "error",
        "an evidence message has no close match in any of its conversations",
    ),
    "evidence_reworded_in_conversation": (
        "review",
        "an evidence message appears in its conversation only in reworded form",
    ),
    "evidence_speaker_mismatch": (
        "error",
        "an evidence message's speaker differs from the conversation's",
    ),
    "wrong_evidence_speaker": (
        "review",
        "user facts said by the assistant, or assistant facts by the user",
    ),
    "answer_restates_question": ("review", "the answer adds no content words beyond the question"),
    "answer_name_not_in_evidence": (
        "review",
        "a name in the answer appears in neither the evidence nor the question",
    ),
    "answer_number_not_in_evidence": (
        "review",
        "a number in the answer is not in the evidence (often a computed total or average)",
    ),
    "latest_evidence_not_answer": (
        "review",
        "changing facts: an earlier evidence message matches the answer better than the last",
    ),
    "answer_terms_in_filler": (
        "review",
        "one filler message of the same persona holds the answer's details, including a name",
    ),
    "question_topic_in_filler": (
        "review",
        "abstention: one filler message contains most of the question's key words",
    ),
    "duplicate_question": ("review", "the same question appears in more than one sampled item"),
}


class Msg(BaseModel):
    speaker: str
    text: str


class Conv(BaseModel):
    id: str = ""
    contains_evidence: bool | None = None
    messages: list[Msg]


class RawItem(BaseModel):
    id: str
    category: str
    evidence_count: int
    persona_file: str
    question: str
    answer: str
    evidence: list[Msg]
    conversations: list[Conv]


class Flag(BaseModel):
    check: str
    severity: str
    detail: str = ""


class ItemResult(BaseModel):
    item: RawItem
    flags: list[Flag] = Field(default_factory=list)

    def add(self, check: str, detail: str = "") -> None:
        self.flags.append(Flag(check=check, severity=CHECKS[check][0], detail=detail))


def parse_raw(
    raw: dict, category: str, evidence_count: int, persona_file: str, index: int
) -> RawItem:
    persona = persona_file.removesuffix(".json")
    return RawItem(
        id=f"{category}/{evidence_count}/{persona}/{index}",
        category=category,
        evidence_count=evidence_count,
        persona_file=persona_file,
        question=raw.get("question") or "",
        answer=raw.get("answer") or "",
        evidence=[
            Msg(speaker=normalize_speaker(m.get("speaker", "")), text=m.get("text", ""))
            for m in raw.get("message_evidences") or []
        ],
        conversations=[
            Conv(
                id=str(c.get("id") or ""),
                contains_evidence=c.get("containsEvidence"),
                messages=[
                    Msg(speaker=normalize_speaker(m.get("speaker", "")), text=m.get("text", ""))
                    for m in c.get("messages") or []
                ],
            )
            for c in raw.get("conversations") or []
        ],
    )


# ----------------------------------------------------------------- per item

REWORDED_RATIO = 0.85
"""Similarity above which a missing evidence message counts as reworded, not absent.
In the samples checked, reworded ones differ only in a lead-in ("Okay, next up is…"
vs "I was thinking about…") and score 0.90-0.98; truly absent ones score far lower."""


def best_match(target: str, messages: list[Msg]) -> tuple[float, Msg | None]:
    best, closest = 0.0, None
    for m in messages:
        ratio = SequenceMatcher(None, target, norm(m.text), autojunk=False).ratio()
        if ratio > best:
            best, closest = ratio, m
    return best, closest


def check_structure(r: ItemResult) -> None:
    it = r.item
    if not it.question.strip() or not it.answer.strip():
        r.add("empty_field")
    if len(it.evidence) != it.evidence_count:
        r.add(
            "evidence_count_mismatch",
            f"{len(it.evidence)} messages in a {it.evidence_count}-evidence folder",
        )
    marked = [c for c in it.conversations if c.contains_evidence]
    if (
        any(c.contains_evidence is not None for c in it.conversations)
        and len(marked) != it.evidence_count
    ):
        r.add("flagged_conversation_count", f"{len(marked)} marked of {len(it.conversations)}")
    messages = [m for c in it.conversations for m in c.messages]
    for i, ev in enumerate(it.evidence):
        target = norm(ev.text)
        hits = [m for m in messages if target and target in norm(m.text)]
        if not hits:
            ratio, closest = best_match(target, messages)
            if ratio >= REWORDED_RATIO:
                r.add(
                    "evidence_reworded_in_conversation", f"evidence {i + 1}: similarity {ratio:.2f}"
                )
                hits = [closest]
            else:
                r.add(
                    "evidence_not_in_conversations",
                    f"evidence {i + 1}: best similarity {ratio:.2f}",
                )
        if hits and all(m.speaker != ev.speaker for m in hits):
            r.add(
                "evidence_speaker_mismatch", f"evidence {i + 1}: {ev.speaker} vs {hits[0].speaker}"
            )
        expected = EXPECTED_SPEAKER.get(it.category)
        if expected and ev.speaker != expected:
            r.add("wrong_evidence_speaker", f"evidence {i + 1} spoken by {ev.speaker}")


def check_content(r: ItemResult) -> None:
    it = r.item
    if it.category not in FACT_CATEGORIES:
        return
    if not key_terms(it.answer, it.question):
        r.add("answer_restates_question")
        return
    details = specifics(it.answer, it.question)
    if details:
        found = set(present(details, " ".join(e.text for e in it.evidence)))
        names = [d for d in details if d not in found and not is_number(d)]
        figures = [d for d in details if d not in found and is_number(d)]
        if names:
            r.add("answer_name_not_in_evidence", f"missing {names[:6]} of {details[:8]}")
        if figures:
            r.add("answer_number_not_in_evidence", f"missing {figures[:6]} of {details[:8]}")
    if it.category == "changing_facts" and len(it.evidence) > 1:
        terms = details or key_terms(it.answer, it.question)
        per = [
            coverage(terms, e.text) if not details else len(present(details, e.text)) / len(details)
            for e in it.evidence
        ]
        if max(per[:-1]) > per[-1]:
            r.add(
                "latest_evidence_not_answer",
                "match by message: " + ", ".join(f"{p:.0%}" for p in per),
            )


def check_filler(r: ItemResult, filler: list[Conv], threshold: float = 0.8) -> None:
    """Does one message of the same persona's unrelated chat hold the answer's names and
    numbers (or, for abstention, the question's key words)? Terms must co-occur in a
    single message: scattered across a long conversation they are usually coincidence."""
    it = r.item
    if it.category in FACT_CATEGORIES:
        terms, check, minimum = specifics(it.answer, it.question), "answer_terms_in_filler", 2
    elif it.category == "abstention":
        terms, check, minimum = content_words(it.question), "question_topic_in_filler", 3
    else:
        return
    if len(terms) < minimum:
        return
    best, best_where, best_text = 0.0, "", ""
    for conv in filler:
        for m in conv.messages:
            hits = present(terms, m.text)
            if check == "answer_terms_in_filler" and all(is_number(h) for h in hits):
                continue  # numbers alone recur by coincidence
            cov = len(hits) / len(terms)
            if cov > best:
                best, best_where, best_text = cov, conv.id, m.text
    if best >= threshold:
        r.add(
            check, f"{best:.0%} of {terms[:6]} in filler {best_where or '?'}: {best_text[:160]!r}"
        )


def check_duplicates(results: Iterable[ItemResult]) -> None:
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for r in results:
        groups[norm(r.item.question)].append(r)
    for rs in groups.values():
        if len(rs) > 1:
            ids = [x.item.id for x in rs]
            for x in rs:
                x.add("duplicate_question", "also " + ", ".join(i for i in ids if i != x.item.id))
