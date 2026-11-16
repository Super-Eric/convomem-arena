"""Stage 0: choose questions for one persona and build the history Vellum will live through.

ConvoMem stores each question with its own evidence conversations, and filler
conversations per persona. A real assistant has *one* history per user, so for one
persona we take N questions per category and merge everything into a single,
time-ordered history:

- all evidence conversations of the chosen questions (the order *within* a question
  is kept, which matters for `changing_facts`: the later conversation wins), and
- filler conversations from the same persona up to `history_size`,

placed at seeded random positions. Each conversation gets a timestamp one day after
the previous one, ending the day before "now", so recency behaves naturally.

One Vellum memory built from this history then answers every chosen question, which
spreads the slow part (capture + consolidation) over many questions.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

from pydantic import BaseModel

from convomem_arena.data import (
    CATEGORIES,
    EVIDENCE_ROOT,
    Conversation,
    EvidenceStream,
    HubSource,
)

DAY_MS = 24 * 60 * 60 * 1000


class PlannedQuestion(BaseModel):
    id: str
    category: str
    evidence_count: int
    question: str
    answer: str
    evidence_messages: list[str]
    evidence_conversation_ids: list[str]


class PlannedConversation(BaseModel):
    id: str
    created_at_ms: int
    is_evidence: bool
    conversation: Conversation


class EvalPlan(BaseModel):
    persona_file: str
    persona_role: str
    seed: int
    questions: list[PlannedQuestion]
    history: list[PlannedConversation]


def persona_role(persona_file: str) -> str:
    return re.sub(r"^[0-9a-f-]{36}_", "", persona_file).removesuffix(".json").replace("_", " ")


def _items_for(
    source: HubSource, persona_file: str, category: str, evidence_counts: Iterable[int]
):
    folder, _ = CATEGORIES[category]
    for n in evidence_counts:
        path = f"{EVIDENCE_ROOT}/{folder}/{n}_evidence"
        if persona_file not in source.list_files(path):
            continue
        payload = source.read_json(f"{path}/{persona_file}")
        for index, raw in enumerate(payload["evidence_items"]):
            if raw.get("conversations"):
                yield n, index, raw


def build_plan(
    persona_file: str,
    per_category: int = 2,
    history_size: int = 24,
    seed: int = 0,
    categories: list[str] | None = None,
    max_evidence: int = 2,
    now_ms: int | None = None,
    source: HubSource | None = None,
) -> EvalPlan:
    """Questions for `persona_file` (up to `per_category` each, evidence count ≤
    `max_evidence`) and one merged history of about `history_size` conversations."""
    import time

    from convomem_arena.data import parse_conversation

    source = source or HubSource()
    rng = random.Random(f"{seed}:{persona_file}")
    questions: list[PlannedQuestion] = []
    evidence: list[list[Conversation]] = []
    seen: set[str] = set()
    for category in categories or list(CATEGORIES):
        _, available = CATEGORIES[category]
        counts = [n for n in available if n <= max_evidence]
        candidates = list(_items_for(source, persona_file, category, counts))
        rng.shuffle(candidates)
        for n, index, raw in candidates[:per_category]:
            convs = [parse_conversation(c) for c in raw["conversations"]]
            # Conversation ids repeat across questions; keep one copy.
            fresh = [c for c in convs if c.id not in seen]
            seen.update(c.id for c in fresh)
            evidence.append(fresh)
            questions.append(
                PlannedQuestion(
                    id=f"{category}/{n}/{index}",
                    category=category,
                    evidence_count=n,
                    question=raw["question"],
                    answer=str(raw["answer"]),
                    evidence_messages=[m["text"] for m in raw.get("message_evidences") or []],
                    evidence_conversation_ids=[c.id for c in convs],
                )
            )

    stream = EvidenceStream(source, seed)
    filler = [c for c in stream.filler(persona_file) if c.id not in seen]
    rng.shuffle(filler)
    n_evidence = sum(len(group) for group in evidence)
    chosen_filler = filler[: max(history_size - n_evidence, 0)]

    # Interleave: evidence groups keep their internal order; everything else is shuffled.
    slots: list[tuple[str, Conversation]] = [("filler", c) for c in chosen_filler]
    for group in evidence:
        positions = sorted(rng.sample(range(len(slots) + len(group)), len(group)))
        for pos, conv in zip(positions, group):
            slots.insert(pos, ("evidence", conv))

    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    start = now_ms - (len(slots) + 1) * DAY_MS
    history = [
        PlannedConversation(
            id=conv.id,
            created_at_ms=start + i * DAY_MS,
            is_evidence=kind == "evidence",
            conversation=conv,
        )
        for i, (kind, conv) in enumerate(slots)
    ]
    return EvalPlan(
        persona_file=persona_file,
        persona_role=persona_role(persona_file),
        seed=seed,
        questions=questions,
        history=history,
    )


def import_payload(plan: EvalPlan, minutes_between_messages: int = 2) -> dict:
    """`assistant conversations import` JSON for the plan's history."""
    conversations = []
    for planned in plan.history:
        messages = [
            {
                "role": "user" if m.speaker == "User" else "assistant",
                "content": m.text,
                "createdAt": planned.created_at_ms + i * minutes_between_messages * 60_000,
            }
            for i, m in enumerate(planned.conversation.messages)
        ]
        conversations.append(
            {
                "sourceKey": f"convomem:{planned.id}",
                "title": f"convomem {planned.id}",
                "createdAt": planned.created_at_ms,
                "updatedAt": messages[-1]["createdAt"] if messages else planned.created_at_ms,
                "messages": messages,
            }
        )
    return {"conversations": conversations}
