"""ConvoMem data: download evidence/filler files, mix test cases, format conversations.

Mixing follows the official ConvoMem `StandardTestCasesGenerator`: every evidence
conversation of an item is kept, `context_size - evidence_count` filler conversations of
the same persona are sampled, and the evidence conversations are placed at random
(sorted) positions among them so their relative order is preserved. A per-task seed
derived from the task id makes every mix reproducible.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict

REPO_ID = "Salesforce/ConvoMem"
REVISION = "e3e9b39115b02346824c70d349350de738f8be41"
EVIDENCE_ROOT = "core_benchmark/evidence_questions"
FILLER_ROOT = "core_benchmark/filler_conversations"

Category = Literal[
    "user_facts",
    "assistant_facts",
    "changing_facts",
    "abstention",
    "preferences",
    "implicit_connections",
]

# category -> (dataset folder, available evidence counts)
CATEGORIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "user_facts": ("user_evidence", (1, 2, 3, 4, 5, 6)),
    "assistant_facts": ("assistant_facts_evidence", (1, 2, 3, 4, 5, 6)),
    "changing_facts": ("changing_evidence", (2, 3, 4, 5, 6)),
    "abstention": ("abstention_evidence", (1, 2, 3)),
    "preferences": ("preference_evidence", (1, 2)),
    "implicit_connections": ("implicit_connection_evidence", (1, 2, 3)),
}


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    speaker: str
    text: str


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    messages: list[Message]


class EvidenceItem(BaseModel):
    """One question with the conversations that hold (or, for abstention, surround) its answer."""

    model_config = ConfigDict(frozen=True)

    category: str
    evidence_count: int
    persona_file: str
    item_index: int
    question: str
    answer: str
    evidence_messages: list[str]
    conversations: list[Conversation]


def normalize_speaker(speaker: str) -> str:
    """The dataset mixes `User`/`user` and `Assistant`/`assistant`."""
    lowered = speaker.strip().lower()
    if lowered == "user":
        return "User"
    if lowered == "assistant":
        return "Assistant"
    return speaker.strip()


def parse_conversation(raw: dict) -> Conversation:
    return Conversation(
        id=str(raw.get("id") or ""),
        messages=[
            Message(speaker=normalize_speaker(m["speaker"]), text=m["text"])
            for m in raw["messages"]
        ],
    )


def parse_evidence_file(
    payload: dict, category: str, evidence_count: int, persona_file: str
) -> list[EvidenceItem]:
    items = []
    for index, raw in enumerate(payload["evidence_items"]):
        if not raw.get("conversations"):
            continue
        items.append(
            EvidenceItem(
                category=category,
                evidence_count=evidence_count,
                persona_file=persona_file,
                item_index=index,
                question=raw["question"],
                answer=raw["answer"],
                evidence_messages=[m["text"] for m in raw.get("message_evidences") or []],
                conversations=[parse_conversation(c) for c in raw["conversations"]],
            )
        )
    return items


def parse_filler_file(payload: dict) -> list[Conversation]:
    return [
        parse_conversation(conversation)
        for item in payload["evidence_items"]
        for conversation in item["conversations"]
    ]


def mix(
    evidence: list[Conversation],
    filler: list[Conversation],
    context_size: int,
    rng: random.Random,
) -> tuple[list[Conversation], list[int]]:
    """Mix evidence into `context_size` conversations; return them with the evidence positions.

    All evidence is always kept, so a context smaller than the evidence count yields just
    the evidence (as in the official generator).
    """
    evidence_ids = {c.id for c in evidence}
    pool = [c for c in filler if c.id not in evidence_ids]
    n_filler = min(max(context_size - len(evidence), 0), len(pool))
    chosen = rng.sample(pool, n_filler)
    total = n_filler + len(evidence)
    positions = sorted(rng.sample(range(total), len(evidence)))
    mixed: list[Conversation] = []
    evidence_iter, filler_iter = iter(evidence), iter(chosen)
    position_set = set(positions)
    for slot in range(total):
        mixed.append(next(evidence_iter) if slot in position_set else next(filler_iter))
    return mixed, positions


def format_conversation(conversation: Conversation, number: int) -> str:
    lines = "\n".join(f"{m.speaker}: {m.text}" for m in conversation.messages)
    return f"Conversation {number}:\n{lines}"


def format_context(conversations: list[Conversation]) -> str:
    """The official `buildConversationContext` layout."""
    return "\n\n".join(format_conversation(c, i + 1) for i, c in enumerate(conversations))


class HubSource:
    """Lazily lists and downloads ConvoMem files from the Hugging Face Hub (cached locally)."""

    def __init__(self, repo_id: str = REPO_ID, revision: str = REVISION) -> None:
        self.repo_id = repo_id
        self.revision = revision

    def list_files(self, folder: str) -> list[str]:
        from huggingface_hub import HfApi

        entries = HfApi().list_repo_tree(
            self.repo_id,
            path_in_repo=folder,
            repo_type="dataset",
            revision=self.revision,
        )
        return sorted(
            e.path.rsplit("/", 1)[-1]
            for e in entries
            if e.path.endswith(".json") and hasattr(e, "size")
        )

    def read_json(self, path: str) -> dict:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision)
        with open(local, encoding="utf-8") as f:
            return json.load(f)


class EvidenceStream:
    """Streams evidence items per category and serves same-persona filler conversations."""

    def __init__(self, source: HubSource, seed: int) -> None:
        self.source = source
        self.seed = seed
        self.filler = lru_cache(maxsize=16)(self._filler)

    def _filler(self, persona_file: str) -> list[Conversation]:
        return parse_filler_file(self.source.read_json(f"{FILLER_ROOT}/{persona_file}"))

    def folder_items(self, category: str, evidence_count: int) -> Iterator[EvidenceItem]:
        """Items of one `<category>/<n>_evidence` folder in a seeded file and item order."""
        folder_name, _ = CATEGORIES[category]
        folder = f"{EVIDENCE_ROOT}/{folder_name}/{evidence_count}_evidence"
        rng = random.Random(f"{self.seed}:{folder}")
        files = self.source.list_files(folder)
        rng.shuffle(files)
        for persona_file in files:
            payload = self.source.read_json(f"{folder}/{persona_file}")
            items = parse_evidence_file(payload, category, evidence_count, persona_file)
            rng.shuffle(items)
            yield from items

    def category_items(
        self, category: str, evidence_counts: list[int] | None
    ) -> Iterator[EvidenceItem]:
        """Round-robin over a category's evidence-count folders."""
        _, available = CATEGORIES[category]
        counts = [n for n in available if evidence_counts is None or n in evidence_counts]
        yield from round_robin([self.folder_items(category, n) for n in counts])


def round_robin(streams: list[Iterator]) -> Iterator:
    active = list(streams)
    while active:
        for stream in list(active):
            try:
                yield next(stream)
            except StopIteration:
                active.remove(stream)
