"""Per-rollout memory tool server over a user's past conversations.

The history comes from the rollout's task data in `setup_task` and never enters the
prompt, so the agent can only reach it through these tools. Every call records which
conversations it surfaced on the shared rollout state, which task scoring reads back as
retrieval metrics.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import verifiers.v1 as vf

from convomem_arena.data import Conversation
from convomem_arena.models import ConvoMemData, MemoryState

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)
PREVIEW_CHARS = 200


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms:
            return []
        scores = []
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_length or 1))
            score = sum(
                self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in terms if t in doc
            )
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda s: (-s[1], s[0]))
        return scores[:k]


class MemoryIndex:
    """Message-level search plus paged reads over a list of conversations."""

    def __init__(self, conversations: list[Conversation]) -> None:
        self.conversations = conversations
        self.locations = [
            (c, m) for c, conv in enumerate(conversations) for m in range(len(conv.messages))
        ]
        self.bm25 = BM25([conversations[c].messages[m].text for c, m in self.locations])

    def search(self, query: str, limit: int) -> list[dict]:
        hits = []
        for doc, _ in self.bm25.top(query, max(1, min(limit, 20))):
            c, m = self.locations[doc]
            message = self.conversations[c].messages[m]
            hits.append(
                {
                    "conversation": c + 1,
                    "message": m + 1,
                    "speaker": message.speaker,
                    "text": message.text,
                }
            )
        return hits

    def read(self, conversation: int, start: int, count: int) -> dict:
        if not 1 <= conversation <= len(self.conversations):
            raise ValueError(
                f"conversation must be between 1 and {len(self.conversations)}, got {conversation}"
            )
        messages = self.conversations[conversation - 1].messages
        first = max(start, 1)
        window = messages[first - 1 : first - 1 + max(1, min(count, 100))]
        return {
            "conversation": conversation,
            "total_messages": len(messages),
            "messages": [
                {"message": first + i, "speaker": m.speaker, "text": m.text}
                for i, m in enumerate(window)
            ],
        }

    def list(self, offset: int, limit: int) -> dict:
        first = max(offset, 0)
        rows = []
        for c in range(first, min(first + max(1, min(limit, 100)), len(self.conversations))):
            messages = self.conversations[c].messages
            opening = next((m.text for m in messages if m.speaker == "User"), "")
            rows.append(
                {
                    "conversation": c + 1,
                    "num_messages": len(messages),
                    "opening": opening[:PREVIEW_CHARS],
                }
            )
        return {"total_conversations": len(self.conversations), "conversations": rows}


class MemoryToolset(vf.Toolset[vf.ToolsetConfig, MemoryState]):
    TOOL_PREFIX = "memory"

    async def setup_task(self, task: ConvoMemData) -> None:
        self.index = MemoryIndex(task.conversations)

    def _record(self, conversations: list[int]) -> None:
        self.state.tool_calls += 1
        seen = set(self.state.retrieved)
        self.state.retrieved = self.state.retrieved + [
            c for c in dict.fromkeys(conversations) if c not in seen
        ]

    @vf.tool
    async def search(self, query: str, limit: int = 8) -> list[dict]:
        """Keyword (BM25) search over every message of the stored past conversations.

        Args:
            query: Words to search for.
            limit: Maximum number of messages to return (1-20).

        Returns the best-matching messages, each with its conversation number, message
        number, speaker and full text.
        """
        hits = self.index.search(query, limit)
        self._record([h["conversation"] - 1 for h in hits])
        return hits

    @vf.tool
    async def read_conversation(self, conversation: int, start: int = 1, count: int = 40) -> dict:
        """Read consecutive messages of one stored conversation.

        Args:
            conversation: Conversation number (1-based).
            start: First message number to return (1-based).
            count: Number of messages to return (1-100).
        """
        result = self.index.read(conversation, start, count)
        self._record([conversation - 1])
        return result

    @vf.tool
    async def list_conversations(self, offset: int = 0, limit: int = 50) -> dict:
        """List stored conversations in chronological order with their message counts and
        the opening of the first user message.

        Args:
            offset: Number of conversations to skip.
            limit: Maximum number of conversations to list (1-100).
        """
        self.state.tool_calls += 1
        return self.index.list(offset, limit)


if __name__ == "__main__":
    MemoryToolset.run()
