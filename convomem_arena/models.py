"""Task data and rollout state, shared by the taskset and the memory tool server."""

from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from convomem_arena.data import Conversation


class ConvoMemData(vf.TaskData):
    question: str
    answer: str
    """Gold answer, or a grading rubric for preferences / implicit connections."""
    category: str
    evidence_count: int
    context_size: int
    """Target number of conversations in the history (evidence is always kept)."""
    evidence_messages: list[str] = Field(default_factory=list)
    evidence_positions: list[int] = Field(default_factory=list)
    """0-based indices of the evidence conversations inside `conversations`."""
    num_conversations: int = 0
    conversations: list[Conversation] = Field(default_factory=list)
    """The mixed history; left empty in `context` mode, where it is already in the prompt."""


class MemoryState(vf.State):
    tool_calls: int = 0
    retrieved: list[int] = Field(default_factory=list)
    """0-based indices of conversations whose messages a memory tool returned."""
