"""Grade Vellum's answers with ConvoMem's official RIGHT/WRONG judges.

The judge client now lives in `convomem_arena.judge_client`, shared with the judge
robustness checker; this module re-exports it for the Vellum harness.
"""

from convomem_arena.judge_client import (
    JudgeEndpoint,
    Verdict,
    judge_answer,
    judge_prompt,
)

__all__ = ["JudgeEndpoint", "Verdict", "judge_answer", "judge_prompt"]
