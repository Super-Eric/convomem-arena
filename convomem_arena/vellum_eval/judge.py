"""Grade Vellum's answers with ConvoMem's official RIGHT/WRONG judges.

The prompt text and the RIGHT/WRONG parser are convomem-arena's own
(`convomem_arena.judges`): each judge class renders its official prompt through
`build_messages`, so scores stay comparable with every other ConvoMem run. Only the
transport differs: the harness calls any OpenAI-compatible `/chat/completions`
endpoint directly (default: local Ollama), outside a verifiers rollout.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import verifiers.v1 as vf

from convomem_arena.judges import format_evidence, judge_for, parse_right_wrong


@dataclass
class JudgeEndpoint:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:30b-a3b"
    api_key_var: str | None = None
    timeout_s: float = 300.0


@dataclass
class Verdict:
    right: bool | None
    raw: str
    error: str | None = None


def judge_prompt(
    category: str,
    evidence_count: int,
    question: str,
    answer: str,
    response: str,
    evidence_messages: list[str],
) -> str:
    """The official ConvoMem judge prompt for this category, filled in."""
    judge = judge_for(category, evidence_count, vf.JudgeConfig())
    rendered = judge.build_messages(
        question=question,
        answer=answer,
        response=response,
        evidence=format_evidence(category, evidence_count, evidence_messages),
    )
    if not isinstance(rendered, str):
        raise TypeError("expected a string prompt from the ConvoMem judge")
    return rendered


def _chat(endpoint: JudgeEndpoint, prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_var:
        headers["Authorization"] = f"Bearer {os.environ[endpoint.api_key_var]}"
    body = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if "11434" in endpoint.base_url:
        body["think"] = False  # Ollama: answer directly, no reasoning trace
    req = urllib.request.Request(
        endpoint.base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        headers,
    )
    with urllib.request.urlopen(req, timeout=endpoint.timeout_s) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def judge_answer(endpoint: JudgeEndpoint, **fields) -> Verdict:
    prompt = judge_prompt(**fields)
    try:
        raw = _chat(endpoint, prompt)
    except Exception as err:  # noqa: BLE001 - recorded, not fatal
        return Verdict(right=None, raw="", error=f"{type(err).__name__}: {err}")
    try:
        return Verdict(right=parse_right_wrong(raw), raw=raw)
    except ValueError as err:
        return Verdict(right=None, raw=raw, error=str(err))
