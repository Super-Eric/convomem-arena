"""Call ConvoMem's official judges outside a verifiers rollout.

The prompt text and the RIGHT/WRONG parser are the environment's own
(`convomem_arena.judges`): each judge class renders its official prompt through
`build_messages`, and verdicts go through `parse_right_wrong`, so anything graded
here is graded exactly as `convomem-arena` grades a rollout. Only the transport
differs: a plain OpenAI-compatible `/chat/completions` call (default: local Ollama).
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import verifiers.v1 as vf

from convomem_arena.judges import format_evidence, judge_for, parse_right_wrong


@dataclass
class JudgeEndpoint:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:30b-a3b"
    api_key_var: str | None = None
    timeout_s: float = 300.0
    temperature: float = 0.0
    think: bool = False
    """Ollama only: let a reasoning model think before answering."""


@dataclass
class Verdict:
    right: bool | None
    raw: str
    error: str | None = None


ChatFn = Callable[[JudgeEndpoint, str], str]


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


def chat(endpoint: JudgeEndpoint, prompt: str) -> str:
    """One user message in, the reply text out."""
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key_var:
        headers["Authorization"] = f"Bearer {os.environ[endpoint.api_key_var]}"
    body = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": endpoint.temperature,
    }
    if "11434" in endpoint.base_url:
        body["think"] = endpoint.think
    req = urllib.request.Request(
        endpoint.base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        headers,
    )
    with urllib.request.urlopen(req, timeout=endpoint.timeout_s) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"] or ""


def judge_rendered(endpoint: JudgeEndpoint, prompt: str, chat_fn: ChatFn = chat) -> Verdict:
    """Send an already-rendered judge prompt and parse the verdict."""
    try:
        raw = chat_fn(endpoint, prompt)
    except Exception as err:  # noqa: BLE001 - recorded, not fatal
        return Verdict(right=None, raw="", error=f"{type(err).__name__}: {err}")
    try:
        return Verdict(right=parse_right_wrong(raw), raw=raw)
    except ValueError as err:
        return Verdict(right=None, raw=raw, error=str(err))


def judge_answer(endpoint: JudgeEndpoint, chat_fn: ChatFn = chat, **fields) -> Verdict:
    return judge_rendered(endpoint, judge_prompt(**fields), chat_fn)
