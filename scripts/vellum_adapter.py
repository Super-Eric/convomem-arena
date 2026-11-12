#!/usr/bin/env python3
"""OpenAI-chat-completions shim in front of a locally hatched Vellum assistant.

Why this exists: `convomem-arena` (via `vf-run`) only knows how to talk to an
OpenAI-compatible `/v1/chat/completions` endpoint — that's how the local-Ollama
quick start in the README works. The Vellum assistant does not expose anything
shaped like that: its closest equivalent is a synchronous, single-shot RPC
(`assistant llm send`, aka `assistant inference send`) with its own
`{message, systemPrompt?, model?, ...}` request/response shape, reached over a
Unix-domain-socket IPC call, not HTTP. There is no way to point `--client.base-url`
directly at a Vellum instance.

This script is the thin adapter that makes that possible: it runs a tiny local
HTTP server that speaks OpenAI's `/v1/chat/completions` shape on the outside, and
on each request shells out to the Vellum `assistant` CLI on the inside, translating
the response back into an OpenAI-shaped JSON body. Point `vf-run` at this
adapter's URL exactly the way the README points it at Ollama.

Scope / limitations (read before using):
  - Only `convomem-arena`'s `context` memory mode is supported. The Vellum
    `inference send` RPC takes a single `message` + optional `systemPrompt` —
    it has no `tools` parameter, so it cannot do real tool calls. Don't point
    convomem-arena's `tools` mode at this adapter; it has nothing to call.
  - Every request blocks on one `assistant llm send` subprocess hitting one
    local daemon, so this does not parallelize the way a real model API does.
    Run `vf-run` with a low `--max-concurrent` (e.g. `-c 1` or `-c 2`) against
    this adapter, not the default 128.
  - Whatever model/provider the target Vellum instance is configured to use
    is what actually answers — this adapter does not change or override that
    configuration. If the instance has no working provider configured,
    `assistant llm send` will return `{"ok": false, "error": ...}` and this
    adapter will surface that as an OpenAI-style error response.

Usage:
    export ASSISTANT_IPC_SOCKET_DIR=...   # from the `vellum-env` shell helper,
    export VELLUM_WORKSPACE_DIR=...       # or however your hatch resolves its socket
    export VELLUM_ASSISTANT_DIR=/path/to/vellum-assistant-saj/assistant

    python3 scripts/vellum_adapter.py       # listens on 127.0.0.1:8800 by default

Then, in another terminal:
    uv run vf-run convomem-arena -n 6 -r 1 -m vellum -c 1 \\
      --client.base-url http://127.0.0.1:8800/v1 --client.api-key-var VELLUM_FAKE_KEY \\
      --env.agent.harness.id null --env.agent.runtime.type subprocess \\
      --env.taskset.context-sizes 10
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ADAPTER_PORT = int(os.environ.get("VELLUM_ADAPTER_PORT", "8800"))
ASSISTANT_DIR = os.environ.get("VELLUM_ASSISTANT_DIR")
BUN_BIN = os.environ.get("BUN_BIN", "bun")
VELLUM_MODEL_OVERRIDE = os.environ.get("VELLUM_MODEL")  # optional: force a --model
SEND_TIMEOUT_SECONDS = float(os.environ.get("VELLUM_ADAPTER_TIMEOUT", "120"))


def _messages_to_send_args(messages: list[dict]) -> tuple[str, str]:
    """Collapse an OpenAI `messages` array into Vellum's `(message, systemPrompt)`."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    system_prompt = "\n\n".join(system_parts)

    turns = [m for m in messages if m.get("role") != "system"]
    if not turns:
        return "", system_prompt
    if len(turns) == 1:
        return turns[0]["content"], system_prompt

    # Multiple turns (shouldn't happen in `context` mode, which is single-shot,
    # but handle it defensively): flatten into one transcript-shaped message.
    transcript = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in turns)
    return transcript, system_prompt


def _call_vellum(message: str, system_prompt: str) -> dict:
    if not ASSISTANT_DIR:
        raise RuntimeError(
            "VELLUM_ASSISTANT_DIR is not set — point it at the `assistant` "
            "package directory (the one containing src/index.ts) in your "
            "vellum-assistant checkout."
        )

    cmd = [BUN_BIN, "run", "src/index.ts", "llm", "send", "--json"]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    if VELLUM_MODEL_OVERRIDE:
        cmd += ["--model", VELLUM_MODEL_OVERRIDE]
    cmd += [message]

    try:
        proc = subprocess.run(
            cmd,
            cwd=ASSISTANT_DIR,
            env=os.environ.copy(),  # inherit ASSISTANT_IPC_SOCKET_DIR / VELLUM_WORKSPACE_DIR
            capture_output=True,
            text=True,
            timeout=SEND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"assistant llm send timed out after {SEND_TIMEOUT_SECONDS}s") from exc

    # `--json` prints exactly one JSON line to stdout; be defensive about any
    # stray output around it and parse the last line that looks like an object.
    candidate_lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    if not candidate_lines:
        raise RuntimeError(
            f"assistant llm send produced no JSON on stdout "
            f"(exit={proc.returncode}, stderr={proc.stderr.strip()[:500]!r})"
        )
    try:
        return json.loads(candidate_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse assistant llm send output as JSON: {exc}") from exc


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _openai_error(self, status: int, message: str, error_type: str = "vellum_adapter_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": error_type}})

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._openai_error(404, f"no such route: {self.path}", "not_found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw_body)
        except json.JSONDecodeError:
            self._openai_error(400, "request body is not valid JSON", "invalid_request_error")
            return

        messages = request.get("messages") or []
        if not messages:
            self._openai_error(400, "request has no `messages`", "invalid_request_error")
            return

        message, system_prompt = _messages_to_send_args(messages)

        try:
            result = _call_vellum(message, system_prompt)
        except RuntimeError as exc:
            self._openai_error(502, str(exc), "vellum_adapter_error")
            return

        if not result.get("ok", True) and "response" not in result:
            self._openai_error(
                502,
                result.get("error", "assistant llm send returned ok=false"),
                result.get("error_class", "vellum_inference_error"),
            )
            return

        usage = result.get("usage") or {}
        prompt_tokens = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)
        response = {
            "id": f"chatcmpl-vellum-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.get("model") or request.get("model") or "vellum",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.get("response", "")},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._send_json(200, response)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            model_id = VELLUM_MODEL_OVERRIDE or "vellum"
            self._send_json(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            return
        self._openai_error(404, f"no such route: {self.path}", "not_found")

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        sys.stderr.write("[vellum-adapter] " + (fmt % args) + "\n")


def main() -> None:
    if not ASSISTANT_DIR:
        print(
            "WARNING: VELLUM_ASSISTANT_DIR is not set. Every request will fail "
            "until it points at your vellum-assistant checkout's `assistant/` directory.",
            file=sys.stderr,
        )
    server = ThreadingHTTPServer(("127.0.0.1", ADAPTER_PORT), Handler)
    print(f"[vellum-adapter] listening on http://127.0.0.1:{ADAPTER_PORT}/v1 (Ctrl+C to stop)")
    print(f"[vellum-adapter] VELLUM_ASSISTANT_DIR={ASSISTANT_DIR}")
    print(f"[vellum-adapter] ASSISTANT_IPC_SOCKET_DIR={os.environ.get('ASSISTANT_IPC_SOCKET_DIR')}")
    print(f"[vellum-adapter] VELLUM_WORKSPACE_DIR={os.environ.get('VELLUM_WORKSPACE_DIR')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
