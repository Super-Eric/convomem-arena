---
title: "Use a Vellum assistant as the model"
description: "Run convomem-arena against a locally hatched Vellum assistant through a small OpenAI-compatible adapter."
---

# Use a Vellum assistant as the model

The [local Ollama quick start](../getting-started/quickstart.md#run-fully-local-with-ollama) works because Ollama serves a real OpenAI-compatible
`/v1/chat/completions` endpoint. A locally hatched [Vellum](https://github.com/vellum-ai/vellum-assistant) assistant
does **not** expose anything like that.

!!! note "This tests Vellum's model, not its memory"
    The history is still pasted into the prompt here. To evaluate how Vellum *remembers*, see [Evaluate a personal assistant](../getting-started/personal-assistant.md).

Its closest equivalent is a synchronous,
single-shot CLI/IPC call (`assistant llm send`, aka `assistant inference send`) with its
own `{message, systemPrompt?, model?, ...}` shape, not OpenAI's `{model, messages}`.
`vf-run --client.base-url` can't point at it directly.

`scripts/vellum_adapter.py` bridges the gap: a small, dependency-free (stdlib-only)
local HTTP server that speaks OpenAI's `/v1/chat/completions` on the outside and shells
out to `assistant llm send --json` on the inside, translating the response back into an
OpenAI-shaped JSON body.

**Scope — read before using:**
- Only convomem-arena's `context` memory mode is supported. `assistant llm send` has no
  `tools` parameter, so it cannot do real tool calls — don't point `tools` mode at it.
- Each request blocks on one subprocess hitting one local daemon; there's no real
  parallelism behind this adapter. Run `vf-run` with a low `--max-concurrent` against it
  (e.g. `-c 1` or `-c 2`), not the default `128`.
- `assistant llm send` does not necessarily use your instance's everyday chat
  `activeProfile` — it can fall back to a different, advisor-oriented profile that needs
  a connection you may not have configured (seen locally: it defaulted to an Anthropic
  profile and failed with `provider_connection "..." not found`, even though the
  instance's real chat profile was a working, keyless Ollama one). If you hit that, check
  `assistant config get llm` for a profile that IS configured and set `VELLUM_PROFILE` to
  its name — see below.
- **Verified working end to end** — 6/6, mean reward 1.000 — using a local Ollama-backed
  Vellum profile (`qwen3:30b-a3b`) as both the agent (through this adapter) and the judge.

```bash
# 1. Have a hatched Vellum instance with a working model profile. A keyless local one is
#    easiest to get working for this, e.g.:
#      assistant config set services.inference.provider ollama   # only if you don't
#      assistant config set services.inference.model qwen3:30b-a3b   # already have one
#    then confirm which profile name is actually configured and keyless:
#      assistant config get llm   # look for a profile with "provider": "ollama"

# 2. Point the adapter at that instance and your vellum-assistant checkout:
export ASSISTANT_IPC_SOCKET_DIR=...     # same vars your shell already uses to talk
export VELLUM_WORKSPACE_DIR=...         # to the hatched instance (e.g. via a `vellum-env` helper)
export VELLUM_ASSISTANT_DIR=/path/to/vellum-assistant/assistant   # dir containing src/index.ts
export VELLUM_PROFILE=qwen-local        # the keyless profile name from step 1

# 3. Start the adapter (separate terminal, left running):
python3 scripts/vellum_adapter.py       # listens on http://127.0.0.1:8800/v1

# 4. Run convomem-arena against it. The adapter only stands in for the *agent* model —
#    you still need a judge endpoint (PRIME_API_KEY, or a local judge as in the Ollama
#    example in the quick start).
export VELLUM_FAKE_KEY=unused   # value is ignored; vf-run just needs the var set
export PRIME_API_KEY=...

uv run vf-run convomem-arena -n 6 -r 1 -m vellum -c 1 \
  --client.base-url http://127.0.0.1:8800/v1 --client.api-key-var VELLUM_FAKE_KEY \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.context-sizes 10
```
