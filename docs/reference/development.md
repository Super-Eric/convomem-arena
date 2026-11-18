---
title: "Development and testing"
description: "Running the tests and linters, validating setup against real Hub data, and testing without an API key."
---

# Development and testing

```bash
uv pip install -e ".[dev]" ruff

uv run pytest tests                                  # offline tests, no network or model
uv run ruff check convomem_arena tests
uv run ruff format --check convomem_arena tests

# setup check against real Hub data, all three modes
for m in context tools files; do
  uv run vf-validate convomem-arena -n 6 --only-setup --runtime.type subprocess \
    --taskset.task.memory-mode $m --taskset.context-sizes 10
done
```

`tests/test_convomem.py` covers: speaker normalization; mixing (evidence kept, order preserved, determinism, contexts smaller than the evidence count); the reference transcript layout; RIGHT/WRONG parsing; the category → judge mapping; evidence substitution in judge prompts; BM25 search, paging and bounds; round-robin ordering; prompt construction per mode; and seed reproducibility. Hub access is replaced by an in-memory `FakeSource`. `tests/test_judge_check.py` and `tests/test_data_check.py` cover the [quality checkers](../quality/judge-robustness.md) with a fake generator, fake judges and hand-built items.

**Testing the whole pipeline without an API key:** the agent client and the judge both take a `base_url`. Pointing both at a small scripted OpenAI-compatible server (one that calls `memory_search`, answers from the result, and returns "RIGHT" to judge prompts) exercises tool-server launch, state sync, metrics and judge dispatch:

```bash
uv run vf-run convomem-arena -n 6 -r 1 -m fake \
  --client.base-url http://127.0.0.1:8765/v1 --client.api-key-var FAKE_KEY \
  --env.taskset.task.judge.base-url http://127.0.0.1:8765/v1 --env.taskset.task.judge.api-key-var FAKE_KEY \
  --env.agent.harness.id null --env.agent.runtime.type subprocess --env.taskset.task.memory-mode tools
```
