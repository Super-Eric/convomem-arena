---
title: "Quick start: evaluate a model"
description: "Install convomem-arena, check that tasks load, and run your first episodes against a hosted or local model."
---

# Quick start: evaluate a model

You need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

## 1. Install

```bash
git clone https://github.com/rahmansunny071/convomem-arena.git
cd convomem-arena
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## 2. Check that tasks load

This builds six tasks, one per category, and sets each one up. It makes no model calls and needs no API key.

```bash
uv run vf-validate convomem-arena -n 6 --only-setup --runtime.type subprocess
```

`-n 6` already covers all six categories, because tasks are interleaved across categories ([why](../concepts/tasks.md#ordering-and-laziness)).

## 3. Run your first episodes

The run needs two models: the **agent** that answers, and the **judge** that grades the answer.

=== "Prime Inference"

    By default both use Prime Inference, with the key in `PRIME_API_KEY`.

    ```bash
    export PRIME_API_KEY=...
    uv run vf-run convomem-arena -n 6 -r 1 -m <model> \
      --env.agent.harness.id null --env.agent.runtime.type subprocess
    ```

=== "Local (Ollama)"

    ## Run fully local with Ollama

    No API key needed: `qwen3:30b-a3b` serves as both the agent and the judge (verified end to end: 6/6, mean reward 1.000).

    ```bash
    # Ollama must be serving the model at 127.0.0.1:11434 first:
    #   ollama pull qwen3:30b-a3b && ollama run qwen3:30b-a3b "reply with one word: ok"

    export OLLAMA_FAKE_KEY=unused   # ignored by Ollama; vf-run just needs the variable set

    uv run vf-run convomem-arena -n 6 -r 1 -m qwen3:30b-a3b \
      --client.base-url http://127.0.0.1:11434/v1 --client.api-key-var OLLAMA_FAKE_KEY \
      --env.agent.harness.id null --env.agent.runtime.type subprocess \
      --env.taskset.context-sizes 10 \
      --env.taskset.task.judge.base-url http://127.0.0.1:11434/v1 \
      --env.taskset.task.judge.api-key-var OLLAMA_FAKE_KEY \
      --env.taskset.task.judge.model qwen3:30b-a3b
    ```

    !!! warning "A smoke test, not a measurement"
        A 30B model judging its own transcript is weaker than a frontier judge on the rubric-graded categories (`preferences`, `implicit_connections`). Ten conversations of history is also an easy setting. Treat a perfect score here as "it works", not as a result.

## 4. Look at the results

`vf-run` writes one line per episode to `traces.jsonl`. [Reading results](../guides/reading-results.md) shows which fields matter and how to compute accuracy per category.

## Next steps

<div class="grid cards" markdown>

-   :material-tune-variant: **Try the other memory modes**

    ---

    Put the history behind search tools, or in files, instead of in the prompt.

    [:octicons-arrow-right-24: Memory modes](../concepts/memory-modes.md)

-   :material-flask-outline: **Run real experiments**

    ---

    Context-length curves, context versus tools, and retrieval versus reasoning failures.

    [:octicons-arrow-right-24: Running evaluations](../guides/running-evaluations.md)

-   :material-scale-balance: **Check your judge**

    ---

    Find out whether the judge model grades what you think it grades.

    [:octicons-arrow-right-24: Judge robustness](../quality/judge-robustness.md)

-   :material-robot-outline: **Evaluate an assistant's memory**

    ---

    Make a personal assistant live through a persona's history, then ask it.

    [:octicons-arrow-right-24: Personal assistant quick start](personal-assistant.md)

</div>
