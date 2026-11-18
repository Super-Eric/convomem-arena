---
title: "Running evaluations"
description: "Recipes for context-length curves, agentic memory, files mode, custom judges and suggested experiments."
---

# Running evaluations

```bash
# Long-context baseline at two history lengths (paired: same items at both sizes)
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.context-sizes 10 30

# Agentic memory on histories too long for context
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.task.memory-mode tools --env.taskset.context-sizes 100 300

# Memory as files, solved by a coding harness in an isolated container
uv run vf-run convomem-arena -n 60 -r 1 -m <model> \
  --env.taskset.task.memory-mode files --env.agent.harness.id bash --env.agent.runtime.type docker

# One category only, harder multi-evidence items
uv run vf-run convomem-arena -n 20 -m <model> --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.categories changing_facts --env.taskset.evidence-counts 4 5 6

# Custom judge endpoint (any OpenAI-compatible API)
uv run vf-run convomem-arena -n 6 -m <model> ... \
  --env.taskset.task.judge.base-url https://api.openai.com/v1 \
  --env.taskset.task.judge.api-key-var OPENAI_API_KEY \
  --env.taskset.task.judge.model gpt-4.1-mini
```

For large-scale evaluation and RL, use prime-rl (`uv run eval convomem-arena …` from a prime-rl checkout with this package installed).

## Suggested experiments

- **Context-length curve:** `context` mode at sizes `1 10 30 50 100`, accuracy per category. The paper reports that long context degrades as histories grow.
- **Context vs. tools:** same items and sizes, `memory-mode context` vs. `tools`. Where does retrieval overtake the full history?
- **Retrieval vs. reasoning failures:** in `tools` mode, compare `correct` with `evidence_recall`. Low recall with a wrong answer is a search failure. High recall with a wrong answer is a reasoning or synthesis failure, which is common on `changing_facts`.
- **Abstention calibration:** does more history (more distractors) make the agent hallucinate more on `abstention`?
