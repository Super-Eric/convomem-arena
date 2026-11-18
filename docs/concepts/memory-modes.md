---
title: "Memory modes"
description: "The same task delivered three ways: the full history in context, behind MCP memory tools, or as files."
---

# Memory modes

## `context`: the long-context baseline

The prompt is ConvoMem's official answer prompt (`EvaluationUtils.getModelAnswerPrompt`), with the history laid out exactly as `buildConversationContext` does it:

```
Answer the question based on the conversations below. If the User in the conversation refers to themselves ("I", "me", "my"), they are the person being asked about. Be direct and factual.

Conversation 1:
User: I'm trying to decide on a hike for this weekend…
Assistant: …

Conversation 2:
…

Question: <question>

Answer:
```

A conversation averages roughly **6,000 characters (~1.5k tokens)**. Rough prompt sizes:

| `context_size` | ≈ prompt tokens |
|---|---|
| 1 | 1.5k |
| 10 | 15k |
| 30 | 45k |
| 100 | 150k |
| 300 | 450k |

In this mode `TaskData.conversations` is left empty, because the history is already in the prompt and storing it twice would double memory.

## `tools`: agentic memory over MCP

The prompt contains only the question and the number of stored conversations:

```
Answer the question based on your past conversations with the User. The 20 conversations, in chronological order, are available through the memory tools. If the User in the conversations refers to themselves ("I", "me", "my"), they are the person being asked about. Be direct and factual.

Question: <question>
```

A **per-rollout** MCP server (`MemoryToolset`, `TOOL_PREFIX = "memory"`) receives the task's history in `setup_task` and exposes:

| Tool | Signature | Returns |
|---|---|---|
| `memory_search` | `(query: str, limit: int = 8)` | Top messages by Okapi BM25 (k1=1.5, b=0.75, light stopword list), each with `conversation`, `message`, `speaker`, full `text`. `limit` is clamped to 1–20 |
| `memory_read_conversation` | `(conversation: int, start: int = 1, count: int = 40)` | Consecutive messages of one conversation (1-based numbering) plus `total_messages`. `count` is clamped to 1–100 |
| `memory_list_conversations` | `(offset: int = 0, limit: int = 50)` | `total_conversations` and, per conversation, `num_messages` and the first 200 characters of its opening user message |

Every search or read records which conversations it surfaced on the shared `MemoryState`, which scoring reads back as retrieval metrics. Listing counts as a tool call but not as retrieval.

This is the mode for long histories (100–300 conversations) that don't fit in context, which is the regime the ConvoMem paper studies with memory systems.

## `files`: memory as a filesystem

`Task.setup` writes one file per conversation (`memory/conversation_001.txt` is the oldest) relative to the runtime's working directory. The prompt points the agent there and asks for the answer as the final message. There are no custom tools: the agent uses whatever the harness gives it (`bash`'s `grep`/`cat`, Claude Code's file tools, …). That follows prime-envs' preference for general harness tools over custom ones. **Run this mode on `docker` or `prime`** (see [Limitations](../reference/limitations.md)).
