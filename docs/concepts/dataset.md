---
title: "The dataset"
description: "Salesforce ConvoMem: personas, evidence items, filler conversations and the six memory categories."
---

# The dataset

**Source:** `Salesforce/ConvoMem` on Hugging Face, pinned to revision `e3e9b39115b02346824c70d349350de738f8be41`. 75,336 question–answer items across 100 synthetic personas (CRM-flavored professions: Account Executive, Data Architect, Telemarketer, …).

The environment reads two folders:

```
core_benchmark/
├── evidence_questions/<category>_evidence/<N>_evidence/<persona-uuid>_<Title>.json   # questions + evidence conversations
└── filler_conversations/<persona-uuid>_<Title>.json                                  # 400 unrelated conversations per persona
```

Each evidence item looks like this, abridged from a real record:

```json
{
  "question": "For the new software vendor contract, what was the precise liability cap in USD that I established as my absolute non-negotiable limit?",
  "answer": "You set your absolute non-negotiable liability cap for the new vendor at $2,500,000.",
  "message_evidences": [{"speaker": "User", "text": "...my absolute walk-away is a $2.5M cap..."}],
  "conversations": [{"id": "…", "messages": [{"speaker": "User", "text": "…"}, {"speaker": "Assistant", "text": "…"}]}],
  "personId": "af47b47b-…", "category": "Professional Life", "scenario_description": "…"
}
```

## Categories

| Category (`--env.taskset.categories`) | Folder | What it tests | Evidence counts | Example |
|---|---|---|---|---|
| `user_facts` | `user_evidence` | Recall of facts the user stated, possibly combined across messages | 1–6 | "What liability cap did I set?" → "$2,500,000" |
| `assistant_facts` | `assistant_facts_evidence` | Recall of what the assistant itself said or recommended | 1–6 | "What KPI did you suggest for the advocacy program?" → "Advocate Sourced Revenue" |
| `changing_facts` | `changing_evidence` | Tracking updates; only the latest value is correct | 2–6 | "When is the CRM update scheduled?" → the last rescheduled date |
| `abstention` | `abstention_evidence` | Recognizing that the answer was never given | 1–3 | "What is the projected contract value…?" → should say it doesn't know |
| `preferences` | `preference_evidence` | Recommendations that respect stated preferences; the answer is a **rubric** | 1–2 | "Recommend a lightweight PM tool" → rubric: simple, visual tools |
| `implicit_connections` | `implicit_connection_evidence` | Connecting indirect context; the answer is a **rubric** | 1–3 | "How should we streamline proposals?" → rubric: agile, low-bureaucracy |

The **evidence count** is the number of conversations the evidence is spread across. Multi-evidence items require combining information from several conversations.

To audit items before trusting results on them, see [data quality](../quality/data-quality.md).

Speaker labels in the raw data mix `User`/`user` and `Assistant`/`assistant`. They're normalized to `User` / `Assistant`.
