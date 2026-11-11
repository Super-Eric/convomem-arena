# BRIEF: convomem-arena tutorial

Interviewed on three structural forks (2026-11-09). The remaining topics are
**authored decisions** derived from the conversation that produced convomem-arena,
labelled as such. No user quotations are invented.

## Interview answers (verbatim selections)

| Question | Answer |
|---|---|
| Page grammar | **Live surface** |
| Aesthetic family | **Dense / technical** |
| The one moment to remember | **Watch a trace assemble** |

## The eight topics

1. **Vibe** (authored): precise, inspectable, honest. References: a debugger's
   variables pane, an aircraft flight recorder readout, a well-kept lab notebook.
2. **Journey** (authored from the user's request "teach me about environment hub,
   verifiers, and prime-env by utilizing examples from your experience creating
   convomem_arena"): one real rollout of convomem-arena, followed stage by stage.
   Hub resolves the package, the package's parts, the taskset builds a task and
   mixes the haystack, setup and memory modes, the agent's real tool call, the
   visitor's own memory search, scoring and the completed trace, prime-envs
   conventions checked against this package, and finally a run command they build.
3. **Energy curve** (authored): calm and orienting at the top, tension at the mix,
   hands-on in the middle, a held quiet before scoring, the loudest change at the
   trace takeover, then quiet competence and a working input.
4. **Feeling and peak**: see the curve and peak below. Peak chosen by the user:
   the trace assembling.
5. **The thing no other page does** (authored, seeded by the peak answer): the
   margin is a real verifiers trace, and every field in it is written by the stage
   that writes it in the framework.
6. **Range**: dense / technical (user's choice). Information-forward, small precise
   type, monospace only for real code and data.
7. **One world or scenes**: answered by the grammar. A single operable surface
   whose state advances; not a continuous flythrough.
8. **Assets**: none supplied; none generated. Everything on screen is real markup
   computed from real data: the convomem_arena source, one real ConvoMem task, and
   one real `vf-run` trace.

## Honesty contract (live-surface rule)

- `data.js` is produced by `build_data.py` from this repo: real source excerpts
  (extracted by AST), the real `ConvoMemData` for task
  `user_evidence/1_evidence/e31fb8db-…_HR_Manager/47/ctx20` built by the actual
  taskset code, real `mix()` outputs at several context sizes, and the trace
  recorded by an actual `vf-run`.
- That run used a **scripted endpoint, not an LLM**: the policy calls
  `memory_search` with the question and answers from the top hit; the judge
  endpoint replies `RIGHT`. The page says this wherever those values appear.
- In-page logic is real and runs in the browser: BM25 (ported from
  `servers/memory.py`, checked against the Python results), the RIGHT/WRONG
  parser, the category to judge mapping, the Hub id to module resolution, the
  convention checks, and the command composer's validation.

## Journey (beats)

1. Orientation: the package is resolved from a Hub id.
2. Breadth: the parts of the package, laid out.
3. Tension: one fact buried among nineteen conversations.
4. Clarity: the same task, three ways in.
5. Motion: the agent's real calls, one node at a time.
6. Control: the visitor searches the memory themselves.
7. Completion (peak): judged, scored, and the trace takes the screen.
8. Competence: the conventions, checked against this code.
9. Readiness: a command they can run.

## Feeling curve

```
1  Orientation   a working surface already resolved: hub id to module to class
2  Curiosity     the package's files sliding past, each with its real code
3  Tension       context_size grows and one highlighted conversation sinks among the rest
4  Clarity       a wipe from a 181k-character prompt to a three-sentence one
5  Motion        the recorded rollout advancing node by node: prompt, tool call, hits, reply
6  Control       the visitor types a query and the evidence lights up or does not
7  Completion    a held quiet status line, then the judge verdict, then the trace takes over the screen
8  Competence    each convention passes or fails on this package's actual code
9  Readiness     an input with a cursor in it that turns flags into a runnable command
```

No two adjacent feelings repeat.

## The peak

> "the trace I'd been watching build in the margin suddenly took over the whole
> screen, finished, with the reward in it, and I could copy it"

Lives in act 7 (Scoring). Largest span on the page. Authored silence: the first
~18% of act 7 is a single status line (`finalize → scoring`) on an otherwise empty
stage. That quiet is deliberate, not dead scroll.

## Tell-someone sentence

It's the site where **a real agent trace writes itself in the margin while you read,
and then takes over the screen when the judge scores it.**

## Signature move

**The trace margin.** A persistent panel holding a real `traces.jsonl` episode. Each
field is stamped with the lifecycle stage that writes it in verifiers (`load`,
`setup`, `agent`, `scoring`) and appears only once the reader passes that stage.
At the peak it leaves the margin and fills the whole canvas, newest fields first,
complete and copyable. Below 1280px the margin is a bottom drawer the reader can
open at any stage, and the peak takeover fills the canvas exactly as on desktop.

## Feel check (cold pass, then diffed)

| Act | Intended | Felt on first pass | Change |
|---|---|---|---|
| 1 Hub | orientation | orientation | none |
| 2 Package | curiosity | curiosity | none |
| 3 Load | tension | **measurement** (evidence tile a solid block at every size) | evidence tile now sinks to an outline at ctx 10 and 20 |
| 4 Setup | clarity | clarity | none |
| 5 Agent | motion | motion | none |
| 6 Tools | control | control | none |
| 7 Scoring | completion | completion, but the takeover covered only the centre column and buried the scoring fields | takeover now spans the whole canvas, newest fields first, content held until the wipe lands |
| 8 Conventions | competence | competence | none |
| 9 Run | readiness | readiness | none |
