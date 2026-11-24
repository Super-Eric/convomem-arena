"""Evaluate a Vellum instance's memory on ConvoMem, one persona at a time.

    python -m convomem_arena.vellum_eval.run \
        --source ~/…/vellum-assistant-saj/.claude/worktrees/vellum-rsi-m2-mlflow \
        --instance vellum-rsi-eval --baseline ~/.vellum-rsi/eval/baseline.vbundle \
        --persona Financial_Advisor --per-category 1 --history-size 14

Stages (each writes into `outputs/vellum-eval/<run-id>/`):

1. **plan**: questions for the persona and one merged, time-ordered history.
2. **build** (Stage A): restore the clean baseline, import the history, run a memory
   retrospective per conversation (Vellum's own "what should I have remembered"
   pass, which calls `remember`), consolidate, and back the result up as this
   persona's memory snapshot (reused with `--reuse-snapshot`).
3. **ask** (Stage B): with automatic memory writing off, ask each question in its own
   new conversation and read the reply from the turn's MLflow trace.
4. **score**: ConvoMem's official judge per category, plus two diagnostics for
   fact categories: is the answer anywhere in memory (capture) and in what was
   injected for this question (retrieval).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from convomem_arena.data import FILLER_ROOT, HubSource
from convomem_arena.vellum_eval.instance import VellumInstance, reply_text
from convomem_arena.vellum_eval.judge import JudgeEndpoint, judge_answer
from convomem_arena.vellum_eval.plan import EvalPlan, build_plan, import_payload

FACT_CATEGORIES = {"user_facts", "assistant_facts", "changing_facts"}
STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "your",
    "you",
    "my",
    "our",
    "their",
    "his",
    "her",
    "they",
    "them",
    "there",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "how",
    "why",
    "should",
    "would",
    "could",
    "can",
    "will",
    "just",
    "also",
    "than",
    "then",
    "into",
    "about",
    "over",
}


def key_terms(answer: str, question: str = "") -> list[str]:
    """Content words of the gold answer that the question doesn't already contain.

    Gold answers often restate the question ("Your go-to spot for a casual weeknight
    dinner is Siam Spice"); only the new words ("siam", "spice") show the fact.
    """

    def words(text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9$][a-z0-9$.:'-]*", text.lower())
        return [w.strip(".:'-") for w in raw]

    asked = set(words(question))
    terms = [w for w in words(answer) if w not in STOP and len(w) >= 3 and w not in asked]
    return list(dict.fromkeys(terms))


def contains_answer(text: str, answer: str, threshold: float = 0.6, question: str = "") -> bool:
    """Rough: share of the answer's new key terms present in `text`."""
    terms = key_terms(answer, question)
    if not terms:
        return False
    low = text.lower()
    return sum(t in low for t in terms) / len(terms) >= threshold


def pick_persona(source: HubSource, needle: str) -> str:
    matches = [p for p in source.list_files(FILLER_ROOT) if needle.lower() in p.lower()]
    if not matches:
        raise SystemExit(f"no persona matches {needle!r}")
    return matches[0]


def stage_build(inst: VellumInstance, plan: EvalPlan, out: Path, args) -> dict:
    snapshot = (
        out.parent
        / "snapshots"
        / f"{plan.persona_file.removesuffix('.json')}-s{plan.seed}-h{len(plan.history)}.vbundle"
    )
    if args.reuse_snapshot and snapshot.exists():
        inst.restore(snapshot)
        return {"reused_snapshot": str(snapshot)}
    if args.baseline:
        inst.restore(Path(args.baseline).expanduser())
    # Retrospectives must be enabled even for the explicit runs below
    # (`runForkBasedRetrospective` returns "disabled" otherwise); the idle sweep
    # stays off. Both writers are switched off again before questions are asked.
    inst.config_set("memory.retrospective.enabled", "true")
    inst.config_set("memory.substrate.sweep_enabled", "false")
    t0 = time.time()
    imported = inst.import_history(import_payload(plan))
    ids = [inst.conversation_id_for_key(f"convomem:{p.id}") for p in plan.history]
    missing_import = [p.id for p, cid in zip(plan.history, ids) if not cid]
    retros = inst.run_retrospective_sweep([cid for cid in ids if cid])
    retros["not_imported"] = missing_import
    consolidation = inst.consolidate()
    # Nothing writes memory from here on: the questions must not change it.
    inst.config_set("memory.retrospective.enabled", "false")
    pages = inst.concept_pages()
    build = {
        "imported": imported,
        "retrospectives": retros,
        "consolidation": consolidation,
        "concept_pages": len(pages),
        "seconds": round(time.time() - t0),
    }
    if not args.no_snapshot:
        build["snapshot"] = str(inst.backup(snapshot))
    return build


def stage_ask(inst: VellumInstance, plan: EvalPlan, run_id: str) -> list[dict]:
    answers = []
    for i, q in enumerate(plan.questions, 1):
        key = f"convomem-eval:{run_id}:{q.id}"
        t = time.time()
        row = {"question_id": q.id, "category": q.category}
        try:
            user_message_id = inst.ask(key, q.question)
            trace = inst.wait_turn_trace(user_message_id)
            cid = inst.conversation_id_for_key(key)
            row.update(
                user_message_id=user_message_id,
                conversation_id=cid,
                trace_id=trace["trace_id"],
                reply=reply_text(trace),
                injected=inst.injected_memory(cid) if cid else [],
                seconds=round(time.time() - t),
            )
        except Exception as err:  # noqa: BLE001 - recorded per question
            row["error"] = str(err)[-300:]
        answers.append(row)
        print(
            f"  asked {i}/{len(plan.questions)} {q.category} ({row.get('seconds', '?')}s)",
            flush=True,
        )
    return answers


def stage_score(
    inst: VellumInstance, plan: EvalPlan, answers: list[dict], judge: JudgeEndpoint
) -> list[dict]:
    pages = inst.concept_pages()
    all_pages = "\n".join(pages.values())
    by_id = {q.id: q for q in plan.questions}
    scores = []
    for a in answers:
        q = by_id[a["question_id"]]
        row = {"question_id": q.id, "category": q.category, "trace_id": a.get("trace_id")}
        if "error" in a:
            row["error"] = a["error"]
            scores.append(row)
            continue
        verdict = judge_answer(
            judge,
            category=q.category,
            evidence_count=q.evidence_count,
            question=q.question,
            answer=q.answer,
            response=a["reply"],
            evidence_messages=q.evidence_messages,
        )
        row["right"] = verdict.right
        if verdict.error:
            row["judge_error"] = verdict.error
        if q.category in FACT_CATEGORIES:
            injected_text = "\n".join(pages.get(c["slug"], "") for c in a["injected"])
            row["answer_in_memory"] = contains_answer(all_pages, q.answer, question=q.question)
            row["answer_in_injection"] = contains_answer(
                injected_text, q.answer, question=q.question
            )
        row["injected_count"] = len(a.get("injected", []))
        scores.append(row)
    return scores


def report(plan: EvalPlan, build: dict, scores: list[dict]) -> str:
    cats: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        cats[s["category"]].append(s)

    def pct(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "–"

    lines = [
        f"# Vellum on ConvoMem: {plan.persona_role}",
        "",
        (
            f"History: {len(plan.history)} conversations "
            f"({sum(h.is_evidence for h in plan.history)} evidence). "
            f"Concept pages after build: {build.get('concept_pages', '?')}. "
            f"Build time: {build.get('seconds', '?')} s."
        ),
        "",
        "| category | judged right | answer in memory | answer in injection |",
        "|---|---|---|---|",
    ]
    for cat, rows in cats.items():
        lines.append(
            f"| {cat} | {pct(rows, 'right')} | {pct(rows, 'answer_in_memory')} | "
            f"{pct(rows, 'answer_in_injection')} |"
        )
    total = [s for s in scores if s.get("right") is not None]
    lines += ["", f"**Overall judged right: {sum(s['right'] for s in total)}/{len(total)}**"]
    pages = build.get("concept_pages")
    if isinstance(pages, int) and pages < 10:
        lines += [
            "",
            (
                f"> Note: only {pages} concept pages. Below 10, memory v3 runs its lean profile "
                "(no selector, no dense lane; every pooled line is injected), so this run does "
                "not exercise the selector."
            ),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vellum_eval")
    p.add_argument("--source", required=True, help="vellum-assistant checkout/worktree")
    p.add_argument("--instance", required=True)
    p.add_argument("--baseline", help=".vbundle to restore before building")
    p.add_argument("--persona", help="substring of the persona file name (new runs)")
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--history-size", type=int, default=14)
    p.add_argument("--max-evidence", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlflow-experiment", default="vellum-rsi-eval")
    p.add_argument("--judge-model", default="qwen3:30b-a3b")
    p.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    p.add_argument("--reuse-snapshot", action="store_true")
    p.add_argument("--no-snapshot", action="store_true")
    p.add_argument("--out", default="outputs/vellum-eval")
    p.add_argument("--stages", default="build,ask,score", help="comma list of build, ask, score")
    p.add_argument("--run-dir", help="resume an existing run directory (keeps its plan)")
    args = p.parse_args(argv)
    stages = set(args.stages.split(","))

    if args.run_dir:
        out = Path(args.run_dir)
        plan = EvalPlan.model_validate_json((out / "plan.json").read_text())
        run_id = out.name
    else:
        source = HubSource()
        persona = pick_persona(source, args.persona)
        plan = build_plan(
            persona,
            per_category=args.per_category,
            history_size=args.history_size,
            seed=args.seed,
            max_evidence=args.max_evidence,
            source=source,
        )
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out = Path(args.out) / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(plan.model_dump_json(indent=1))
    print(
        f"[plan] {plan.persona_role}: {len(plan.questions)} questions, "
        f"{len(plan.history)} conversations → {out}",
        flush=True,
    )

    inst = VellumInstance(
        args.instance, Path(args.source).expanduser(), mlflow_experiment=args.mlflow_experiment
    )
    if "build" in stages:
        print("[build] stage A", flush=True)
        build = stage_build(inst, plan, out, args)
        (out / "build.json").write_text(json.dumps(build, indent=1))
    build = json.loads((out / "build.json").read_text()) if (out / "build.json").exists() else {}
    if "ask" in stages:
        print("[ask] stage B", flush=True)
        answers = stage_ask(inst, plan, run_id)
        (out / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
    if "score" in stages:
        print("[score]", flush=True)
        answers = [json.loads(line) for line in (out / "answers.jsonl").read_text().splitlines()]
        scores = stage_score(
            inst, plan, answers, JudgeEndpoint(args.judge_base_url, args.judge_model)
        )
        (out / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
        text = report(plan, build, scores)
        (out / "report.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
