"""Judge every case and variant with each judge model, then score robustness.

For each (judge, case) the baseline is graded `repeats` times, and so is each variant
the judge can see. The majority verdict of each group is compared with the
baseline's:

- a **same** variant (a paraphrase) is robust when its verdict equals the baseline's;
- a **flip** variant is robust when a RIGHT baseline becomes WRONG. If the baseline
  was not RIGHT the check is *inconclusive*: there was nothing to flip.

Repeats that disagree are reported as instability, the judge's own noise floor:
robustness only means something above it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from convomem_arena.judge_check.perturb import Case
from convomem_arena.judge_check.variants import JUDGE_KINDS, VARIANTS, expectation
from convomem_arena.judge_client import ChatFn, JudgeEndpoint, chat, judge_prompt, judge_rendered

BASELINE = "baseline"


def render(case: Case, variant: str) -> str:
    """The official judge prompt for `case` with one input swapped for `variant`."""
    fields = {
        "question": case.question,
        "answer": case.reference,
        "response": case.response,
        "evidence_messages": case.evidence,
    }
    if variant != BASELINE:
        value = case.perturbations[variant]
        target = VARIANTS[variant][0]
        key = {
            "question": "question",
            "evidence": "evidence_messages",
            "reference": "answer",
            "response": "response",
        }[target]
        fields[key] = value
    return judge_prompt(category=case.category, evidence_count=case.evidence_count, **fields)


def planned_variants(case: Case) -> list[str]:
    """Variants this case's judge can see, whose perturbation passed validation, and
    whose prompt really differs from the baseline's."""
    if not case.response:
        return []
    base = render(case, BASELINE)
    out = []
    for variant in VARIANTS:
        if expectation(variant, case.kind) is None or not case.usable(variant):
            continue
        if render(case, variant) != base:
            out.append(variant)
    return out


def _key(row: dict) -> tuple:
    return (row["judge"], row["case"], row["variant"], row["repeat"])


def judge_cases(
    cases: list[Case],
    judges: list[JudgeEndpoint],
    verdicts_path: Path,
    repeats: int = 3,
    workers: int = 4,
    chat_fn: ChatFn = chat,
) -> int:
    """Grade everything not already in `verdicts_path` (resumable). Returns calls made."""
    done = set()
    if verdicts_path.exists():
        done = {_key(json.loads(line)) for line in verdicts_path.read_text().splitlines() if line}
    jobs = []
    for judge in judges:
        for case in cases:
            for variant in [BASELINE, *planned_variants(case)]:
                for r in range(repeats):
                    if (judge.model, case.id, variant, r) not in done:
                        jobs.append((judge, case, variant, r))

    def run(job):
        judge, case, variant, r = job
        v = judge_rendered(judge, render(case, variant), chat_fn)
        return {
            "judge": judge.model,
            "case": case.id,
            "kind": case.kind,
            "category": case.category,
            "variant": variant,
            "repeat": r,
            "right": v.right,
            "raw": v.raw[:200],
            "error": v.error,
        }

    with verdicts_path.open("a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(run, jobs), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"  judged {i}/{len(jobs)}", flush=True)
    return len(jobs)


# ------------------------------------------------------------------ scoring


def _majority(values: list[bool | None]) -> bool | None:
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    (top, n), *rest = counts.most_common()
    return None if rest and rest[0][1] == n else top


def score(rows: Iterable[dict]) -> list[dict]:
    """One result per (judge, case, variant): majority verdict, stability, and the
    robustness outcome against the baseline (pass / fail / inconclusive / n/a)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["judge"], row["case"], row["variant"])].append(row)
    results = []
    for (judge, case, variant), rs in groups.items():
        verdicts = [r["right"] for r in rs]
        results.append(
            {
                "judge": judge,
                "case": case,
                "kind": rs[0]["kind"],
                "category": rs[0]["category"],
                "variant": variant,
                "verdict": _majority(verdicts),
                "stable": len(set(verdicts)) == 1 and None not in verdicts,
                "parse_failures": sum(r["error"] is not None for r in rs),
                "n": len(rs),
            }
        )
    base = {(r["judge"], r["case"]): r["verdict"] for r in results if r["variant"] == BASELINE}
    for r in results:
        if r["variant"] == BASELINE:
            r["expect"], r["outcome"] = (
                "right",
                ("pass" if r["verdict"] is True else "fail" if r["verdict"] is False else "n/a"),
            )
            continue
        expect = expectation(r["variant"], r["kind"])
        b, v = base.get((r["judge"], r["case"])), r["verdict"]
        r["expect"] = expect
        if b is None or v is None:
            r["outcome"] = "n/a"
        elif expect == "same":
            r["outcome"] = "pass" if v == b else "fail"
        elif b is not True:
            r["outcome"] = "inconclusive"
        else:
            r["outcome"] = "pass" if v is False else "fail"
    return results


def _rate(rs: list[dict]) -> str:
    decided = [r for r in rs if r["outcome"] in ("pass", "fail")]
    if not decided:
        return "–"
    passed = sum(r["outcome"] == "pass" for r in decided)
    return f"{100 * passed / len(decided):.0f}% ({passed}/{len(decided)})"


def report(results: list[dict], cases: list[Case]) -> str:
    judges = sorted({r["judge"] for r in results})
    out = [
        "# ConvoMem judge robustness",
        "",
        (
            f"{len(cases)} items, {len(judges)} judge model(s). Each verdict is the majority "
            "of its repeats. **same** variants keep the meaning (a robust judge keeps its "
            "verdict); **flip** variants change it (a robust judge moves a RIGHT baseline "
            "to WRONG)."
        ),
        "",
        "## Summary",
        "",
        (
            "| judge | baseline RIGHT on correct responses | invariance (same) | "
            "sensitivity (flip) | unstable groups | parse failures |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for j in judges:
        rs = [r for r in results if r["judge"] == j]
        base = [r for r in rs if r["variant"] == BASELINE]
        same = [r for r in rs if r.get("expect") == "same"]
        flip = [r for r in rs if r.get("expect") == "flip"]
        unstable = sum(not r["stable"] for r in rs)
        fails = sum(r["parse_failures"] for r in rs)
        out.append(
            f"| {j} | {_rate(base)} | {_rate(same)} | {_rate(flip)} | "
            f"{unstable}/{len(rs)} | {fails} |"
        )
    out += [
        "",
        (
            "Rates count only decided checks. A flip check is *inconclusive* when the "
            "baseline was not RIGHT (nothing to flip) and is left out of the rate."
        ),
    ]
    for j in judges:
        rs = [r for r in results if r["judge"] == j]
        out += [
            "",
            f"## {j}",
            "",
            "| variant | expect | " + " | ".join(JUDGE_KINDS) + " | all | inconclusive |",
            "|---|---|" + "---|" * (len(JUDGE_KINDS) + 2),
        ]
        for variant in [BASELINE, *VARIANTS]:
            vr = [r for r in rs if r["variant"] == variant]
            if not vr:
                continue
            expect = vr[0]["expect"]
            cells = [_rate([r for r in vr if r["kind"] == k]) for k in JUDGE_KINDS]
            inconclusive = sum(r["outcome"] == "inconclusive" for r in vr)
            out.append(
                f"| {variant} | {expect} | "
                + " | ".join(cells)
                + f" | {_rate(vr)} | {inconclusive} |"
            )
        failures = [r for r in rs if r["outcome"] == "fail"]
        if failures:
            out += ["", f"Failed checks ({len(failures)}):", ""]
            base = {r["case"]: r["verdict"] for r in rs if r["variant"] == BASELINE}
            for r in failures:
                b = {True: "RIGHT", False: "WRONG", None: "?"}[base.get(r["case"])]
                v = {True: "RIGHT", False: "WRONG", None: "?"}[r["verdict"]]
                out.append(f"- `{r['case']}` {r['variant']}: baseline {b} → {v}")
    return "\n".join(out) + "\n"
