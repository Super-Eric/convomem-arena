"""Judge robustness checker for ConvoMem's official judges.

    # 1. sample items and generate perturbations (cached in the run directory)
    python -m convomem_arena.judge_check perturb --per-category 4 --out outputs/judge-check/run1

    # 2. grade baseline + every variant, 3 repeats each, with one or more judges
    python -m convomem_arena.judge_check judge --run-dir outputs/judge-check/run1 \
        --judge-model qwen3:30b-a3b --judge-model gpt-oss:20b --repeats 3

    # 3. write report.md (also printed)
    python -m convomem_arena.judge_check report --run-dir outputs/judge-check/run1

Read `review.md` before trusting a report: it shows every perturbation next to its
original, and a bad paraphrase makes a robust judge look fragile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convomem_arena.judge_check.perturb import Case, perturb, review_markdown, sample_cases
from convomem_arena.judge_check.run import judge_cases, report, score
from convomem_arena.judge_client import JudgeEndpoint


def load_cases(run_dir: Path) -> list[Case]:
    return [
        Case.model_validate_json(line)
        for line in (run_dir / "cases.jsonl").read_text().splitlines()
        if line
    ]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m convomem_arena.judge_check")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("perturb", help="sample items and generate perturbations")
    g.add_argument("--out", required=True)
    g.add_argument("--per-category", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--generator-model", default="qwen3:30b-a3b")
    g.add_argument("--generator-base-url", default="http://127.0.0.1:11434/v1")
    g.add_argument("--generator-api-key-var")
    g.add_argument(
        "--generator-think", action="store_true", help="Ollama: let the generator reason"
    )

    j = sub.add_parser("judge", help="grade baseline and variants with each judge")
    j.add_argument("--run-dir", required=True)
    j.add_argument("--judge-model", action="append", required=True)
    j.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    j.add_argument("--judge-api-key-var")
    j.add_argument("--repeats", type=int, default=3)
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--temperature", type=float, default=0.0)

    r = sub.add_parser("report", help="score robustness and write report.md")
    r.add_argument("--run-dir", required=True)

    args = p.parse_args(argv)

    if args.cmd == "perturb":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        gen = JudgeEndpoint(
            base_url=args.generator_base_url,
            model=args.generator_model,
            api_key_var=args.generator_api_key_var,
            think=args.generator_think,
            temperature=0.0,
        )
        cases = sample_cases(args.per_category, args.seed)
        print(f"[perturb] {len(cases)} items, generator {gen.model}", flush=True)
        with (out / "cases.jsonl").open("w") as f:
            for i, case in enumerate(cases, 1):
                perturb(case, gen)
                f.write(case.model_dump_json() + "\n")
                f.flush()
                bad = sum(bool(v) for v in case.problems.values())
                print(f"  {i}/{len(cases)} {case.id} ({bad} field(s) flagged)", flush=True)
        (out / "review.md").write_text(review_markdown(cases))
        print(f"[perturb] wrote {out / 'cases.jsonl'} and {out / 'review.md'}")

    elif args.cmd == "judge":
        run_dir = Path(args.run_dir)
        cases = load_cases(run_dir)
        judges = [
            JudgeEndpoint(
                base_url=args.judge_base_url,
                model=m,
                api_key_var=args.judge_api_key_var,
                temperature=args.temperature,
            )
            for m in args.judge_model
        ]
        n = judge_cases(cases, judges, run_dir / "verdicts.jsonl", args.repeats, args.workers)
        print(f"[judge] {n} new judge calls → {run_dir / 'verdicts.jsonl'}")

    elif args.cmd == "report":
        run_dir = Path(args.run_dir)
        cases = load_cases(run_dir)
        rows = [
            json.loads(line)
            for line in (run_dir / "verdicts.jsonl").read_text().splitlines()
            if line
        ]
        results = score(rows)
        (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
        text = report(results, cases)
        (run_dir / "report.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
