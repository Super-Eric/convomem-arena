"""Data quality checker for ConvoMem.

    python -m convomem_arena.data_check --out outputs/data-check/run1
    python -m convomem_arena.data_check --out outputs/data-check/full --files-per-folder 1000 \
        --items-per-file 1000 --filler-personas 100      # the whole dataset (large download)

Samples items across many personas (a few per persona file, in every category and
evidence-count folder), runs the checks in `checks.py`, and writes `report.md`,
`flags.jsonl` (every item and its flags) and `review.md` (flagged items with their text,
for a human to confirm or dismiss). Filler checks download each sampled persona's
filler file (~5.6 MB); `--filler-personas` caps how many.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from convomem_arena.data import CATEGORIES, EVIDENCE_ROOT, FILLER_ROOT, HubSource
from convomem_arena.data_check.checks import (
    CHECKS,
    Conv,
    ItemResult,
    RawItem,
    check_content,
    check_duplicates,
    check_filler,
    check_structure,
    parse_raw,
)


def sample_items(
    source: HubSource,
    seed: int,
    files_per_folder: int,
    items_per_file: int,
    categories: list[str],
) -> list[RawItem]:
    """Up to `items_per_file` items from each of `files_per_folder` persona files, in
    every evidence-count folder of every category; seeded and spread across personas."""
    items = []
    for category in categories:
        folder_name, counts = CATEGORIES[category]
        for n in counts:
            folder = f"{EVIDENCE_ROOT}/{folder_name}/{n}_evidence"
            rng = random.Random(f"{seed}:{folder}")
            files = source.list_files(folder)
            rng.shuffle(files)
            for persona_file in files[:files_per_folder]:
                raw_items = source.read_json(f"{folder}/{persona_file}")["evidence_items"]
                order = list(range(len(raw_items)))
                rng.shuffle(order)
                for idx in order[:items_per_file]:
                    items.append(parse_raw(raw_items[idx], category, n, persona_file, idx))
    return items


def load_filler(source: HubSource, persona_file: str) -> list[Conv]:
    payload = source.read_json(f"{FILLER_ROOT}/{persona_file}")
    return [
        Conv(
            id=str(c.get("id") or ""),
            messages=[
                {"speaker": m.get("speaker", ""), "text": m.get("text", "")}
                for m in c.get("messages") or []
            ],
        )
        for item in payload["evidence_items"]
        for c in item["conversations"]
    ]


def run_checks(
    items: list[RawItem], filler_for: dict[str, list[Conv]] | None = None
) -> list[ItemResult]:
    results = [ItemResult(item=it) for it in items]
    for r in results:
        check_structure(r)
        check_content(r)
        if filler_for is not None and r.item.persona_file in filler_for:
            check_filler(r, filler_for[r.item.persona_file])
    check_duplicates(results)
    return results


def report(results: list[ItemResult], filler_checked: set[str]) -> str:
    cats = list(dict.fromkeys(r.item.category for r in results))
    by_cat: dict[str, list[ItemResult]] = defaultdict(list)
    for r in results:
        by_cat[r.item.category].append(r)
    personas = {r.item.persona_file for r in results}
    filler_items = sum(r.item.persona_file in filler_checked for r in results)
    out = [
        "# ConvoMem data quality",
        "",
        (
            f"{len(results)} items from {len(personas)} personas. Filler checks ran on "
            f"{filler_items} items ({len(filler_checked)} personas)."
        ),
        "",
        (
            "**error** = structurally inconsistent; **review** = a heuristic pointer that "
            "needs a human look (see review.md)."
        ),
        "",
        "| check | severity | " + " | ".join(cats) + " | all |",
        "|---|---|" + "---|" * (len(cats) + 1),
        "| items checked | | "
        + " | ".join(str(len(by_cat[c])) for c in cats)
        + f" | {len(results)} |",
    ]
    for check, (severity, _) in CHECKS.items():
        counts = [sum(any(f.check == check for f in r.flags) for r in by_cat[c]) for c in cats]
        total = sum(counts)
        if not total:
            continue
        cells = [
            f"{n} ({100 * n / len(by_cat[c]):.0f}%)" if n else "–" for n, c in zip(counts, cats)
        ]
        out.append(f"| {check} | {severity} | " + " | ".join(cells) + f" | {total} |")
    clean = sum(not r.flags for r in results)
    out += [
        "",
        f"**Items with no flags: {clean}/{len(results)}.**",
        "",
        "## What each check means",
        "",
    ]
    out += [f"- `{c}` ({s}): {d}" for c, (s, d) in CHECKS.items()]
    return "\n".join(out) + "\n"


def review_markdown(results: list[ItemResult]) -> str:
    out = ["# Flagged ConvoMem items to review", ""]
    for r in results:
        if not r.flags:
            continue
        it = r.item
        out += [f"## {it.id}", ""]
        out += [f"- ⚑ **{f.check}** ({f.severity}): {f.detail}" for f in r.flags]
        out += ["", f"- **question:** {it.question}", f"- **answer:** {it.answer}"]
        out += [
            f"- **evidence {i + 1} ({e.speaker}):** {e.text}" for i, e in enumerate(it.evidence)
        ]
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m convomem_arena.data_check")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--files-per-folder", type=int, default=5)
    p.add_argument("--items-per-file", type=int, default=3)
    p.add_argument("--filler-personas", type=int, default=25, help="0 skips filler checks")
    p.add_argument("--category", action="append", choices=list(CATEGORIES))
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = HubSource()
    items = sample_items(
        source,
        args.seed,
        args.files_per_folder,
        args.items_per_file,
        args.category or list(CATEGORIES),
    )
    print(f"[data-check] sampled {len(items)} items", flush=True)

    counts = Counter(it.persona_file for it in items)
    chosen = sorted(counts, key=lambda pf: (-counts[pf], pf))[: args.filler_personas]
    filler_for = {}
    for i, pf in enumerate(chosen, 1):
        filler_for[pf] = load_filler(source, pf)
        print(f"  filler {i}/{len(chosen)} loaded", flush=True)

    results = run_checks(items, filler_for)
    (out / "flags.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results))
    (out / "review.md").write_text(review_markdown(results))
    text = report(results, set(filler_for))
    (out / "report.md").write_text(text)
    print(text)
    by_check = Counter(f.check for r in results for f in r.flags)
    print(json.dumps(by_check, indent=1))


if __name__ == "__main__":
    main()
