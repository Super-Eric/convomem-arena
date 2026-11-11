"""Build data.js for the tutorial page from the real convomem_arena repo and a real trace.

Usage (from the repo root, with the repo's venv):
    .venv/bin/python scrollcraft/builds/convomem-tutorial/build_data.py <traces.jsonl>

Everything the page shows is produced here from real artifacts:
- source excerpts extracted by AST from convomem_arena/*
- the ConvoMemData the taskset actually builds for one task (tools mode, ctx 20)
- real mix() outputs for the same item at several context sizes
- the three real prompts (context / tools / files)
- the episode recorded by `vf-run` (scripted endpoint, no LLM), trimmed for size
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "data.js"
sys.path.insert(0, str(ROOT))

from convomem_arena.data import CATEGORIES, REVISION  # noqa: E402
from convomem_arena.judges import judge_for  # noqa: E402
from convomem_arena.taskset import ConvoMemConfig, ConvoMemTaskConfig, ConvoMemTaskset  # noqa: E402

TASK_PREFIX = "user_evidence/1_evidence/e31fb8db-8641-408f-8724-23371bccecf8_HR_Manager/47/ctx"
SIZES = [1, 3, 10, 20]


def source(rel: str, *names: str) -> str:
    """Exact source of a top-level def/class, or `Class.method`."""
    path = ROOT / rel
    text = path.read_text()
    tree = ast.parse(text)
    out = []
    for name in names:
        parts = name.split(".")
        nodes = tree.body
        node = None
        for part in parts:
            node = next(
                n for n in nodes
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == part
            )
            nodes = node.body
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        lines = text.splitlines()[start - 1 : node.end_lineno]
        indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
        out.append("\n".join(l[indent:] for l in lines))
    return "\n\n".join(out)


def task_for(size: int, mode: str):
    config = ConvoMemConfig(
        categories=["user_facts"],
        evidence_counts=[1],
        context_sizes=[size],
        task=ConvoMemTaskConfig(memory_mode=mode),
    )
    for task in ConvoMemTaskset(config).take(40):
        if task.data.id == f"{TASK_PREFIX}{size}":
            return task
    raise LookupError(f"{TASK_PREFIX}{size} not in the first 40 tasks")


def main(trace_path: str) -> None:
    # --- the task, as the taskset builds it -------------------------------
    tools_task = task_for(20, "tools")
    data = tools_task.data
    context_prompt = task_for(20, "context").data.prompt
    files_prompt = task_for(20, "files").data.prompt

    mixes = []
    for size in SIZES:
        d = task_for(size, "tools").data
        mixes.append(
            {
                "size": size,
                "evidence": d.evidence_positions,
                "conversations": [
                    {
                        "messages": len(c.messages),
                        "opening": next((m.text for m in c.messages if m.speaker == "User"), "")[:90],
                    }
                    for c in d.conversations
                ],
            }
        )

    # --- the recorded episode ---------------------------------------------
    episode = json.loads(Path(trace_path).read_text().splitlines()[0])
    trace = episode["traces"][0]
    task_data = {k: v for k, v in episode["task"]["data"].items() if k != "conversations"}
    task_data["conversations"] = f"[{len(data.conversations)} conversations, kept on the evaluator]"
    nodes = [n["message"] for n in trace["nodes"]]
    recorded = {
        "id": episode["id"],
        "env": episode["env"],
        "task": {"type": episode["task"]["type"], "key": episode["task"]["key"], "data": task_data},
        "ok": episode["ok"],
        "trace": {
            "version": trace["version"],
            "id": trace["id"],
            "verifiers": trace["verifiers"],
            "agent": {
                "harness": trace["agent"]["config"]["harness"]["id"],
                "runtime": trace["agent"]["config"]["runtime"]["type"],
                "model": trace["agent"]["config"]["model"],
            },
            "tools": [t["name"] for t in trace["nodes"][0]["tools"]],
            "nodes": nodes,
            "calls": [{k: c[k] for k in ("node", "model", "finish_reason")} for c in trace["calls"]],
            "info": trace["info"],
            "rewards": trace["rewards"],
            "metrics": trace["metrics"],
            "stop_condition": trace["stop_condition"],
            "is_completed": trace["is_completed"],
            "timing": {
                phase: round(trace["timing"][phase]["end"] - trace["timing"][phase]["start"], 3)
                for phase in ("setup", "agent", "finalize", "scoring")
            },
        },
    }
    decoder, hits = json.JSONDecoder(), []
    content = next(m["content"] for m in nodes if m.get("role") == "tool").strip()
    while content:
        hit, end = decoder.raw_decode(content)
        hits.append(hit)
        content = content[end:].strip()
    recorded["trace"]["tool_hits"] = hits
    tool_defs = [
        {"name": t["name"], "description": " ".join(t["description"].split()), "parameters": t["parameters"]}
        for t in trace["nodes"][0]["tools"]
    ]

    # --- category -> judge, computed by the real mapping ------------------
    judges = []
    from verifiers.v1 import JudgeConfig

    for category, (_, counts) in CATEGORIES.items():
        for n in (counts[0], counts[-1]) if counts[0] != counts[-1] else (counts[0],):
            judge = judge_for(category, n, JudgeConfig())
            judges.append({"category": category, "evidence_count": n, "judge": type(judge).__name__,
                           "reward_name": judge.reward_name})

    # --- source + counts ---------------------------------------------------
    files = {}
    for rel in ["convomem_arena/__init__.py", "convomem_arena/taskset.py", "convomem_arena/data.py",
                "convomem_arena/models.py", "convomem_arena/judges.py", "convomem_arena/servers/memory.py",
                "tests/test_convomem.py", "pyproject.toml"]:
        files[rel] = len((ROOT / rel).read_text().splitlines())
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    test_count = sum(1 for line in collected.splitlines() if "::" in line)

    code = {
        "init": (ROOT / "convomem_arena/__init__.py").read_text().strip(),
        "pyproject": (ROOT / "pyproject.toml").read_text().strip(),
        "taskdata": source("convomem_arena/models.py", "ConvoMemData"),
        "state": source("convomem_arena/models.py", "MemoryState"),
        "configs": source("convomem_arena/taskset.py", "ConvoMemTaskConfig", "ConvoMemConfig"),
        "load": source("convomem_arena/taskset.py", "ConvoMemTaskset.load"),
        "mix": source("convomem_arena/data.py", "mix"),
        "toolsets": source("convomem_arena/taskset.py", "ConvoMemTask.toolsets"),
        "setup": source("convomem_arena/taskset.py", "ConvoMemTask.setup"),
        "search": source("convomem_arena/servers/memory.py", "MemoryToolset.setup_task", "MemoryToolset.search"),
        "correct": source("convomem_arena/taskset.py", "ConvoMemTask.correct"),
        "metric": source("convomem_arena/taskset.py", "ConvoMemTask.memory"),
        "stop": source("convomem_arena/taskset.py", "ConvoMemTask.turn_limit"),
        "parse": source("convomem_arena/judges.py", "parse_right_wrong"),
        "judge_for": source("convomem_arena/judges.py", "judge_for"),
    }

    payload = {
        "task": {
            "id": data.id,
            "question": data.question,
            "answer": data.answer,
            "category": data.category,
            "evidence_count": data.evidence_count,
            "context_size": data.context_size,
            "evidence_positions": data.evidence_positions,
            "evidence_messages": data.evidence_messages,
            "conversations": [
                [{"speaker": m.speaker, "text": m.text} for m in c.messages] for c in data.conversations
            ],
        },
        "prompts": {
            "context": {"length": len(context_prompt), "head": context_prompt[:520], "tail": context_prompt[-260:]},
            "tools": tools_task.data.prompt,
            "files": files_prompt,
        },
        "mixes": mixes,
        "recorded": recorded,
        "tool_defs": tool_defs,
        "judges": judges,
        "files": files,
        "test_count": test_count,
        "revision": REVISION,
        "code": code,
    }
    OUT.write_text("window.CONVOMEM = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), tests={test_count}")


if __name__ == "__main__":
    main(sys.argv[1])
