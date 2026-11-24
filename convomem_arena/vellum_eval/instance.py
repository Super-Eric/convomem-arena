"""Drive one local Vellum instance through its own CLIs and read its local databases.

Everything goes through supported surfaces: `vellum` (hatch, sleep/wake, backup,
restore, message) and `assistant` (config, conversations import, memory
retrospective) from a chosen Vellum source tree, plus read-only queries against the
instance's SQLite databases and memory files. Turn completion and replies are read
from the instance's MLflow turn traces (Vellum-RSI M2), keyed by user message id.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CONSOLIDATION_JOBS = ("memory_v2_consolidate", "memory_v2_reembed", "memory_v3_maintain")
BUFFER_ENTRY = re.compile(r"^- ", re.MULTILINE)


@dataclass
class VellumInstance:
    name: str
    source_dir: Path
    """A vellum-assistant checkout (or worktree) with `cli/` and `assistant/`."""
    mlflow_url: str = "http://127.0.0.1:5055"
    mlflow_experiment: str = "vellum-rsi-eval"
    log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- paths

    @property
    def workspace(self) -> Path:
        return Path.home() / ".local/share/vellum/assistants" / self.name / ".vellum/workspace"

    def db(self, which: str = "assistant") -> Path:
        return self.workspace / "data/db" / f"{which}.db"

    # ---------------------------------------------------------- commands

    @property
    def ipc_socket_dir(self) -> Path:
        """Where this instance's assistant IPC socket lives: the Vellum CLI's own rule.

        The workspace, unless `<workspace>/assistant.sock` would exceed macOS's
        103-byte AF_UNIX limit; then `$TMPDIR/vellum-ipc-<sha256(workspace)[:12]>`.
        """
        ws = str(self.workspace)
        if len(f"{ws}/assistant.sock".encode()) <= 103 or os.uname().sysname != "Darwin":
            return self.workspace
        digest = hashlib.sha256(ws.encode()).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / f"vellum-ipc-{digest}"

    def _env(self, for_daemon: bool = False) -> dict[str, str]:
        env = dict(os.environ)
        # A shell profile may export another instance's workspace and socket dir
        # (e.g. a `vellum-env` helper). Inherited by a daemon, they make it bind
        # the other instance's socket; inherited by the CLI, they make commands
        # reach the other instance. Never pass them through.
        for var in ("VELLUM_WORKSPACE_DIR", "ASSISTANT_IPC_SOCKET_DIR", "GATEWAY_IPC_SOCKET_DIR"):
            env.pop(var, None)
        if not for_daemon:
            env["VELLUM_WORKSPACE_DIR"] = str(self.workspace)
            env["ASSISTANT_IPC_SOCKET_DIR"] = str(self.ipc_socket_dir)
        return env

    def _run(
        self, argv: list[str], timeout: float = 600, check: bool = True, for_daemon: bool = False
    ) -> str:
        self.log.append(" ".join(argv[:6]))
        # Run from the instance's workspace: some CLI paths drop marker files
        # (e.g. `.qdrant-initialized`) into the current directory.
        cwd = self.workspace if self.workspace.is_dir() else None
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(for_daemon),
            cwd=cwd,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"{' '.join(argv[:5])} … failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-800:]}"
            )
        return proc.stdout

    def vellum(self, *args: str, timeout: float = 600, check: bool = True) -> str:
        # `vellum` starts/stops daemons: give it a clean env so each daemon derives
        # its own workspace and socket dir.
        return self._run(
            ["bun", str(self.source_dir / "cli/src/index.ts"), *args],
            timeout,
            check,
            for_daemon=True,
        )

    def assistant(self, *args: str, timeout: float = 600, check: bool = True) -> str:
        return self._run(
            ["bun", str(self.source_dir / "assistant/src/index.ts"), *args], timeout, check
        )

    def config_set(self, key: str, value: str) -> None:
        self.assistant("config", "set", key, value)

    # ------------------------------------------------- snapshots (Stage A cache)

    # The CLIs run from the instance workspace (see `_run`), so paths handed to
    # them must be absolute.

    def backup(self, path: Path) -> Path:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vellum("backup", self.name, "--output", str(path), timeout=900)
        return path

    def restore(self, path: Path) -> None:
        self.vellum("restore", self.name, "--from", str(path.expanduser().resolve()), timeout=900)
        self.wait_healthy()

    def wait_healthy(self, timeout_s: float = 300) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            out = self.assistant("status", check=False, timeout=60)
            if "Assistant Version" in out:
                return
            time.sleep(3)
        raise TimeoutError(f"{self.name} did not come up")

    # ------------------------------------------------------------ history

    def import_history(self, payload: dict) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            out = self.assistant("conversations", "import", "--file", path, "--json", timeout=900)
        finally:
            os.unlink(path)
        return json.loads(out.strip().splitlines()[-1])

    def conversation_id_for_key(self, key: str) -> str | None:
        with sqlite3.connect(f"file:{self.db()}?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT conversation_id FROM conversation_keys WHERE conversation_key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def run_retrospective(self, conversation_id: str) -> dict:
        out = self.assistant(
            "memory", "retrospective", "run", conversation_id, "--json", timeout=1800
        )
        try:
            return json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return {"raw": out[-500:]}

    def retrospective_rows(self) -> list[dict]:
        out = self.assistant("memory", "retrospective", "list", "--limit", "500", "--json")
        try:
            return json.loads(out.strip().splitlines()[-1]).get("rows", [])
        except (json.JSONDecodeError, IndexError):
            return []

    def retrospective_state_rows(self) -> int:
        return len(self.retrospective_rows())

    def run_retrospective_sweep(
        self,
        conversation_ids: list[str],
        timeout_s: float = 7200,
        stall_s: float = 900,
        lookback_days: int = 120,
    ) -> dict:
        """Have the daemon's scheduled retrospective sweep review every conversation.

        This is Vellum's own path for messages no live turn reviewed: the daemon
        runs a fork-based retrospective per conversation (the agent calls
        `remember`). The CLI's `memory retrospective run` exits before the wake's
        tool calls finish, so it can't be used. The sweep is made frequent and its
        lookback wide enough for back-dated imports, then restored to defaults.
        Done when every conversation has a processed-state row, or progress stalls.
        """
        wanted = set(conversation_ids)
        self.config_set("memory.retrospective.enabled", "true")
        self.config_set("memory.retrospective.sweepIntervalMs", "60000")
        self.config_set("memory.retrospective.sweepLookbackMs", str(lookback_days * 86_400_000))
        start = last_change = time.time()
        done: set[str] = set()
        try:
            while time.time() - start < timeout_s and time.time() - last_change < stall_s:
                now_done = {
                    r.get("conversationId") or r.get("sourceConversationId")
                    for r in self.retrospective_rows()
                    if r.get("lastProcessedMessageId")
                } & wanted
                if now_done != done:
                    done, last_change = now_done, time.time()
                    print(
                        f"  retrospectives done {len(done)}/{len(wanted)} "
                        f"(buffer {self.buffer_entries()})",
                        flush=True,
                    )
                if done >= wanted:
                    break
                time.sleep(20)
        finally:
            self.config_set("memory.retrospective.sweepIntervalMs", str(8 * 3_600_000))
            self.config_set("memory.retrospective.sweepLookbackMs", str(7 * 86_400_000))
        return {
            "done": len(done),
            "wanted": len(wanted),
            "missing": sorted(wanted - done),
            "seconds": round(time.time() - start),
        }

    # ------------------------------------------------------- consolidation

    def buffer_entries(self) -> int:
        buf = self.workspace / "memory/buffer.md"
        return len(BUFFER_ENTRY.findall(buf.read_text())) if buf.exists() else 0

    def active_jobs(self, types: tuple[str, ...] = CONSOLIDATION_JOBS) -> int:
        marks = ",".join("?" * len(types))
        with sqlite3.connect(f"file:{self.db('assistant-memory')}?mode=ro", uri=True) as con:
            (n,) = con.execute(
                f"SELECT COUNT(*) FROM memory_jobs WHERE type IN ({marks}) "
                "AND status IN ('pending','running')",
                types,
            ).fetchone()
        return n

    def consolidate(self, timeout_s: float = 3600, normal_threshold: int = 100) -> dict:
        """Drain the buffer into concept pages now, then restore the normal trigger.

        Consolidation normally waits hours (or 100 buffered lines); a threshold of 1
        makes the worker start a pass on its next check. Returns timing and counts.
        """
        start = time.time()
        before = self.buffer_entries()
        self.config_set("memory.substrate.consolidation_max_buffer_lines", "1")
        try:
            idle_since = None
            while time.time() - start < timeout_s:
                busy = self.buffer_entries() > 0 or self.active_jobs() > 0
                if not busy:
                    idle_since = idle_since or time.time()
                    if time.time() - idle_since > 20:  # stable: nothing re-queued
                        break
                else:
                    idle_since = None
                time.sleep(5)
            else:
                raise TimeoutError("consolidation did not drain")
        finally:
            self.config_set(
                "memory.substrate.consolidation_max_buffer_lines", str(normal_threshold)
            )
        return {"buffer_before": before, "seconds": round(time.time() - start)}

    def concept_pages(self) -> dict[str, str]:
        root = self.workspace / "memory/concepts"
        return {
            str(p.relative_to(root).with_suffix("")): p.read_text(errors="replace")
            for p in root.rglob("*.md")
        }

    # ------------------------------------------------------------- asking

    def ask(self, conversation_key: str, question: str) -> str:
        """Send `question` as a new conversation turn; returns the user message id."""
        out = self.vellum(
            "message", self.name, "--conversation-key", conversation_key, question, timeout=300
        )
        m = re.search(r"id:\s*([0-9a-f-]{36})", out)
        if not m:
            raise RuntimeError(f"message not accepted: {out[-300:]}")
        return m.group(1)

    def _mlflow(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.mlflow_url + path,
            json.dumps(body).encode() if body is not None else None,
            {"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def experiment_id(self) -> str:
        name = urllib.parse.quote(self.mlflow_experiment)
        return self._mlflow(f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}")[
            "experiment"
        ]["experiment_id"]

    def wait_turn_trace(self, user_message_id: str, timeout_s: float = 900) -> dict:
        """The finished turn's MLflow trace (root + spans), found by its user message id."""
        exp = self.experiment_id()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            found = self._mlflow(
                "/api/3.0/mlflow/traces/search",
                {
                    "locations": [
                        {"type": "MLFLOW_EXPERIMENT", "mlflow_experiment": {"experiment_id": exp}}
                    ],
                    "filter": f"tags.`vellum.user_message_id` = '{user_message_id}'",
                    "max_results": 1,
                },
            ).get("traces", [])
            if found:
                trace_id = found[0]["trace_id"]
                spans = self._mlflow(
                    f"/ajax-api/3.0/mlflow/get-trace-artifact?request_id={trace_id}"
                ).get("spans", [])
                return {"trace_id": trace_id, "spans": spans}
            time.sleep(5)
        raise TimeoutError(f"no turn trace for {user_message_id}")

    def injected_memory(self, conversation_id: str) -> list[dict]:
        """What memory v3 chose for the conversation's first turn: slug, section, lane."""
        with sqlite3.connect(f"file:{self.db('assistant-memory')}?mode=ro", uri=True) as con:
            row = con.execute(
                "SELECT candidates_json FROM memory_v3_pools WHERE conversation_id = ? "
                "ORDER BY turn LIMIT 1",
                (conversation_id,),
            ).fetchone()
        if not row:
            return []
        return [c for c in json.loads(row[0]) if c.get("chosen")]


def reply_text(trace: dict) -> str:
    """The final reply recorded on the turn trace's root span."""
    for span in trace["spans"]:
        if not span.get("parent_span_id"):
            outputs = (span.get("attributes") or {}).get("mlflow.spanOutputs")
            if outputs:
                value = json.loads(outputs) if isinstance(outputs, str) else outputs
                value = json.loads(value) if isinstance(value, str) else value
                reply = value.get("reply", "")
                return reply if isinstance(reply, str) else json.dumps(reply)
    return ""
