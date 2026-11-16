"""Vellum eval harness: planning, judge prompts, scoring helpers, instance plumbing.

No network, model or Vellum instance: a tiny in-memory ConvoMem source stands in for
the Hugging Face Hub.
"""

import json
from pathlib import Path

from convomem_arena.data import EVIDENCE_ROOT, FILLER_ROOT
from convomem_arena.vellum_eval.instance import VellumInstance, reply_text
from convomem_arena.vellum_eval.judge import judge_prompt
from convomem_arena.vellum_eval.plan import DAY_MS, build_plan, import_payload
from convomem_arena.vellum_eval.run import contains_answer, key_terms

PERSONA = "00000000-0000-0000-0000-000000000000_Test_Owner.json"


def conv(cid, *texts):
    msgs = [{"speaker": "User" if i % 2 == 0 else "Assistant", "text": t} for i, t in enumerate(texts)]
    return {"id": cid, "messages": msgs}


class FakeSource:
    def __init__(self):
        self.files = {
            f"{EVIDENCE_ROOT}/user_evidence/1_evidence/{PERSONA}": {
                "evidence_items": [
                    {"question": "What is my shop called?", "answer": "Aether Pets",
                     "message_evidences": [{"text": "My shop is Aether Pets."}],
                     "conversations": [conv("u1", "My shop is Aether Pets.", "Nice!")]},
                    {"question": "Where do I live?", "answer": "Denver",
                     "message_evidences": [{"text": "I live in Denver."}],
                     "conversations": [conv("u2", "I live in Denver.", "Got it.")]},
                ]
            },
            f"{EVIDENCE_ROOT}/changing_evidence/2_evidence/{PERSONA}": {
                "evidence_items": [
                    {"question": "When does the walker come?", "answer": "9:30 AM",
                     "message_evidences": [{"text": "Walker at 8 AM."}, {"text": "Walker now 9:30 AM."}],
                     "conversations": [conv("c1", "Walker at 8 AM.", "ok"), conv("c2", "Walker now 9:30 AM.", "ok")]},
                ]
            },
            f"{FILLER_ROOT}/{PERSONA}": {
                "evidence_items": [{"conversations": [conv(f"f{i}", f"filler {i}", "ok") for i in range(10)]}]
            },
        }

    def list_files(self, folder):
        return sorted(p.rsplit("/", 1)[1] for p in self.files if p.rsplit("/", 1)[0] == folder)

    def read_json(self, path):
        return self.files[path]


def test_plan_merges_one_history_and_keeps_changing_order():
    plan = build_plan(PERSONA, per_category=2, history_size=8, seed=3,
                      categories=["user_facts", "changing_facts"], now_ms=100 * DAY_MS,
                      source=FakeSource())
    assert {q.category for q in plan.questions} == {"user_facts", "changing_facts"}
    assert len(plan.questions) == 3
    ids = [h.id for h in plan.history]
    assert len(ids) == 8 and len(set(ids)) == 8
    assert {"u1", "u2", "c1", "c2"} <= set(ids)
    assert ids.index("c1") < ids.index("c2")  # the later fact stays later
    times = [h.created_at_ms for h in plan.history]
    assert times == sorted(times) and times[-1] < 100 * DAY_MS
    assert sum(h.is_evidence for h in plan.history) == 4
    assert plan.persona_role == "Test Owner"


def test_plan_is_deterministic_per_seed():
    a = build_plan(PERSONA, history_size=6, seed=1, now_ms=0, source=FakeSource())
    b = build_plan(PERSONA, history_size=6, seed=1, now_ms=0, source=FakeSource())
    assert a.model_dump() == b.model_dump()


def test_import_payload_shape():
    plan = build_plan(PERSONA, history_size=5, seed=0, now_ms=10 * DAY_MS,
                      categories=["user_facts"], source=FakeSource())
    payload = import_payload(plan)
    first = payload["conversations"][0]
    assert first["sourceKey"] == f"convomem:{plan.history[0].id}"
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
    assert first["messages"][1]["createdAt"] > first["messages"][0]["createdAt"]


def test_judge_prompts_are_the_official_ones_filled_in():
    p = judge_prompt(category="changing_facts", evidence_count=2, question="Q?",
                     answer="9:30 AM", response="R", evidence_messages=["a", "b"])
    assert "9:30 AM" in p and "{answer}" not in p and "RIGHT" in p
    rubric = judge_prompt(category="preferences", evidence_count=1, question="Q?",
                          answer="must recommend Rover", response="R", evidence_messages=["x"])
    assert "must recommend Rover" in rubric


def test_answer_matching_helpers():
    assert key_terms("The vendor is Staples Business Advantage.") == [
        "vendor", "staples", "business", "advantage"]
    assert contains_answer("we use staples business advantage", "Staples Business Advantage")
    assert not contains_answer("we use fedex", "Staples Business Advantage")
    assert contains_answer("save $500 each month", "$500")
    # Words the question already contains don't count as evidence of the answer.
    q = "What is our go-to spot for a casual weeknight dinner?"
    gold = "Your go-to spot for a casual weeknight dinner is Siam Spice."
    assert key_terms(gold, q) == ["siam", "spice"]
    assert contains_answer("we love siam spice on tuesdays", gold, question=q)


def test_socket_dir_follows_the_cli_rule(monkeypatch):
    inst = VellumInstance("a", Path("/src"))
    monkeypatch.setattr(VellumInstance, "workspace", property(lambda self: Path("/w/ws")))
    assert inst.ipc_socket_dir == Path("/w/ws")  # short enough: the socket stays in the workspace
    monkeypatch.setattr(VellumInstance, "workspace", property(lambda self: Path("/w/" + "x" * 120)))
    assert inst.ipc_socket_dir.name.startswith("vellum-ipc-")  # too long: hashed tmp dir


def test_env_drops_inherited_instance_vars(monkeypatch):
    monkeypatch.setenv("VELLUM_WORKSPACE_DIR", "/other/workspace")
    monkeypatch.setenv("ASSISTANT_IPC_SOCKET_DIR", "/other/sock")
    inst = VellumInstance("mine", Path("/src"))
    daemon_env = inst._env(for_daemon=True)
    assert "VELLUM_WORKSPACE_DIR" not in daemon_env
    assert "ASSISTANT_IPC_SOCKET_DIR" not in daemon_env
    cli_env = inst._env()
    assert cli_env["VELLUM_WORKSPACE_DIR"].endswith("/assistants/mine/.vellum/workspace")
    assert cli_env["ASSISTANT_IPC_SOCKET_DIR"] != "/other/sock"


def test_reply_text_reads_the_root_span():
    trace = {"spans": [
        {"parent_span_id": "", "attributes": {"mlflow.spanOutputs": json.dumps({"reply": "640 dollars"})}},
        {"parent_span_id": "abc", "attributes": {"mlflow.spanOutputs": json.dumps({"x": 1})}},
    ]}
    assert reply_text(trace) == "640 dollars"
