import random

import pytest

from convomem_arena.data import (
    CATEGORIES,
    Conversation,
    EvidenceStream,
    HubSource,
    Message,
    format_context,
    mix,
    normalize_speaker,
    parse_evidence_file,
)
from convomem_arena.judges import (
    AbstentionJudge,
    ConvoMemJudge,
    RubricAnswerJudge,
    UserFactsMultiJudge,
    UserFactsSingleJudge,
    format_evidence,
    judge_for,
    parse_right_wrong,
)
from convomem_arena.servers.memory import MemoryIndex
from convomem_arena.taskset import ConvoMemConfig, ConvoMemTaskConfig, ConvoMemTaskset


def conv(cid: str, *texts: str) -> Conversation:
    speakers = ["User", "Assistant"]
    return Conversation(
        id=cid,
        messages=[Message(speaker=speakers[i % 2], text=t) for i, t in enumerate(texts)],
    )


def raw_conv(cid: str, text: str) -> dict:
    return {
        "id": cid,
        "containsEvidence": True,
        "messages": [{"speaker": "user", "text": text}, {"speaker": "assistant", "text": "ok"}],
    }


class FakeSource(HubSource):
    """Two personas' worth of evidence and filler, served from memory."""

    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, dict] = {}
        for category, (folder, counts) in CATEGORIES.items():
            for n in counts:
                for persona in ("p1_Analyst.json", "p2_Founder.json"):
                    items = [
                        {
                            "question": f"{category} q{i} {persona}",
                            "answer": f"a{i}",
                            "message_evidences": [
                                {"speaker": "User", "text": f"fact {k}"} for k in range(n)
                            ],
                            "conversations": [
                                raw_conv(f"{persona}-{category}-{n}-{i}-{k}", f"fact {k}")
                                for k in range(n)
                            ],
                        }
                        for i in range(3)
                    ]
                    path = f"core_benchmark/evidence_questions/{folder}/{n}_evidence/{persona}"
                    self.files[path] = {"evidence_items": items}
        for persona in ("p1_Analyst.json", "p2_Founder.json"):
            self.files[f"core_benchmark/filler_conversations/{persona}"] = {
                "evidence_items": [
                    {
                        "question": "t",
                        "answer": "d",
                        "conversations": [raw_conv(f"{persona}-f{j}", f"filler {j}")],
                    }
                    for j in range(50)
                ]
            }

    def list_files(self, folder: str) -> list[str]:
        prefix = folder + "/"
        return sorted(p[len(prefix) :] for p in self.files if p.startswith(prefix))

    def read_json(self, path: str) -> dict:
        return self.files[path]


@pytest.fixture
def source(monkeypatch) -> FakeSource:
    fake = FakeSource()
    monkeypatch.setattr("convomem_arena.taskset.HubSource", lambda *a, **k: fake)
    return fake


def test_normalize_speaker():
    assert normalize_speaker("user") == "User"
    assert normalize_speaker("Assistant ") == "Assistant"


def test_mix_keeps_all_evidence_in_order():
    evidence = [conv(f"e{i}", f"evidence {i}") for i in range(3)]
    filler = [conv(f"f{i}", f"filler {i}") for i in range(40)]
    mixed, positions = mix(evidence, filler, 20, random.Random(0))
    assert len(mixed) == 20
    assert [mixed[p].id for p in positions] == ["e0", "e1", "e2"]
    assert positions == sorted(positions)
    assert len({c.id for c in mixed}) == 20


def test_mix_is_deterministic_and_never_drops_evidence():
    evidence = [conv(f"e{i}", "x") for i in range(4)]
    filler = [conv(f"f{i}", "y") for i in range(10)]
    first = mix(evidence, filler, 8, random.Random("s"))
    assert first == mix(evidence, filler, 8, random.Random("s"))
    small, positions = mix(evidence, filler, 1, random.Random(0))
    assert [c.id for c in small] == ["e0", "e1", "e2", "e3"] and positions == [0, 1, 2, 3]


def test_format_context_matches_reference_layout():
    text = format_context([conv("a", "hi", "hello"), conv("b", "bye")])
    assert text == "Conversation 1:\nUser: hi\nAssistant: hello\n\nConversation 2:\nUser: bye"


def test_parse_evidence_file_skips_empty_items():
    payload = {
        "evidence_items": [
            {"question": "q", "answer": "a", "message_evidences": [], "conversations": []},
            {
                "question": "q2",
                "answer": "a2",
                "message_evidences": [{"speaker": "User", "text": "e"}],
                "conversations": [raw_conv("c", "e")],
            },
        ]
    }
    items = parse_evidence_file(payload, "user_facts", 1, "p.json")
    assert len(items) == 1 and items[0].item_index == 1 and items[0].evidence_messages == ["e"]
    assert items[0].conversations[0].messages[0].speaker == "User"


@pytest.mark.parametrize(
    ("text", "verdict"),
    [("RIGHT", True), ("wrong", False), ("Answer: RIGHT.", True), ("RIGHT or WRONG", False)],
)
def test_parse_right_wrong(text, verdict):
    assert parse_right_wrong(text) is verdict


def test_parse_right_wrong_raises_on_no_verdict():
    with pytest.raises(ValueError):
        parse_right_wrong("I am not sure")


def test_judge_for_matches_reference_evaluations():
    import verifiers.v1 as vf

    config = vf.JudgeConfig()
    assert type(judge_for("user_facts", 1, config)) is UserFactsSingleJudge
    assert type(judge_for("user_facts", 3, config)) is UserFactsMultiJudge
    assert type(judge_for("abstention", 2, config)) is AbstentionJudge
    assert type(judge_for("preferences", 1, config)) is RubricAnswerJudge
    assert type(judge_for("implicit_connections", 2, config)) is RubricAnswerJudge
    assert type(judge_for("changing_facts", 2, config)) is ConvoMemJudge
    assert type(judge_for("assistant_facts", 1, config)) is ConvoMemJudge


def test_user_facts_judge_prompt_includes_evidence():
    judge = UserFactsMultiJudge()
    evidence = format_evidence("user_facts", 2, ["I own 40 NVDA", "bought 20 more"])
    prompt = judge.build_messages(
        question="How many?", answer="60", response="60 shares", evidence=evidence
    )
    assert "Evidence Message 1: I own 40 NVDA\nEvidence Message 2: bought 20 more" in prompt
    assert "{" not in prompt.split("**Evaluation Criteria:**")[0]


def test_memory_index_search_read_list():
    index = MemoryIndex(
        [
            conv("a", "Let's plan the quarterly offsite agenda", "Sure"),
            conv("b", "My favorite spreadsheet color for hot leads is green", "Noted"),
            conv("c", "Weather is nice", "Indeed"),
        ]
    )
    hits = index.search("what color do I use for hot leads", 3)
    assert hits[0]["conversation"] == 2 and hits[0]["message"] == 1
    assert index.search("the and of", 3) == []
    read = index.read(2, start=2, count=5)
    assert read["messages"] == [{"message": 2, "speaker": "Assistant", "text": "Noted"}]
    listing = index.list(offset=1, limit=1)
    assert listing["total_conversations"] == 3
    assert listing["conversations"][0]["conversation"] == 2
    with pytest.raises(ValueError):
        index.read(4, 1, 1)


def test_stream_round_robins_evidence_counts():
    stream = EvidenceStream(FakeSource(), seed=0)
    counts = [
        item.evidence_count for item in list(stream.category_items("changing_facts", None))[:5]
    ]
    assert counts == [2, 3, 4, 5, 6]
    filtered = {i.evidence_count for i in stream.category_items("user_facts", [1, 6])}
    assert filtered == {1, 6}


def test_taskset_interleaves_categories_and_builds_prompts(source):
    config = ConvoMemConfig(context_sizes=[1, 10])
    tasks = list(ConvoMemTaskset(config).take(12))
    assert [t.data.category for t in tasks[::2]] == list(CATEGORIES)
    first = tasks[1].data
    assert first.context_size == 10 and first.num_conversations == 10
    assert first.conversations == []
    assert first.prompt.startswith("Answer the question based on the conversations below.")
    assert first.prompt.count("Conversation ") == 10
    assert first.prompt.rstrip().endswith(f"Question: {first.question}\n\nAnswer:")
    assert len({t.key for t in tasks}) == 12


def test_taskset_tools_mode_keeps_history_out_of_prompt(source):
    config = ConvoMemConfig(
        categories=["user_facts"], context_sizes=[12], task=ConvoMemTaskConfig(memory_mode="tools")
    )
    task = next(iter(ConvoMemTaskset(config)))
    data = task.data
    assert len(data.conversations) == 12
    assert "fact 0" not in data.prompt and "memory tools" in data.prompt
    assert data.conversations[data.evidence_positions[0]].messages[0].text == "fact 0"
    assert [type(s).__name__ for s in type(task).toolsets(task.config)] == ["MemoryToolset"]


def test_taskset_is_reproducible(source):
    def ids(seed):
        return [
            (t.data.id, t.data.evidence_positions)
            for t in ConvoMemTaskset(ConvoMemConfig(seed=seed)).take(9)
        ]

    assert ids(0) == ids(0)
    assert ids(0) != ids(1)
