"""Offline tests for the data quality checker: hand-built items, one problem each."""

from __future__ import annotations

from convomem_arena.data_check.__main__ import report, run_checks
from convomem_arena.data_check.checks import Conv, Msg, parse_raw
from convomem_arena.text import coverage, key_terms

EVIDENCE = "On busy weeknights we always order from Siam Spice by the lake."


def raw_item(
    question="What is our go-to weeknight dinner spot?",
    answer="Your go-to weeknight spot is Siam Spice.",
    evidence=(("User", EVIDENCE),),
    contains=True,
):
    convs = [
        {
            "id": f"c{i}",
            "containsEvidence": contains,
            "messages": [
                {"speaker": "User", "text": "Long week at work."},
                {"speaker": speaker, "text": text},
            ],
        }
        for i, (speaker, text) in enumerate(evidence)
    ]
    return {
        "question": question,
        "answer": answer,
        "message_evidences": [{"speaker": s, "text": t} for s, t in evidence],
        "conversations": convs,
    }


def flags(raw, category="user_facts", n=1, filler=None):
    item = parse_raw(raw, category, n, "p.json", 0)
    (result,) = run_checks([item], {"p.json": filler} if filler is not None else None)
    return {f.check for f in result.flags}


def test_text_helpers_match_whole_words():
    terms = key_terms(
        "Your go-to weeknight spot is Siam Spice.", "What is our go-to weeknight dinner spot?"
    )
    assert terms == ["siam", "spice"]
    assert coverage(["rep"], "a quarterly report") == 0.0
    assert coverage(["rep"], "hire one rep") == 1.0


def test_specifics_are_names_and_numbers_not_advice_words():
    from convomem_arena.text import specifics

    answer = "Certainly. Based on your goals, use GMT for Dynamic Digital; it costs $1,150."
    assert specifics(answer, "Which time zone?") == ["1150", "gmt", "dynamic", "digital"]
    listed = "Here is the plan:\n1. **Listening:** pause first.\n2. Use Slack daily."
    assert specifics(listed) == ["slack"]
    assert specifics("The styles are: 1) a crisp Pilsner, 2) a Saison.") == ["pilsner", "saison"]


def test_reworded_evidence_is_review_not_error():
    raw = raw_item()
    raw["conversations"][0]["messages"][1]["text"] = (
        "So for busy weeknights we always order from Siam Spice by the lake."
    )
    got = flags(raw)
    assert "evidence_reworded_in_conversation" in got
    assert "evidence_not_in_conversations" not in got


def test_clean_item_has_no_flags():
    assert flags(raw_item()) == set()


def test_structural_errors():
    assert "empty_field" in flags(raw_item(answer=" "))
    assert "evidence_count_mismatch" in flags(raw_item(), n=2)
    assert "flagged_conversation_count" in flags(raw_item(contains=False))
    raw = raw_item()
    raw["conversations"][0]["messages"][1]["text"] = "Something else entirely."
    assert "evidence_not_in_conversations" in flags(raw)
    raw = raw_item()
    raw["message_evidences"][0]["speaker"] = "Assistant"
    assert "evidence_speaker_mismatch" in flags(raw)


def test_wrong_speaker_for_category():
    assert "wrong_evidence_speaker" in flags(raw_item(), category="assistant_facts")
    assert "wrong_evidence_speaker" not in flags(raw_item(), category="changing_facts", n=1)


def test_answer_grounding_and_restating():
    assert "answer_name_not_in_evidence" in flags(
        raw_item(answer="Your go-to spot is Gino's Pizzeria.")
    )
    assert "answer_number_not_in_evidence" in flags(raw_item(answer="Siam Spice, about 42 times."))
    assert "answer_restates_question" in flags(raw_item(answer="Our go-to weeknight dinner spot."))


def test_changing_facts_latest_evidence():
    evidence = (
        ("User", "We switched to Siam Spice last month."),
        ("User", "We now always get Gino's Pizzeria."),
    )
    got = flags(
        raw_item(answer="It is Siam Spice.", evidence=evidence), category="changing_facts", n=2
    )
    assert "latest_evidence_not_answer" in got
    good = flags(
        raw_item(answer="It is Gino's Pizzeria.", evidence=evidence), category="changing_facts", n=2
    )
    assert "latest_evidence_not_answer" not in good


def test_filler_contamination():
    leak = [Conv(id="f1", messages=[Msg(speaker="User", text="Siam Spice again tonight!")])]
    assert "answer_terms_in_filler" in flags(raw_item(), filler=leak)
    clean = [Conv(id="f2", messages=[Msg(speaker="User", text="Pizza tonight.")])]
    assert "answer_terms_in_filler" not in flags(raw_item(), filler=clean)
    q = "What was the last name of the freelance writer David we hired?"
    abst = raw_item(question=q, answer="There is no information in prior conversations")
    topic = [
        Conv(
            id="f3",
            messages=[
                Msg(speaker="User", text="We hired David, a freelance writer; last name Okafor.")
            ],
        )
    ]
    assert "question_topic_in_filler" in flags(abst, category="abstention", filler=topic)


def test_duplicates_and_report():
    a = parse_raw(raw_item(), "user_facts", 1, "p.json", 0)
    b = parse_raw(raw_item(), "user_facts", 1, "q.json", 3)
    results = run_checks([a, b])
    assert all(any(f.check == "duplicate_question" for f in r.flags) for r in results)
    text = report(results, set())
    assert "duplicate_question" in text and "Items with no flags: 0/2" in text
