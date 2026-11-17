"""Offline tests for the judge robustness checker (fake generator and judges, no network)."""

from __future__ import annotations

import json

import pytest

from convomem_arena.judge_check.perturb import Case, extract_json, fields_for, perturb, validate
from convomem_arena.judge_check.run import (
    BASELINE,
    judge_cases,
    planned_variants,
    render,
    report,
    score,
)
from convomem_arena.judge_check.variants import expectation, judge_kind
from convomem_arena.judge_client import JudgeEndpoint

MARK = "ZZWRONG"


def make_case(category: str, evidence_count: int = 1) -> Case:
    return Case(
        id=f"{category}/{evidence_count}/p/0",
        category=category,
        evidence_count=evidence_count,
        kind=judge_kind(category, evidence_count),
        question="What city did I move to in 2024?",
        reference="You moved to Oakland in 2024.",
        evidence=["I moved to Oakland in 2024 for the workshop space."] * evidence_count,
    )


def fake_generator(endpoint, prompt):
    """Paraphrases keep meaning; every meaning-changing field carries MARK."""
    n = (
        prompt.count("\n1. ")
        and prompt.split("Evidence messages:\n")[1].split("\nReference")[0].count("\n") + 1
    )
    return (
        "```json\n"
        + json.dumps(
            {
                "question_paraphrase": "In 2024, which city did I relocate to?",
                "question_negated": f"Which city did I NOT move to in 2024? {MARK}",
                "evidence_paraphrase": ["In 2024 I relocated to Oakland to get workshop space."]
                * n,
                "evidence_changed": [f"I moved to Denver in 2024. {MARK}"] * n,
                "reference_changed": f"You moved to Denver in 2024. {MARK}",
                "response_wrong": f"You moved to Denver. {MARK}",
                "response_correct": "Good idea: try Arrival, it matches your taste.",
            }
        )
        + "\n```"
    )


def naive_judge(endpoint, prompt):
    """Flips on every meaning change, including where it should not (abstention)."""
    return "WRONG" if MARK in prompt else "RIGHT"


def robust_judge(endpoint, prompt):
    """Flips on meaning changes, but an abstention judge only checks that the reply
    declines, so a negated (still unanswerable) question keeps it RIGHT."""
    if "ABSTENTION test" in prompt:
        response = prompt.split("Model Response:")[1]
        return "WRONG" if MARK in response else "RIGHT"
    return naive_judge(endpoint, prompt)


def gullible_judge(endpoint, prompt):
    return "RIGHT"


# ------------------------------------------------------------- expectations


def test_expectations_follow_what_each_judge_sees():
    assert expectation("question_paraphrase", "default") == "same"
    assert expectation("question_negated", "default") == "flip"
    assert expectation("question_negated", "abstention") == "same"
    assert expectation("evidence_paraphrase", "user_facts_single") == "same"
    assert expectation("evidence_changed", "user_facts_multi") == "flip"
    # inputs a judge never sees are not tested
    assert expectation("evidence_paraphrase", "default") is None
    assert expectation("evidence_changed", "rubric") is None
    assert expectation("reference_changed", "abstention") is None


def test_fields_for_kind():
    assert "evidence_changed" in fields_for("user_facts_single")
    assert "evidence_changed" not in fields_for("default")
    assert "reference_changed" not in fields_for("abstention")
    assert "response_correct" in fields_for("rubric")


# --------------------------------------------------------- generation checks


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_flags_bad_perturbations():
    c = make_case("user_facts")
    assert validate(c, "question_paraphrase", "What city did I move to in 2024?") == [
        "identical to the original"
    ]
    assert "numbers changed in a paraphrase" in validate(
        c, "question_paraphrase", "Which city did I relocate to in 2023?"
    )
    assert validate(c, "evidence_paraphrase", ["a", "b"]) == ["2 messages, expected 1"]
    assert validate(c, "question_paraphrase", "Which city did I relocate to in 2024?") == []


def test_perturb_fills_fields_and_baseline_response():
    gen = JudgeEndpoint(model="fake")
    c = perturb(make_case("user_facts", 2), gen, chat_fn=fake_generator)
    assert c.response == c.reference
    assert len(c.perturbations["evidence_paraphrase"]) == 2
    assert not c.problems
    r = perturb(make_case("preferences"), gen, chat_fn=fake_generator)
    assert r.response.startswith("Good idea")
    assert "response_correct" not in r.perturbations


# ------------------------------------------------------------- render / plan


def test_render_swaps_exactly_one_input():
    c = perturb(make_case("user_facts"), JudgeEndpoint(model="fake"), chat_fn=fake_generator)
    base = render(c, BASELINE)
    para = render(c, "question_paraphrase")
    assert c.question in base and c.question not in para
    assert "relocate" in para
    assert "Denver" in render(c, "evidence_changed")


def test_planned_variants_skip_unseen_inputs():
    gen = JudgeEndpoint(model="fake")
    default = perturb(make_case("assistant_facts"), gen, chat_fn=fake_generator)
    assert "evidence_paraphrase" not in planned_variants(default)
    abst = perturb(make_case("abstention"), gen, chat_fn=fake_generator)
    assert set(planned_variants(abst)) == {
        "question_paraphrase",
        "question_negated",
        "response_wrong",
    }
    uf = perturb(make_case("user_facts"), gen, chat_fn=fake_generator)
    assert set(planned_variants(uf)) == {
        "question_paraphrase",
        "question_negated",
        "evidence_paraphrase",
        "evidence_changed",
        "reference_changed",
        "response_wrong",
    }


# ------------------------------------------------------------ judge + score


def _run(tmp_path, judge_fn, categories=("user_facts", "assistant_facts", "abstention")):
    gen = JudgeEndpoint(model="fake")
    cases = [perturb(make_case(cat), gen, chat_fn=fake_generator) for cat in categories]
    path = tmp_path / "verdicts.jsonl"
    judge_cases(cases, [JudgeEndpoint(model="j")], path, repeats=2, workers=2, chat_fn=judge_fn)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return cases, path, rows, score(rows)


def test_robust_judge_passes_every_check(tmp_path):
    _, _, _, results = _run(tmp_path, robust_judge)
    decided = [r for r in results if r["outcome"] in ("pass", "fail")]
    assert decided and all(r["outcome"] == "pass" for r in decided)
    # abstention's negated question must stay RIGHT, and the robust judge keeps it
    abst = next(
        r for r in results if r["kind"] == "abstention" and r["variant"] == "question_negated"
    )
    assert abst["expect"] == "same"


def test_abstention_negation_marked_wrong_is_a_failure(tmp_path):
    _, _, _, results = _run(tmp_path, naive_judge, categories=("abstention",))
    r = next(r for r in results if r["variant"] == "question_negated")
    assert r["outcome"] == "fail"  # the naive judge says WRONG; the verdict should stay


def test_gullible_judge_fails_every_flip(tmp_path):
    _, _, _, results = _run(tmp_path, gullible_judge)
    flips = [r for r in results if r.get("expect") == "flip"]
    assert flips and all(r["outcome"] == "fail" for r in flips)
    text = report(results, [])
    assert "sensitivity (flip)" in text and "0% (" in text


def test_flip_is_inconclusive_when_baseline_not_right(tmp_path):
    _, _, _, results = _run(tmp_path, lambda e, p: "WRONG")
    flips = [r for r in results if r.get("expect") == "flip"]
    assert flips and all(r["outcome"] == "inconclusive" for r in flips)


def test_unparseable_and_unstable_verdicts_are_reported(tmp_path):
    calls = {"n": 0}

    def flaky(endpoint, prompt):
        calls["n"] += 1
        return "maybe" if calls["n"] % 3 == 0 else robust_judge(endpoint, prompt)

    _, _, rows, results = _run(tmp_path, flaky)
    assert any(r["error"] for r in rows)
    assert any(not r["stable"] for r in results)


def test_judging_resumes_without_repeating_calls(tmp_path):
    cases, path, rows, _ = _run(tmp_path, robust_judge)
    again = judge_cases(cases, [JudgeEndpoint(model="j")], path, repeats=2, chat_fn=robust_judge)
    assert again == 0
    assert len(path.read_text().splitlines()) == len(rows)
