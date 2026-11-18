"""Sample ConvoMem items and generate the perturbed inputs for each one.

Perturbations come from a generator model (default: local Ollama) and are cached, so
every judge is compared on identical inputs. Each one is then checked
deterministically (present, actually different, numbers kept in paraphrases, sane
length); a field that fails is recorded with its problems and its variant skipped.
`review.md` lists every original next to its perturbations for a human spot check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from convomem_arena.data import CATEGORIES, EvidenceItem, EvidenceStream, HubSource
from convomem_arena.judge_check.variants import SEES, judge_kind
from convomem_arena.judge_client import ChatFn, JudgeEndpoint, chat


class Case(BaseModel):
    """One ConvoMem item, the response the baseline grades, and its perturbations."""

    id: str
    category: str
    evidence_count: int
    kind: str
    question: str
    reference: str
    """The gold answer (a rubric for preferences / implicit connections)."""
    evidence: list[str]
    response: str = ""
    """The baseline response: the gold answer itself, or a generated reply for rubrics."""
    perturbations: dict[str, str | list[str]] = Field(default_factory=dict)
    problems: dict[str, list[str]] = Field(default_factory=dict)
    generator: str = ""

    def usable(self, field: str) -> bool:
        return field in self.perturbations and not self.problems.get(field)


def case_id(item: EvidenceItem) -> str:
    persona = item.persona_file.removesuffix(".json")
    return f"{item.category}/{item.evidence_count}/{persona}/{item.item_index}"


def sample_cases(per_category: int, seed: int, source: HubSource | None = None) -> list[Case]:
    stream = EvidenceStream(source or HubSource(), seed)
    cases = []
    for category in CATEGORIES:
        for i, item in enumerate(stream.category_items(category, None)):
            if i >= per_category:
                break
            cases.append(
                Case(
                    id=case_id(item),
                    category=category,
                    evidence_count=item.evidence_count,
                    kind=judge_kind(category, item.evidence_count),
                    question=item.question,
                    reference=item.answer,
                    evidence=list(item.evidence_messages),
                )
            )
    return cases


# --------------------------------------------------------------- generation

FIELD_INSTRUCTIONS = {
    "question_paraphrase": (
        "The same question reworded. Same meaning, same names and numbers, "
        "different wording. A correct answer to the original must still answer it."
    ),
    "question_negated": (
        "The question reversed or negated (for example asking what the user did NOT "
        "do, or asking for the opposite), so the original correct answer no longer "
        "answers it."
    ),
    "evidence_paraphrase": (
        "A list with one entry per evidence message, each reworded in the same voice "
        "and keeping every fact, name and number."
    ),
    "evidence_changed": (
        "A list with one entry per evidence message, each rewritten in the same voice "
        "so its key fact is different (another name, number, place or choice) and it "
        "contradicts the reference answer."
    ),
    "reference_changed": (
        "A replacement reference answer in the same style that states a different, "
        "incompatible fact. If the reference is a grading rubric, write a rubric that "
        "demands the opposite preference or behaviour."
    ),
    "response_wrong": (
        "A fluent, confident assistant reply to the question that is wrong: it gives a "
        "different, incorrect fact (or, for a rubric, plainly ignores what the rubric "
        "demands). If the question is unanswerable, confidently invent a specific answer."
    ),
    "response_correct": (
        "A natural assistant reply to the question that fully satisfies every "
        "criterion in the rubric."
    ),
}


def fields_for(kind: str) -> list[str]:
    """The perturbations a judge of this kind can be tested with."""
    sees = SEES[kind]
    fields = ["question_paraphrase", "question_negated", "response_wrong"]
    if "evidence" in sees:
        fields += ["evidence_paraphrase", "evidence_changed"]
    if "reference" in sees:
        fields.append("reference_changed")
    if kind == "rubric":
        fields.append("response_correct")
    return fields


def generator_prompt(case: Case) -> str:
    fields = fields_for(case.kind)
    evidence = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(case.evidence)) or "(none)"
    wanted = "\n".join(f'- "{f}": {FIELD_INSTRUCTIONS[f]}' for f in fields)
    return f"""You are helping test the robustness of an answer-grading model. Given a \
question, the evidence messages it is based on, and its reference answer, write \
controlled variations.

Category: {case.category}
Question: {case.question}
Evidence messages:
{evidence}
Reference answer: {case.reference}

Return ONLY a JSON object with exactly these keys:
{wanted}

Keep the original's register and length. Output the JSON object and nothing else."""


def extract_json(text: str) -> dict:
    """The first JSON object in a model reply (fences and chatter tolerated)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise TypeError("reply JSON is not an object")
    return obj


# --------------------------------------------------------------- validation


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)*", text))


def _original(case: Case, field: str) -> str | list[str] | None:
    if field.startswith("question"):
        return case.question
    if field.startswith("evidence"):
        return case.evidence
    if field == "reference_changed":
        return case.reference
    if field == "response_wrong":
        return case.response or case.reference
    return None


def validate(case: Case, field: str, value: object) -> list[str]:
    """Deterministic sanity checks; an empty list means the perturbation is usable."""
    original = _original(case, field)
    if field.startswith("evidence"):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return ["not a list of strings"]
        if len(value) != len(case.evidence):
            return [f"{len(value)} messages, expected {len(case.evidence)}"]
        pairs = list(zip(case.evidence, value))
    else:
        if not isinstance(value, str):
            return ["not a string"]
        pairs = [(original, value)] if isinstance(original, str) else [("", value)]
    problems = []
    for before, after in pairs:
        if not after.strip():
            problems.append("empty")
        elif before and _norm(before) == _norm(after):
            problems.append("identical to the original")
        elif before and field.endswith("_paraphrase"):
            if _numbers(before) != _numbers(after):
                problems.append("numbers changed in a paraphrase")
            ratio = len(after) / max(len(before), 1)
            if not 0.4 <= ratio <= 2.5:
                problems.append(f"length ratio {ratio:.1f}")
    return sorted(set(problems))


def perturb(
    case: Case, generator: JudgeEndpoint, chat_fn: ChatFn = chat, attempts: int = 2
) -> Case:
    """Fill `case.perturbations` (and the rubric baseline response) from the generator."""
    if case.kind != "rubric":
        case.response = case.reference
    fields = fields_for(case.kind)
    data: dict = {}
    error = ""
    for _ in range(attempts):
        try:
            data = extract_json(chat_fn(generator, generator_prompt(case)))
            break
        except Exception as err:  # noqa: BLE001 - retried, then recorded
            error = f"{type(err).__name__}: {err}"
    case.generator = generator.model
    for field in fields:
        if field not in data:
            case.problems[field] = [f"missing from generator output{': ' + error if error else ''}"]
            continue
        value = data[field]
        if field == "response_correct":
            if isinstance(value, str) and value.strip():
                case.response = value.strip()
            else:
                case.problems[field] = ["empty"]
            continue
        case.perturbations[field] = value
        problems = validate(case, field, value)
        if problems:
            case.problems[field] = problems
    if case.kind == "rubric" and not case.response:
        case.problems.setdefault("response_correct", ["no baseline response"])
    return case


def review_markdown(cases: Iterable[Case]) -> str:
    """Every original next to its perturbations, for a human to spot-check."""
    out = ["# Judge robustness checker: perturbations to review", ""]
    for c in cases:
        out += [f"## {c.id} ({c.kind})", "", f"- **question:** {c.question}"]
        for i, e in enumerate(c.evidence):
            out.append(f"- **evidence {i + 1}:** {e}")
        out += [f"- **reference:** {c.reference}", f"- **baseline response:** {c.response}", ""]
        for field, value in c.perturbations.items():
            flag = f" ⚠ {', '.join(c.problems[field])}" if c.problems.get(field) else ""
            if isinstance(value, list):
                out.append(f"- **{field}**{flag}:")
                out += [f"  - {v}" for v in value]
            else:
                out.append(f"- **{field}**{flag}: {value}")
        for field, problems in c.problems.items():
            if field not in c.perturbations:
                out.append(f"- **{field}** ⚠ {', '.join(problems)}")
        out.append("")
    return "\n".join(out)
