"""Validity-preserving mutations over test cases.

Mutations act on whole ``TestCase`` objects, not raw strings. Rephrasing
mutations change how an operator is *expressed* ("NOT members of X" ->
"other than members of X") while keeping the constraints and the answer set,
so a mutated case is exactly as verifiable as its parent. LLM mutations can
also rewrite the case (rarer entities, an added constraint), and are
re-validated against the schema.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

from searchprobe.cases.models import TestCase

MutationFn = Callable[[TestCase, random.Random], TestCase | None]

# Equivalent surface forms of each operator: (pattern, replacements).
NEGATION_FORMS: list[tuple[str, list[str]]] = [
    (
        r"\bthat are not\b",
        [
            "other than those that are",
            "excluding those that are",
            "that aren't",
            "apart from those that are",
        ],
    ),
    (r"\bwithout\b", ["lacking", "with no", "that do not have"]),
    (r"\bthat do not\b", ["that don't", "which do not", "other than those that"]),
    (r"\bbut not of\b", ["and not of", "while not being members of", "excluding members of"]),
]
NUMERIC_FORMS: list[tuple[str, list[str]]] = [
    (
        r"\bbetween ([\d.,]+(?: million)?) and ([\d.,]+(?: million)?)\b",
        [r"from \1 to \2", r"of at least \1 but no more than \2", r"in the \1 to \2 range"],
    ),
    (r"\bmore than\b", ["over", "upwards of"]),
]
TEMPORAL_FORMS: list[tuple[str, list[str]]] = [
    (r"\bin (\d{4})\b", [r"during \1", r"for the year \1", r"(\1)"]),
    (r"\bpublished in\b", ["from", "dated", "released in"]),
]

FORMS_BY_OPERATOR: dict[str, list[tuple[str, list[str]]]] = {
    "negation": NEGATION_FORMS,
    "boolean": NEGATION_FORMS,
    "numeric": NUMERIC_FORMS,
    "temporal": TEMPORAL_FORMS,
    "conjunction": NUMERIC_FORMS + NEGATION_FORMS,
}


def _child(parent: TestCase, text: str, mutation: str) -> TestCase:
    """A new case with the parent's constraints/answers and a new query text."""
    history = (parent.provenance.notes or "").strip()
    data = parent.model_dump()
    data.update(
        id="",  # recomputed from the new text
        text=text,
        twins={},
        twin_of=None,
        twin_relation=None,
        validation="auto",
    )
    data["provenance"].update(
        generator="discovered",
        notes=f"{history} | {mutation}" if history else mutation,
    )
    return TestCase.model_validate(data)


def rephrase_operator(case: TestCase, rng: random.Random) -> TestCase | None:
    """Express the case's operator differently; constraints and answers are unchanged."""
    forms = []
    for op in case.operators:
        forms.extend(FORMS_BY_OPERATOR.get(op, []))
    rng.shuffle(forms)
    for pattern, replacements in forms:
        if re.search(pattern, case.text, flags=re.IGNORECASE):
            repl = rng.choice(replacements)
            text = re.sub(pattern, repl, case.text, count=1, flags=re.IGNORECASE)
            text = re.sub(r"\s{2,}", " ", text).strip()
            if text.lower() != case.text.lower():
                return _child(case, text, f"rephrase_operator({pattern!r} -> {repl!r})")
    return None


def add_distractor_context(case: TestCase, rng: random.Random) -> TestCase | None:
    """Add topical context that pulls toward the violating answers (hard for embeddings)."""
    if not case.negative_set or case.category.value == "entity_disambiguation":
        return None  # namesakes share the label, so the distractor would be ambiguous
    distractor = rng.choice(case.negative_set).label
    templates = [
        f"{case.text} (unlike {distractor})",
        f"{case.text}, so not including {distractor}",
        f"{case.text} — {distractor} does not count",
    ]
    return _child(case, rng.choice(templates), f"add_distractor_context({distractor})")


def question_form(case: TestCase, rng: random.Random) -> TestCase | None:
    """Turn a keyword-style query into a natural question."""
    if case.text.endswith("?") or case.category.value == "entity_disambiguation":
        return None
    first_words = case.text.split()[:2]
    # Only plural noun phrases ("Countries that...", "Capital cities in...") read naturally.
    if not any(w.lower().endswith("s") for w in first_words) or case.text.lower().startswith(
        ("history", "news", "an introduction")
    ):
        return None
    lead = rng.choice(["Which are the", "What are the", "Can you list the", "List all"])
    text = f"{lead} {case.text[0].lower()}{case.text[1:]}"
    if not lead.startswith("List"):
        text += "?"
    return _child(case, text, "question_form")


RULE_MUTATIONS: dict[str, MutationFn] = {
    "rephrase_operator": rephrase_operator,
    "add_distractor_context": add_distractor_context,
    "question_form": question_form,
}


LLM_MUTATION_SYSTEM = (
    "You rewrite a search test case so that it is harder for embedding-based search while "
    "staying natural, answerable from the public web, and exactly as constrained as before. "
    "Keep the same meaning and the same set of correct answers. Express the logical operator "
    "less explicitly (e.g. implicit negation, paraphrased numbers), or reorder clauses. Return "
    "one rewritten query."
)
LLM_MUTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "change": {"type": "string"}},
    "required": ["query", "change"],
    "additionalProperties": False,
}


async def llm_rephrase(case: TestCase, client: Any, model: str) -> tuple[TestCase | None, float]:
    """LLM rewrite that must keep constraints (and therefore answers) unchanged.

    Returns the child (None if the rewrite is unusable) and the call's cost.
    """
    from searchprobe.llm import structured_call

    constraints = "\n".join(f"- {c.describe()}" for c in case.constraints)
    result = await structured_call(
        client,
        model=model,
        system=LLM_MUTATION_SYSTEM,
        prompt=f'Query: "{case.text}"\nConstraints (must all still hold):\n{constraints}',
        schema=LLM_MUTATION_SCHEMA,
        effort="low",
    )
    text = (result.data.get("query") or "").strip().strip('"')
    if not text or text.lower() == case.text.lower():
        return None, result.cost_usd
    change = result.data.get("change", "")[:80]
    return _child(case, text, f"llm_rephrase({change})"), result.cost_usd
