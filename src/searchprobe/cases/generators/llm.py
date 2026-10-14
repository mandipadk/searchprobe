"""LLM-generated cases with typed constraints, validation and answerability checks.

The model must state, for every query, typed constraints (not free-text
hints) and a minimal-pair twin. Outputs are validated against the TestCase
schema (contradictory numeric ranges, empty date windows and duplicate ids are
rejected). Optionally each case is checked for answerability: a reference
search system must return at least one document the judge rates as fully
satisfying, otherwise the case is dropped (an unanswerable query measures
nothing about retrieval).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from searchprobe.cases.models import Constraint, Provenance, TestCase
from searchprobe.cases.twins import attach_twins
from searchprobe.llm import structured_call
from searchprobe.queries.taxonomy import AdversarialCategory, get_category_metadata

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "2"
_CONSTRAINT_ADAPTER: TypeAdapter[Any] = TypeAdapter(Constraint)

_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_NULLABLE_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_STR_LIST = {"type": "array", "items": {"type": "string"}}

CONSTRAINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "type": {
            "type": "string",
            "enum": [
                "topic",
                "exclude",
                "numeric",
                "date",
                "entity",
                "relation",
                "doc_type",
                "language",
                "freeform",
            ],
        },
        "hard": {"type": "boolean"},
        "description": {"type": "string"},
        "subject": _NULLABLE_STR,
        "concept": _NULLABLE_STR,
        "attribute": _NULLABLE_STR,
        "op": {
            "anyOf": [
                {"type": "string", "enum": ["eq", "approx", "lt", "lte", "gt", "gte", "between"]},
                {"type": "null"},
            ]
        },
        "value": _NULLABLE_NUM,
        "value_max": _NULLABLE_NUM,
        "unit": _NULLABLE_STR,
        "field": {"anyOf": [{"type": "string", "enum": ["published", "event"]}, {"type": "null"}]},
        "start": _NULLABLE_STR,
        "end": _NULLABLE_STR,
        "name": _NULLABLE_STR,
        "wrong": _STR_LIST,
        "predicate": _NULLABLE_STR,
        "object": _NULLABLE_STR,
        "allowed": _STR_LIST,
        "disallowed": _STR_LIST,
        "code": _NULLABLE_STR,
        "requirement": _NULLABLE_STR,
    },
    "required": [
        "id",
        "type",
        "hard",
        "description",
        "subject",
        "concept",
        "attribute",
        "op",
        "value",
        "value_max",
        "unit",
        "field",
        "start",
        "end",
        "name",
        "wrong",
        "predicate",
        "object",
        "allowed",
        "disallowed",
        "code",
        "requirement",
    ],
    "additionalProperties": False,
}

CASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["medium", "hard"]},
        "adversarial_reason": {"type": "string"},
        "constraints": {"type": "array", "items": CONSTRAINT_SCHEMA},
        "twin": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "relation": {"type": "string"},
                        "constraints": {"type": "array", "items": CONSTRAINT_SCHEMA},
                    },
                    "required": ["text", "relation", "constraints"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["text", "difficulty", "adversarial_reason", "constraints", "twin"],
    "additionalProperties": False,
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"cases": {"type": "array", "items": CASE_SCHEMA}},
    "required": ["cases"],
    "additionalProperties": False,
}

SYSTEM = """You design verifiable adversarial test cases for evaluating web search engines.

Each case is a realistic query plus typed constraints that define exactly what a correct result is. Use the most specific constraint type available:
- topic: what the result must be about (subject)
- exclude: what the result must NOT be (concept); being X violates, mentioning X does not
- numeric: attribute, op (eq/approx/lt/lte/gt/gte/between), value, value_max, unit
- date: field ("published" = the page's publication date, "event" = when the described thing happened), start/end as YYYY-MM-DD
- entity: name of the intended entity and the confusable ones in "wrong"
- relation: subject, predicate, object (direction matters)
- doc_type: allowed and disallowed document kinds
- language: ISO 639-1 code of the required language
- freeform: only when nothing else fits (requirement)
Fill unused fields with null or empty lists. Number constraint ids c1, c2, ...

Also give a twin: the same query with exactly one logical operator changed (negation removed, number scaled, date shifted, arguments swapped, polarity flipped) and its own constraints, so the correct answers change. Use null if no meaningful twin exists.

Queries must be answerable from the public web and must not depend on today's date (no "last week")."""


def _to_constraint(raw: dict[str, Any]) -> Any | None:
    """Convert a flat LLM constraint into a typed constraint; None if invalid."""
    data = {k: v for k, v in raw.items() if v not in (None, [], "")}
    data.setdefault("hard", raw.get("hard", True))
    for key in ("start", "end"):
        if key in data:
            try:
                data[key] = date.fromisoformat(str(data[key])[:10])
            except ValueError:
                return None
    try:
        return _CONSTRAINT_ADAPTER.validate_python(data)
    except ValidationError:
        return None


def build_cases(
    items: list[dict[str, Any]], category: AdversarialCategory, model: str
) -> tuple[list[TestCase], list[str]]:
    """Validate generated items into TestCases (with twins); returns (cases, rejections)."""
    from searchprobe.cases.convert import CATEGORY_OPERATORS

    cases: list[TestCase] = []
    rejected: list[str] = []
    prov = Provenance(
        generator="llm",
        version=GENERATOR_VERSION,
        source=model,
        created=datetime.now(UTC).date().isoformat(),
    )
    for item in items:
        constraints = [c for c in (_to_constraint(r) for r in item.get("constraints", [])) if c]
        if len(constraints) != len(item.get("constraints", [])):
            rejected.append(f"{item.get('text')!r}: invalid constraint(s)")
            continue
        try:
            case = TestCase(
                text=item["text"].strip(),
                category=category,
                operators=CATEGORY_OPERATORS.get(category.value, []),
                difficulty=item.get("difficulty", "medium"),
                constraints=constraints,
                provenance=prov,
                validation="auto",
                adversarial_reason=item.get("adversarial_reason"),
            )
        except ValidationError as e:
            rejected.append(f"{item.get('text')!r}: {e.errors()[0]['msg']}")
            continue
        if not case.is_verifiable:
            rejected.append(f"{case.text!r}: only free-form constraints")
            continue
        twin_raw = item.get("twin")
        if twin_raw:
            twin_constraints = [_to_constraint(r) for r in twin_raw.get("constraints", [])]
            if twin_constraints and all(twin_constraints):
                try:
                    twin = TestCase(
                        text=twin_raw["text"].strip(),
                        category=category,
                        operators=case.operators,
                        constraints=twin_constraints,
                        twin_of=case.id,
                        twin_relation=twin_raw.get("relation") or "llm_twin",
                        provenance=prov,
                        validation="auto",
                    )
                    case = case.model_copy(
                        update={"twins": {twin.twin_relation: twin.id}}  # type: ignore[dict-item]
                    )
                    cases.append(twin)
                except ValidationError:
                    pass
        cases.append(case)
    # Rule-based twins for cases the model did not twin.
    untwinned = [c for c in cases if not c.twins and not c.twin_of]
    twinned = attach_twins(untwinned)
    by_id = {c.id: c for c in cases}
    for c in twinned:
        by_id[c.id] = c
    return list(by_id.values()), rejected


async def generate_llm_cases(
    client: Any,
    model: str,
    category: AdversarialCategory,
    count: int = 10,
    avoid: list[str] | None = None,
    effort: str | None = "high",
) -> tuple[list[TestCase], list[str]]:
    """Ask the model for ``count`` cases in ``category``; returns (cases, rejections)."""
    meta = get_category_metadata(category)
    avoid_text = "\n".join(f"- {t}" for t in (avoid or [])[:30])
    prompt = f"""Category: {meta.display_name}
What it tests: {meta.description}
Why embedding search fails here: {meta.failure_hypothesis}
Examples (do not repeat): {"; ".join(meta.example_queries)}
{f"Already in the dataset (do not repeat):{chr(10)}{avoid_text}" if avoid_text else ""}

Generate {count} new, diverse cases (different domains: geography, science, business, culture, technology, sports, history)."""
    result = await structured_call(
        client, model=model, system=SYSTEM, prompt=prompt, schema=SCHEMA, effort=effort
    )
    return build_cases(result.data.get("cases", []), category, result.model)


async def check_answerable(case: TestCase, provider: Any, judge: Any, fetcher: Any, k: int = 5):
    """Return (answerable, url) using a reference provider + the document judge."""
    from searchprobe.judging.checkers import run_checkers
    from searchprobe.judging.llm_judge import evidence_from
    from searchprobe.providers.models import SearchRequest

    response = await provider.search(SearchRequest(query=case.text, num_results=k))
    if response.error or not response.results:
        return False, None
    urls = [str(r.url) for r in response.results]
    snaps = await fetcher.fetch_many(urls) if fetcher else {}
    for r in response.results:
        url = str(r.url)
        ev = evidence_from(url, snaps.get(url), r.title, r.content or r.snippet)
        verdicts = run_checkers(case, url, snaps.get(url))
        judgment = await judge.judge(case, ev, decided=verdicts)
        if judgment.is_fully_satisfying(case):
            return True, url
    return False, None
