"""Convert v1 queries (``Query`` + ``GroundTruth``) into v2 ``TestCase`` objects."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from searchprobe.cases.models import (
    DateConstraint,
    DocTypeConstraint,
    EntityConstraint,
    ExcludeConstraint,
    FreeformConstraint,
    LanguageConstraint,
    NumericConstraint,
    Provenance,
    TestCase,
    TopicConstraint,
)
from searchprobe.queries.taxonomy import AdversarialCategory

# Logical operator(s) each category primarily exercises.
CATEGORY_OPERATORS: dict[str, list[str]] = {
    "negation": ["negation"],
    "numeric_precision": ["numeric"],
    "temporal_constraint": ["temporal"],
    "multi_constraint": ["conjunction"],
    "polysemy": ["disambiguation"],
    "compositional": ["relation"],
    "antonym_confusion": ["polarity"],
    "specificity_gradient": ["specificity"],
    "cross_lingual": ["language"],
    "counterfactual": ["modality"],
    "boolean_logic": ["boolean"],
    "entity_disambiguation": ["disambiguation"],
    "instruction_following": ["doc_type"],
}


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def ground_truth_to_constraints(gt: dict[str, Any] | None) -> list[Any]:
    """Map a v1 ``GroundTruth`` dict onto typed v2 constraints."""
    if not gt:
        return []
    constraints: list[Any] = []

    def next_id() -> str:
        return f"c{len(constraints) + 1}"

    if gt.get("must_contain_keywords"):
        terms = gt["must_contain_keywords"]
        constraints.append(
            TopicConstraint(
                id=next_id(),
                subject=" / ".join(terms),
                hard=False,  # v1 keyword lists were hints, not definitions
                description=f"The result is about (at least one of): {', '.join(terms)}",
            )
        )
    if gt.get("must_not_contain_keywords"):
        terms = gt["must_not_contain_keywords"]
        constraints.append(ExcludeConstraint(id=next_id(), concept=" / ".join(terms)))
    if gt.get("expected_entity"):
        constraints.append(
            EntityConstraint(
                id=next_id(), name=gt["expected_entity"], wrong=gt.get("wrong_entities") or []
            )
        )
    lo, hi, exact = gt.get("numeric_min"), gt.get("numeric_max"), gt.get("numeric_exact")
    attribute = gt.get("numeric_field") or "value"
    if exact is not None:
        constraints.append(
            NumericConstraint(id=next_id(), attribute=attribute, op="eq", value=exact)
        )
    elif lo is not None and hi is not None:
        constraints.append(
            NumericConstraint(
                id=next_id(), attribute=attribute, op="between", value=lo, value_max=hi
            )
        )
    elif lo is not None:
        constraints.append(NumericConstraint(id=next_id(), attribute=attribute, op="gte", value=lo))
    elif hi is not None:
        constraints.append(NumericConstraint(id=next_id(), attribute=attribute, op="lte", value=hi))
    start, end = _as_date(gt.get("date_min")), _as_date(gt.get("date_max"))
    if start or end:
        constraints.append(DateConstraint(id=next_id(), field="event", start=start, end=end))
    if gt.get("expected_language"):
        constraints.append(LanguageConstraint(id=next_id(), code=gt["expected_language"]))
    if gt.get("expected_types"):
        constraints.append(DocTypeConstraint(id=next_id(), allowed=gt["expected_types"]))
    if gt.get("manual_judgment"):
        constraints.append(FreeformConstraint(id=next_id(), requirement=gt["manual_judgment"]))
    return constraints


def query_to_case(
    text: str,
    category: str,
    ground_truth: dict[str, Any] | None = None,
    *,
    difficulty: str | None = None,
    tier: str | None = None,
    case_id: str | None = None,
    adversarial_reason: str | None = None,
) -> TestCase:
    """Build a ``TestCase`` from v1 query fields."""
    gt = ground_truth or {}
    constraints = ground_truth_to_constraints(gt)
    # The query itself is always a hard criterion: without it, a document that only
    # meets the mechanical constraints (e.g. any German page for a language
    # constraint) would count as fully satisfying.
    constraints.append(
        FreeformConstraint(
            id=f"c{len(constraints) + 1}",
            requirement=f"The result directly satisfies the query: '{text}'",
        )
    )
    return TestCase(
        id=case_id or "",
        text=text,
        category=AdversarialCategory(category),
        operators=CATEGORY_OPERATORS.get(category, []),
        difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium",
        constraints=constraints,
        known_positive_urls=gt.get("reference_urls") or [],
        provenance=Provenance(generator=tier or "v1"),
        validation="none",
        adversarial_reason=adversarial_reason,
        notes=gt.get("notes"),
    )


def db_query_to_case(row: dict[str, Any]) -> TestCase:
    """Build a ``TestCase`` from a ``queries`` table row (v2 spec preferred)."""
    import json

    spec = row.get("spec")
    if spec:
        data = json.loads(spec) if isinstance(spec, str) else spec
        # The spec's id is the stable case id (shared across query sets); the
        # row id only identifies this copy of the case in the database.
        return TestCase.model_validate(data)
    gt = row.get("ground_truth")
    if isinstance(gt, str):
        gt = json.loads(gt) if gt else {}
    return query_to_case(
        row["text"],
        row["category"],
        gt,
        difficulty=row.get("difficulty"),
        tier=row.get("tier"),
        case_id=row["id"],
    )
