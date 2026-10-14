"""TestCase v2: verifiable adversarial test cases with typed constraints.

A test case is a query plus a machine-readable statement of what a correct
result must (and must not) be. Constraints are typed so that mechanical ones
(dates, languages, membership in a known answer set) can be checked
deterministically, and the rest are judged per document by an LLM with a
fixed verdict vocabulary.

Verdict semantics, per (constraint, document):
    satisfied   - the document clearly meets the constraint
    violated    - the document clearly breaks it (for exclusions: the document
                  IS the excluded thing, not merely mentions it)
    unaddressed - the document does not let you tell
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from searchprobe.queries.taxonomy import AdversarialCategory

SCHEMA_VERSION = 2

Verdict = Literal["satisfied", "violated", "unaddressed"]
VERDICTS: tuple[str, ...] = ("satisfied", "violated", "unaddressed")


class _ConstraintBase(BaseModel):
    """Fields shared by every constraint type."""

    id: str = Field(..., description="Stable id within the case, e.g. 'c1'")
    hard: bool = Field(default=True, description="Hard constraints define correctness")
    description: str = Field(default="", description="Human-readable statement")

    def describe(self) -> str:
        """One-line statement used in judge prompts and reports."""
        return self.description or self._default_description()

    def _default_description(self) -> str:  # pragma: no cover - overridden
        return self.id


class TopicConstraint(_ConstraintBase):
    """The document must be about this subject (the positive core of the query)."""

    type: Literal["topic"] = "topic"
    subject: str

    def _default_description(self) -> str:
        return f"The result is about {self.subject}"


class ExcludeConstraint(_ConstraintBase):
    """The document must NOT be (or be about) the excluded concept or entities."""

    type: Literal["exclude"] = "exclude"
    concept: str
    entity_qids: list[str] = Field(default_factory=list)

    def _default_description(self) -> str:
        return (
            f"The result is NOT {self.concept} (mentioning {self.concept} is fine; "
            f"being {self.concept} is a violation)"
        )


NumericOp = Literal["eq", "approx", "lt", "lte", "gt", "gte", "between"]


class NumericConstraint(_ConstraintBase):
    """A numeric attribute of the result's subject must satisfy a comparison."""

    type: Literal["numeric"] = "numeric"
    attribute: str = Field(..., description="e.g. 'number of employees', 'height'")
    op: NumericOp
    value: float
    value_max: float | None = Field(default=None, description="Upper bound for 'between'")
    unit: str | None = None
    tolerance: float = Field(default=0.0, description="Relative tolerance for 'approx'/'eq'")

    @model_validator(mode="after")
    def _check_between(self) -> NumericConstraint:
        if self.op == "between":
            if self.value_max is None:
                raise ValueError("'between' requires value_max")
            if self.value_max < self.value:
                raise ValueError("value_max must be >= value")
        return self

    def check(self, x: float) -> bool:
        """Evaluate the comparison against a known value."""
        v, tol = self.value, self.tolerance
        if self.op in ("eq", "approx"):
            band = abs(v) * tol
            return v - band <= x <= v + band
        if self.op == "lt":
            return x < v
        if self.op == "lte":
            return x <= v
        if self.op == "gt":
            return x > v
        if self.op == "gte":
            return x >= v
        assert self.value_max is not None
        return v <= x <= self.value_max

    def _default_description(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        if self.op == "between":
            return f"{self.attribute} is between {self.value:g} and {self.value_max:g}{unit}"
        words = {
            "eq": "exactly",
            "approx": "approximately",
            "lt": "less than",
            "lte": "at most",
            "gt": "more than",
            "gte": "at least",
        }
        return f"{self.attribute} is {words[self.op]} {self.value:g}{unit}"


class DateConstraint(_ConstraintBase):
    """A date must fall in a window.

    ``field='published'`` refers to the document's publication date and can be
    checked mechanically; ``field='event'`` refers to when the described event
    happened and needs reading.
    """

    type: Literal["date"] = "date"
    field: Literal["published", "event"] = "event"
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _check_window(self) -> DateConstraint:
        if self.start is None and self.end is None:
            raise ValueError("date constraint needs start and/or end")
        if self.start and self.end and self.end < self.start:
            raise ValueError("end must be >= start")
        return self

    def contains(self, d: date) -> bool:
        return (self.start is None or d >= self.start) and (self.end is None or d <= self.end)

    def _default_description(self) -> str:
        what = "The result was published" if self.field == "published" else "The event happened"
        lo = self.start.isoformat() if self.start else "-inf"
        hi = self.end.isoformat() if self.end else "+inf"
        return f"{what} between {lo} and {hi}"


class EntityConstraint(_ConstraintBase):
    """The result must be about a specific entity, not a same-named one."""

    type: Literal["entity"] = "entity"
    name: str
    qid: str | None = None
    wrong: list[str] = Field(default_factory=list, description="Confusable entities")

    def _default_description(self) -> str:
        extra = f" (not {', '.join(self.wrong)})" if self.wrong else ""
        return f"The result is about {self.name}{extra}"


class RelationConstraint(_ConstraintBase):
    """A directed relation must hold: subject --predicate--> object."""

    type: Literal["relation"] = "relation"
    subject: str
    predicate: str
    object: str
    directed: bool = True

    def _default_description(self) -> str:
        arrow = "(direction matters)" if self.directed else ""
        return f"{self.subject} {self.predicate} {self.object} {arrow}".strip()


class DocTypeConstraint(_ConstraintBase):
    """The result must be a certain kind of document."""

    type: Literal["doc_type"] = "doc_type"
    allowed: list[str] = Field(default_factory=list)
    disallowed: list[str] = Field(default_factory=list)

    def _default_description(self) -> str:
        parts = []
        if self.allowed:
            parts.append("is a " + " or ".join(self.allowed))
        if self.disallowed:
            parts.append("is not a " + " or ".join(self.disallowed))
        return "The result " + " and ".join(parts)


class LanguageConstraint(_ConstraintBase):
    """The result must be written in a given language (ISO 639-1 code)."""

    type: Literal["language"] = "language"
    code: str

    @field_validator("code")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()

    def _default_description(self) -> str:
        return f"The result is written in language '{self.code}'"


class FreeformConstraint(_ConstraintBase):
    """A requirement that does not fit a typed constraint; always LLM-judged."""

    type: Literal["freeform"] = "freeform"
    requirement: str

    def _default_description(self) -> str:
        return self.requirement


Constraint = Annotated[
    TopicConstraint
    | ExcludeConstraint
    | NumericConstraint
    | DateConstraint
    | EntityConstraint
    | RelationConstraint
    | DocTypeConstraint
    | LanguageConstraint
    | FreeformConstraint,
    Field(discriminator="type"),
]


class KnownEntity(BaseModel):
    """An entity in a case's answer set or negative set."""

    label: str
    qid: str | None = None
    wikipedia_url: str | None = None
    aliases: list[str] = Field(default_factory=list)
    values: dict[str, Any] = Field(
        default_factory=dict, description="Known attribute values, e.g. {'height': 452}"
    )
    violates: list[str] = Field(
        default_factory=list,
        description="For negative-set entities: ids of the constraints this entity breaks "
        "(empty = every exclude constraint)",
    )


class Provenance(BaseModel):
    """Where a case came from."""

    generator: str = Field(..., description="seed | template | llm | wikidata | discovered | ...")
    version: str = "1"
    source: str | None = Field(default=None, description="SPARQL query, template, or model id")
    created: str | None = None
    notes: str | None = None


class TestCase(BaseModel):
    """A verifiable adversarial test case."""

    __test__ = False  # not a pytest test class

    schema_version: int = SCHEMA_VERSION
    id: str = ""
    text: str
    category: AdversarialCategory
    operators: list[str] = Field(
        default_factory=list,
        description="Logical operators exercised, e.g. ['negation'], ['numeric', 'conjunction']",
    )
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    constraints: list[Constraint] = Field(default_factory=list)

    # Minimal pairs. ``twins`` maps relation -> case id for cases derived from
    # this one; ``twin_of`` points from a derived case back to its base.
    twins: dict[str, str] = Field(default_factory=dict)
    twin_of: str | None = None
    twin_relation: str | None = None

    answer_set: list[KnownEntity] = Field(default_factory=list)
    negative_set: list[KnownEntity] = Field(default_factory=list)
    answer_set_complete: bool = Field(
        default=False, description="True if answer_set lists every correct entity"
    )
    known_positive_urls: list[str] = Field(default_factory=list)
    known_negative_urls: list[str] = Field(default_factory=list)

    provenance: Provenance = Field(default_factory=lambda: Provenance(generator="manual"))
    validation: Literal["gold", "auto", "none"] = "none"
    adversarial_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _finalize(self) -> TestCase:
        if not self.id:
            self.id = case_id_for(self.text, self.category.value)
        ids = [c.id for c in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate constraint ids in case {self.id}: {ids}")
        return self

    @property
    def hard_constraints(self) -> list[Any]:
        return [c for c in self.constraints if c.hard]

    def constraint(self, constraint_id: str) -> Any:
        for c in self.constraints:
            if c.id == constraint_id:
                return c
        raise KeyError(constraint_id)

    @property
    def is_verifiable(self) -> bool:
        """True if correctness is defined by something other than free text alone."""
        return any(c.type != "freeform" for c in self.hard_constraints) or bool(self.answer_set)


def case_id_for(text: str, category: str) -> str:
    """Deterministic id so the same query always maps to the same case."""
    digest = hashlib.sha256(f"{category}::{' '.join(text.lower().split())}".encode()).hexdigest()
    return f"case-{digest[:16]}"
