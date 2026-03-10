"""Data models for the Query Understanding Engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ComparisonOp(str, Enum):
    """Comparison operators for numeric/temporal constraints."""
    EQ = "eq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    APPROX = "approx"


class ConstraintType(str, Enum):
    """Types of constraints extracted from queries."""
    NEGATION = "negation"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    DOMAIN = "domain"
    BOOLEAN = "boolean"


class Constraint(BaseModel):
    """A single constraint extracted from the query."""
    type: ConstraintType
    value: Any = None
    operator: ComparisonOp | None = None
    field: str = Field(default="", description="What this constraint applies to (e.g. 'stars', 'date')")
    raw_text: str = Field(default="", description="Original text this was extracted from")
    unit: str = Field(default="", description="Unit if applicable (e.g. 'employees', 'dollars')")

    # For BETWEEN operator
    value_upper: Any = None

    # For negation
    negated_terms: list[str] = Field(default_factory=list)


class EntityReference(BaseModel):
    """An extracted entity with disambiguation context."""
    text: str
    entity_type: str = Field(default="", description="e.g. 'person', 'company', 'language'")
    disambiguation: str = Field(default="", description="Context to disambiguate")


class StructuredQuery(BaseModel):
    """The fully parsed, structured representation of a user query."""

    original_query: str
    core_intent: str = Field(default="", description="The main search intent, stripped of constraints")
    positive_query: str = Field(default="", description="Query with negations removed, for embedding")

    constraints: list[Constraint] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list, description="Terms to exclude from results")
    entities: list[EntityReference] = Field(default_factory=list)

    predicted_failure_modes: list[str] = Field(
        default_factory=list, description="Predicted FailureMode values"
    )

    # Strategy hints computed from failure mode predictions
    retriever_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.4, "sparse": 0.3, "structured": 0.2, "hyde": 0.1}
    )
    use_decomposition: bool = Field(default=False, description="Whether to decompose into sub-queries")
    verification_strategies: list[str] = Field(default_factory=list)
