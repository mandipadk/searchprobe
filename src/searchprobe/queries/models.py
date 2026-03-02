"""Data models for queries and ground truth."""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from searchprobe.queries.taxonomy import AdversarialCategory


class GroundTruth(BaseModel):
    """Flexible ground truth for verifying search result correctness.

    Supports multiple verification strategies depending on the query type.
    """

    strategy: Literal[
        "must_contain",
        "must_not_contain",
        "entity_match",
        "numeric_range",
        "date_range",
        "language_match",
        "type_match",
        "manual",
    ] = Field(..., description="Verification strategy to use")

    # For must_contain / must_not_contain strategies
    must_contain_keywords: list[str] | None = Field(
        default=None, description="Keywords that MUST appear in results"
    )
    must_not_contain_keywords: list[str] | None = Field(
        default=None, description="Keywords that must NOT appear in results"
    )

    # For entity_match strategy
    expected_entity: str | None = Field(
        default=None, description="The specific entity results should reference"
    )
    wrong_entities: list[str] | None = Field(
        default=None, description="Entities that should NOT appear (disambiguation)"
    )

    # For numeric_range strategy
    numeric_field: str | None = Field(
        default=None, description="Field name to extract numeric value from"
    )
    numeric_min: float | None = Field(default=None, description="Minimum acceptable value")
    numeric_max: float | None = Field(default=None, description="Maximum acceptable value")
    numeric_exact: float | None = Field(default=None, description="Exact value required")

    # For date_range strategy
    date_min: datetime | None = Field(default=None, description="Minimum date")
    date_max: datetime | None = Field(default=None, description="Maximum date")

    # For language_match strategy
    expected_language: str | None = Field(
        default=None, description="Expected language code (e.g., 'en', 'es')"
    )

    # For type_match strategy
    expected_types: list[str] | None = Field(
        default=None, description="Expected content types (e.g., 'academic paper', 'blog post')"
    )

    # For manual strategy
    manual_judgment: str | None = Field(
        default=None, description="Instructions for manual verification"
    )

    # Reference URLs that are known-good results
    reference_urls: list[str] | None = Field(
        default=None, description="Known-good URLs that should appear in results"
    )

    # Additional context
    notes: str | None = Field(
        default=None, description="Additional notes about ground truth"
    )


class Query(BaseModel):
    """A single adversarial query with metadata and ground truth."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., description="The query text")
    category: AdversarialCategory = Field(..., description="Adversarial category")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Difficulty level"
    )
    tier: Literal["seed", "template", "llm"] = Field(
        default="seed", description="How the query was generated"
    )
    ground_truth: GroundTruth | None = Field(
        default=None, description="Ground truth for verification"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Why this query is adversarial
    adversarial_reason: str | None = Field(
        default=None,
        description="Explanation of why this query challenges embedding search",
    )


class QuerySet(BaseModel):
    """A collection of queries for a benchmark run."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str | None = Field(default=None, description="Human-readable name")
    queries: list[Query] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    config: dict[str, Any] = Field(
        default_factory=dict, description="Generation configuration"
    )

    @property
    def total_queries(self) -> int:
        """Total number of queries in this set."""
        return len(self.queries)

    def queries_by_category(self) -> dict[AdversarialCategory, list[Query]]:
        """Group queries by category."""
        result: dict[AdversarialCategory, list[Query]] = {}
        for query in self.queries:
            if query.category not in result:
                result[query.category] = []
            result[query.category].append(query)
        return result

    def queries_by_difficulty(self) -> dict[str, list[Query]]:
        """Group queries by difficulty."""
        result: dict[str, list[Query]] = {"easy": [], "medium": [], "hard": []}
        for query in self.queries:
            result[query.difficulty].append(query)
        return result

    def add_query(self, query: Query) -> None:
        """Add a query to the set."""
        self.queries.append(query)

    def to_json_list(self) -> list[dict[str, Any]]:
        """Export queries as a list of dictionaries."""
        return [q.model_dump(mode="json") for q in self.queries]
