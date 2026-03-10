"""Document model with extracted metadata for indexing."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class IndexedDocument(BaseModel):
    """A document enriched with extracted metadata for structured queries."""

    url: str
    title: str = ""
    content: str = ""
    snippet: str = ""
    source: str = ""

    # Extracted metadata
    published_date: datetime | None = None
    domain: str = ""
    language: str = Field(default="en")
    word_count: int = 0

    # Numeric metadata extracted from content
    numeric_values: dict[str, float] = Field(
        default_factory=dict,
        description="Named numeric values (e.g. {'employees': 500, 'revenue_millions': 42})",
    )

    # Entity metadata
    entities: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
