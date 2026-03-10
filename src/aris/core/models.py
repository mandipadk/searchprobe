"""Core data models for Aris search engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl


class Document(BaseModel):
    """A document retrieved from any source, before scoring."""

    url: str = Field(..., description="Document URL")
    title: str = Field(default="", description="Page title")
    content: str = Field(default="", description="Full text content")
    snippet: str = Field(default="", description="Short excerpt")
    source: str = Field(default="", description="Data source name (e.g. 'duckduckgo', 'brave')")
    published_date: datetime | None = Field(default=None, description="Publication date if known")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    @property
    def domain(self) -> str:
        parsed = urlparse(self.url)
        return parsed.netloc.replace("www.", "")


class ScoredDocument(BaseModel):
    """A document with scores from retrieval, verification, and ranking."""

    document: Document
    retrieval_score: float = Field(default=0.0, description="Score from retrieval stage")
    verification_score: float = Field(default=1.0, description="Score from constraint verification")
    semantic_score: float = Field(default=0.0, description="Score from cross-encoder reranking")
    final_score: float = Field(default=0.0, description="Fused final score")
    constraint_results: dict[str, bool] = Field(
        default_factory=dict, description="Per-constraint pass/fail"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Result confidence")

    @property
    def url(self) -> str:
        return self.document.url

    @property
    def title(self) -> str:
        return self.document.title


class SearchResult(BaseModel):
    """Final search result returned to the user."""

    title: str
    url: str
    snippet: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    constraint_satisfaction: dict[str, bool] = Field(default_factory=dict)
    source: str = Field(default="")
    published_date: datetime | None = None

    @classmethod
    def from_scored(cls, scored: ScoredDocument) -> SearchResult:
        doc = scored.document
        return cls(
            title=doc.title,
            url=doc.url,
            snippet=doc.snippet or doc.content[:300] if doc.content else "",
            score=scored.final_score,
            confidence=scored.confidence,
            constraint_satisfaction=scored.constraint_results,
            source=doc.source,
            published_date=doc.published_date,
        )


class ArisResponse(BaseModel):
    """Complete response from an Aris search."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total_candidates: int = Field(default=0)
    iterations: int = Field(default=1)
    latency_ms: float = Field(default=0.0)
    strategy_used: str = Field(default="")
    predicted_failure_modes: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.results) > 0
