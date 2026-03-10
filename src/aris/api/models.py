"""Request/Response schemas for the Aris API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search API request."""
    query: str = Field(..., description="Search query", min_length=1)
    num_results: int = Field(default=10, ge=1, le=100)
    use_cross_encoder: bool = Field(default=False, description="Enable cross-encoder reranking")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max refinement iterations")


class SearchResultResponse(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str
    score: float
    confidence: float
    constraint_satisfaction: dict[str, bool] = Field(default_factory=dict)
    source: str = ""


class SearchResponse(BaseModel):
    """Search API response."""
    query: str
    results: list[SearchResultResponse] = Field(default_factory=list)
    total_candidates: int = 0
    iterations: int = 1
    latency_ms: float = 0.0
    strategy_used: str = ""
    predicted_failure_modes: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = ""
    sources_available: list[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    """Index API request."""
    urls: list[str] = Field(..., min_length=1)
