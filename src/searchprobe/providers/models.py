"""Normalized data models for search requests and responses across all providers."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class SearchRequest(BaseModel):
    """Normalized search request that works across all providers."""

    query: str = Field(..., description="The search query")
    num_results: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    include_content: bool = Field(default=True, description="Whether to retrieve page content")
    max_content_chars: int = Field(
        default=5000, ge=100, le=50000, description="Max characters of content per result"
    )
    search_mode: str | None = Field(
        default=None, description="Provider-specific search mode (e.g., 'auto', 'neural', 'deep')"
    )


class SearchResult(BaseModel):
    """Normalized search result from any provider."""

    title: str = Field(..., description="Page title")
    url: HttpUrl = Field(..., description="Page URL")
    snippet: str = Field(default="", description="Short excerpt or description")
    content: str | None = Field(default=None, description="Full page content if retrieved")
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Relevance score (0-1, normalized)"
    )
    published_date: datetime | None = Field(default=None, description="Publication date if known")
    source_domain: str = Field(default="", description="Domain of the source")
    position: int = Field(default=0, ge=0, description="Position in results (0-indexed)")
    provider_raw: dict[str, Any] | None = Field(
        default=None, description="Original provider response for debugging"
    )

    @classmethod
    def from_url(cls, url: str | HttpUrl) -> "SearchResult":
        """Extract domain from URL."""
        url_str = str(url)
        # Extract domain from URL
        from urllib.parse import urlparse

        parsed = urlparse(url_str)
        domain = parsed.netloc.replace("www.", "")
        return domain


class SearchResponse(BaseModel):
    """Normalized response from a search provider."""

    provider: str = Field(..., description="Name of the search provider")
    search_mode: str | None = Field(default=None, description="Search mode used")
    query: str = Field(..., description="Original query")
    results: list[SearchResult] = Field(default_factory=list, description="Search results")
    latency_ms: float = Field(default=0.0, ge=0, description="Request latency in milliseconds")
    cost_usd: float = Field(default=0.0, ge=0, description="Estimated cost in USD")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When search was run")
    error: str | None = Field(default=None, description="Error message if search failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific metadata"
    )

    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return self.error is None and len(self.results) > 0

    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.results)
