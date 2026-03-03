"""Fixtures for integration tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Any

from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult
from searchprobe.storage import Database


class MockSearchProvider(SearchProvider):
    """Mock search provider for testing."""

    NAME = "mock"
    SUPPORTED_MODES = ["default"]
    COST_PER_QUERY = {"default": 0.001}

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        """Initialize mock provider."""
        super().__init__(api_key="mock-key")
        self._results = results or []
        self.search_count = 0

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Return mock search results."""
        self.search_count += 1

        search_results = []
        for i, r in enumerate(self._results[:request.num_results]):
            search_results.append(
                SearchResult(
                    title=r.get("title", f"Result {i}"),
                    url=r.get("url", f"https://example.com/{i}"),
                    snippet=r.get("snippet", f"Snippet for result {i}"),
                    content=r.get("content", f"Content for result {i}"),
                    score=r.get("score", 1.0 - (i * 0.1)),
                    source_domain="example.com",
                    position=i,
                )
            )

        return SearchResponse(
            provider=self.NAME,
            search_mode=request.search_mode,
            query=request.query,
            results=search_results,
            latency_ms=10.0,
            cost_usd=self.COST_PER_QUERY.get("default", 0.001),
        )


@pytest.fixture
def mock_provider():
    """Create a mock search provider with sample results."""
    results = [
        {
            "title": f"Test Result {i}",
            "url": f"https://example.com/page-{i}",
            "snippet": f"This is test snippet {i}",
            "content": f"This is detailed content for test result {i}",
            "score": 1.0 - (i * 0.1),
        }
        for i in range(10)
    ]
    return MockSearchProvider(results=results)


@pytest.fixture
def integration_db():
    """Create a temporary database for integration testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    yield db

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_query_set(integration_db):
    """Create a sample query set in the database."""
    qs_id = integration_db.create_query_set(
        name="integration_test", config={"test": True}
    )

    queries = [
        ("companies that are NOT in AI", "negation", "medium", "seed"),
        ("startups with exactly 50 employees", "numeric_precision", "hard", "seed"),
        ("news from January 2025", "temporal_constraint", "hard", "seed"),
        ("Python AND (FastAPI OR Django)", "boolean_logic", "hard", "template"),
        ("Michael Jordan the professor at Berkeley", "entity_disambiguation", "medium", "seed"),
    ]

    query_ids = []
    for text, category, difficulty, tier in queries:
        qid = integration_db.add_query(
            query_set_id=qs_id,
            text=text,
            category=category,
            difficulty=difficulty,
            tier=tier,
        )
        query_ids.append(qid)

    return {"query_set_id": qs_id, "query_ids": query_ids, "queries": queries}
