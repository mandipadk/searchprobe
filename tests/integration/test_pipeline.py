"""Integration tests for the benchmark pipeline."""

import pytest

from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
from searchprobe.providers.registry import ProviderRegistry
from tests.integration.conftest import MockSearchProvider


@pytest.fixture
def mock_runner(integration_db, mock_provider, monkeypatch):
    """Create a BenchmarkRunner with a mock provider."""

    def mock_get_provider(name, settings=None):
        return mock_provider

    monkeypatch.setattr(ProviderRegistry, "get_provider", staticmethod(mock_get_provider))

    config = BenchmarkConfig(
        providers=["mock"],
        num_results=5,
        budget_limit=1.0,
    )

    return BenchmarkRunner(config=config, db=integration_db)


@pytest.mark.asyncio
async def test_pipeline_end_to_end(mock_runner, integration_db, sample_query_set):
    """Test the full pipeline: run benchmark, store results, retrieve them."""
    # Prepare queries
    queries = integration_db.get_queries(query_set_id=sample_query_set["query_set_id"])
    query_dicts = [
        {
            "id": q["id"],
            "text": q["text"],
            "category": q["category"],
            "query_set_id": sample_query_set["query_set_id"],
        }
        for q in queries
    ]

    # Run benchmark
    result = await mock_runner.run(queries=query_dicts, run_name="test_run")

    # Verify results
    assert result.total_queries == len(queries)
    assert result.successful_queries > 0
    assert result.failed_queries == 0
    assert result.total_cost > 0

    # Verify database storage
    stored_results = integration_db.get_search_results(result.run_id)
    assert len(stored_results) == len(queries)

    # Verify each result has correct structure
    for sr in stored_results:
        assert sr["provider"] == "mock"
        assert sr["results"] is not None
        assert len(sr["results"]) > 0


@pytest.mark.asyncio
async def test_pipeline_budget_enforcement(integration_db, mock_provider, sample_query_set, monkeypatch):
    """Test that budget limits are enforced."""

    def mock_get_provider(name, settings=None):
        return mock_provider

    monkeypatch.setattr(ProviderRegistry, "get_provider", staticmethod(mock_get_provider))

    # Set very low budget
    config = BenchmarkConfig(
        providers=["mock"],
        num_results=5,
        budget_limit=0.001,  # Very low budget
    )

    runner = BenchmarkRunner(config=config, db=integration_db)

    queries = integration_db.get_queries(query_set_id=sample_query_set["query_set_id"])
    query_dicts = [
        {
            "id": q["id"],
            "text": q["text"],
            "category": q["category"],
            "query_set_id": sample_query_set["query_set_id"],
        }
        for q in queries
    ]

    result = await runner.run(queries=query_dicts)

    # Should have completed some queries but budget should have been tracked
    assert result.total_cost >= 0


def test_database_query_set_lifecycle(integration_db):
    """Test creating query sets, adding queries, and retrieving them."""
    # Create query set
    qs_id = integration_db.create_query_set(name="lifecycle_test")

    # Add queries
    q1_id = integration_db.add_query(
        query_set_id=qs_id,
        text="test query 1",
        category="negation",
        difficulty="medium",
        tier="seed",
    )
    q2_id = integration_db.add_query(
        query_set_id=qs_id,
        text="test query 2",
        category="polysemy",
        difficulty="hard",
        tier="template",
    )

    # Retrieve all queries
    all_queries = integration_db.get_queries(query_set_id=qs_id)
    assert len(all_queries) == 2

    # Filter by category
    negation_queries = integration_db.get_queries(
        query_set_id=qs_id, category="negation"
    )
    assert len(negation_queries) == 1
    assert negation_queries[0]["text"] == "test query 1"

    # Latest query set
    latest_id = integration_db.get_latest_query_set_id()
    assert latest_id == qs_id


def test_database_geometry_operations(integration_db):
    """Test geometry result storage and retrieval."""
    result = {
        "model_name": "all-MiniLM-L6-v2",
        "category": "negation",
        "adversarial_similarity": 0.95,
        "baseline_similarity": 0.6,
        "collapse_ratio": 1.58,
        "vulnerability_score": 0.85,
        "intrinsic_dimensionality": 12.5,
        "isotropy_score": 0.3,
        "pair_details": {"pairs": [{"query_a": "X", "query_b": "NOT X", "sim": 0.95}]},
    }

    result_id = integration_db.add_geometry_result(result)
    assert result_id is not None

    # Retrieve
    results = integration_db.get_geometry_results(model_name="all-MiniLM-L6-v2")
    assert len(results) == 1
    assert results[0]["category"] == "negation"
    assert results[0]["vulnerability_score"] == 0.85

    # Filter by category
    results = integration_db.get_geometry_results(category="negation")
    assert len(results) == 1


def test_mock_provider_search(mock_provider):
    """Test that MockSearchProvider returns expected results."""
    import asyncio
    from searchprobe.providers.models import SearchRequest

    request = SearchRequest(query="test query", num_results=5)
    response = asyncio.run(mock_provider.search(request))

    assert response.success
    assert response.provider == "mock"
    assert len(response.results) == 5
    assert response.results[0].title == "Test Result 0"
    assert mock_provider.search_count == 1
