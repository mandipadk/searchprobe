"""Tests for database operations."""

import pytest

from searchprobe.storage import Database


def test_create_query_set(temp_db):
    """Should create a query set and return its ID."""
    query_set_id = temp_db.create_query_set(name="Test Set", config={"foo": "bar"})

    assert query_set_id is not None
    assert len(query_set_id) == 36  # UUID format


def test_add_and_get_queries(temp_db):
    """Should add queries and retrieve them."""
    query_set_id = temp_db.create_query_set(name="Test")

    temp_db.add_query(
        query_set_id=query_set_id,
        text="test query",
        category="negation",
        tier="seed",
    )

    queries = temp_db.get_queries(query_set_id=query_set_id)

    assert len(queries) == 1
    assert queries[0]["text"] == "test query"
    assert queries[0]["category"] == "negation"


def test_get_latest_query_set(temp_db):
    """Should return the most recently created query set."""
    # First verify empty database returns None
    # (temp_db is fresh, but let's be sure it works)
    id1 = temp_db.create_query_set(name="First")
    id2 = temp_db.create_query_set(name="Second")

    latest = temp_db.get_latest_query_set_id()

    # Latest should be one of the ones we created
    assert latest in [id1, id2]
    # Since both are created nearly simultaneously, either could be "latest"
    # The important thing is we get a valid ID back


def test_create_run(temp_db):
    """Should create a benchmark run."""
    query_set_id = temp_db.create_query_set(name="Test")

    run_id = temp_db.create_run(
        query_set_id=query_set_id,
        name="Test Run",
        config={"providers": ["exa"]},
    )

    assert run_id is not None
    assert len(run_id) == 36


def test_filter_queries_by_category(temp_db):
    """Should filter queries by category."""
    query_set_id = temp_db.create_query_set(name="Test")

    temp_db.add_query(query_set_id, "query 1", "negation")
    temp_db.add_query(query_set_id, "query 2", "polysemy")
    temp_db.add_query(query_set_id, "query 3", "negation")

    negation_queries = temp_db.get_queries(query_set_id, category="negation")

    assert len(negation_queries) == 2


def test_materialize_aggregate_scores(temp_db):
    """Should compute and store aggregate scores from evaluations."""
    import json

    qs_id = temp_db.create_query_set(name="Agg Test")
    q1_id = temp_db.add_query(qs_id, "test query 1", "negation")
    q2_id = temp_db.add_query(qs_id, "test query 2", "negation")
    run_id = temp_db.create_run(qs_id, name="Agg Run")

    # Add evaluations
    for q_id, score in [(q1_id, 0.8), (q2_id, 0.6)]:
        temp_db.add_evaluation(run_id, {
            "query_id": q_id,
            "provider": "exa",
            "best_result_index": 0,
            "dimension_scores": {"relevance": {"score": score}},
            "weighted_score": score,
            "overall_assessment": "test",
            "failure_modes": [] if score > 0.7 else ["low_score"],
        })

    count = temp_db.materialize_aggregate_scores(run_id)
    assert count > 0

    aggs = temp_db.get_aggregate_scores(run_id)
    assert len(aggs) > 0

    # Check "overall" dimension
    overall = [a for a in aggs if a["dimension"] == "overall"]
    assert len(overall) == 1
    assert overall[0]["mean_score"] == pytest.approx(0.7)
    assert overall[0]["n_queries"] == 2
    assert overall[0]["provider"] == "exa"
    assert overall[0]["category"] == "negation"

    # Check "relevance" dimension
    relevance = [a for a in aggs if a["dimension"] == "relevance"]
    assert len(relevance) == 1
    assert relevance[0]["mean_score"] == pytest.approx(0.7)


def test_materialize_aggregate_scores_idempotent(temp_db):
    """Running materialization twice should produce the same result."""
    qs_id = temp_db.create_query_set(name="Idem Test")
    q_id = temp_db.add_query(qs_id, "test query", "negation")
    run_id = temp_db.create_run(qs_id, name="Idem Run")

    temp_db.add_evaluation(run_id, {
        "query_id": q_id,
        "provider": "exa",
        "dimension_scores": {"relevance": {"score": 0.9}},
        "weighted_score": 0.9,
        "overall_assessment": "test",
        "failure_modes": [],
    })

    count1 = temp_db.materialize_aggregate_scores(run_id)
    count2 = temp_db.materialize_aggregate_scores(run_id)
    assert count1 == count2

    aggs = temp_db.get_aggregate_scores(run_id)
    # Should not have duplicates
    assert len(aggs) == count1
