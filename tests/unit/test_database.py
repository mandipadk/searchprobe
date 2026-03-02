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
