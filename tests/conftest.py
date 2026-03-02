"""Pytest fixtures for SearchProbe tests."""

import pytest
import tempfile
from pathlib import Path

from searchprobe.storage import Database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    yield db

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {
            "id": "test-1",
            "text": "companies that are NOT in AI",
            "category": "negation",
        },
        {
            "id": "test-2",
            "text": "startups with exactly 50 employees",
            "category": "numeric_precision",
        },
        {
            "id": "test-3",
            "text": "news from January 2025",
            "category": "temporal_constraint",
        },
    ]
