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


@pytest.fixture
def sample_search_results():
    """Sample search results for ground truth testing."""
    return [
        {
            "title": "Top Python Libraries for Data Science",
            "url": "https://example.com/python-libs",
            "snippet": "Python is the most popular language for data science with 1500 stars on GitHub.",
            "content": "Python libraries like pandas, numpy, and scikit-learn are essential for data science. Founded in 2020.",
        },
        {
            "title": "Java vs Python for Machine Learning",
            "url": "https://blog.example.com/java-python",
            "snippet": "Comparing Java and Python for ML workloads. Java has 200 employees.",
            "content": "While Java is faster, Python's ecosystem is more mature for ML tasks.",
        },
        {
            "title": "Best Rust Frameworks in 2024",
            "url": "https://rust.example.com/frameworks",
            "snippet": "Rust frameworks gaining popularity with exactly 50 contributors.",
            "content": "Actix-web and Rocket are leading Rust web frameworks.",
        },
    ]
