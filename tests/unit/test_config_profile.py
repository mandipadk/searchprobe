"""Tests for TOML configuration profiles."""

import tempfile
from pathlib import Path

import pytest

from searchprobe.core.config import SearchProbeProfile


def test_defaults():
    profile = SearchProbeProfile(name="test")
    assert profile.name == "test"
    assert profile.providers == ["exa"]
    assert profile.exa_modes == ["auto"]
    assert profile.categories == []
    assert profile.queries_per_category == 5
    assert profile.run_geometry is False
    assert profile.run_evolution is False
    assert profile.budget_limit == 10.0


def test_from_toml():
    content = """
name = "My Test"
description = "Testing TOML loading"
providers = ["exa", "tavily"]
categories = ["negation"]
run_geometry = true
evolution_generations = 5
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name

    try:
        profile = SearchProbeProfile.from_toml(path)
        assert profile.name == "My Test"
        assert profile.description == "Testing TOML loading"
        assert profile.providers == ["exa", "tavily"]
        assert profile.categories == ["negation"]
        assert profile.run_geometry is True
        assert profile.evolution_generations == 5
        # Defaults preserved
        assert profile.run_perturbation is False
        assert profile.budget_limit == 10.0
    finally:
        Path(path).unlink()


def test_from_toml_not_found():
    with pytest.raises(FileNotFoundError):
        SearchProbeProfile.from_toml("/nonexistent/path.toml")


def test_to_dict():
    profile = SearchProbeProfile(name="test", providers=["brave"])
    d = profile.to_dict()
    assert d["name"] == "test"
    assert d["providers"] == ["brave"]
    assert "budget_limit" in d


def test_validation():
    with pytest.raises(Exception):
        SearchProbeProfile(name="bad", queries_per_category=-1)
