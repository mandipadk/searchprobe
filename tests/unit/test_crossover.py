"""Tests for adversarial crossover operators."""

import random

import pytest

from searchprobe.adversarial.crossover import (
    CROSSOVER_OPERATORS,
    apply_random_crossover,
    clause_swap,
    constraint_merge,
    template_blend,
)
from searchprobe.adversarial.models import Individual


def _make_individual(query: str, category: str = "negation") -> Individual:
    return Individual(query=query, category=category, generation=0)


class TestClauseSwap:
    def test_basic_swap(self):
        random.seed(42)
        a = _make_individual("companies not in AI")
        b = _make_individual("startups with exactly 50 employees")
        result = clause_swap(a, b)
        assert result.query != a.query
        assert result.query != b.query
        assert result.generation == 1
        assert a.id in result.parent_ids
        assert b.id in result.parent_ids

    def test_short_query_returns_parent(self):
        a = _make_individual("x")
        b = _make_individual("y")
        result = clause_swap(a, b)
        assert result.query == "x"

    def test_mutation_history(self):
        a = _make_individual("companies not in AI")
        b = _make_individual("startups founded after 2020")
        result = clause_swap(a, b)
        assert "clause_swap" in result.mutation_history


class TestConstraintMerge:
    def test_merges_queries(self):
        a = _make_individual("AI companies")
        b = _make_individual("founded after 2020")
        result = constraint_merge(a, b)
        assert "AI companies" in result.query
        assert "founded after 2020" in result.query

    def test_truncates_long_queries(self):
        a = _make_individual(" ".join(["word"] * 20))
        b = _make_individual(" ".join(["other"] * 20))
        result = constraint_merge(a, b)
        assert len(result.query.split()) <= 30

    def test_mutation_history(self):
        a = _make_individual("AI companies")
        b = _make_individual("in Europe")
        result = constraint_merge(a, b)
        assert "constraint_merge" in result.mutation_history


class TestTemplateBlend:
    def test_blends_content_words(self):
        random.seed(42)
        a = _make_individual("companies in the AI sector")
        b = _make_individual("healthcare startups with revenue")
        result = template_blend(a, b)
        assert result.query != a.query

    def test_falls_back_to_clause_swap(self):
        # Parent b has only function words -> fallback
        a = _make_individual("companies in AI")
        b = _make_individual("the a an in on")
        result = template_blend(a, b)
        assert isinstance(result, Individual)

    def test_mutation_history(self):
        random.seed(42)
        a = _make_individual("companies in the AI sector")
        b = _make_individual("healthcare ventures with employees")
        result = template_blend(a, b)
        assert result.mutation_history[-1] in ("template_blend", "clause_swap")


class TestCrossoverRegistry:
    def test_all_operators_registered(self):
        expected = {"clause_swap", "constraint_merge", "template_blend"}
        assert set(CROSSOVER_OPERATORS.keys()) == expected

    def test_apply_random_crossover(self):
        random.seed(42)
        a = _make_individual("companies not in AI sector")
        b = _make_individual("startups founded after 2020")
        result = apply_random_crossover(a, b)
        assert isinstance(result, Individual)
        assert result.query

    def test_all_operators_callable(self):
        a = _make_individual("companies not in AI sector")
        b = _make_individual("startups with many employees worldwide")
        for name, op in CROSSOVER_OPERATORS.items():
            result = op(a, b)
            assert isinstance(result, Individual), f"{name} returned wrong type"
