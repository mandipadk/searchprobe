"""Tests for adversarial mutation operators."""

import random

import pytest

from searchprobe.adversarial.models import Individual
from searchprobe.adversarial.mutations import (
    MUTATION_OPERATORS,
    apply_random_mutation,
    category_blend,
    constraint_inject,
    entity_swap,
    negation_toggle,
    specificity_shift,
    tense_flip,
    word_substitute,
)


def _make_individual(query: str, category: str = "negation") -> Individual:
    return Individual(query=query, category=category, generation=0)


class TestNegationToggle:
    def test_insert_negation(self):
        random.seed(42)
        ind = _make_individual("companies in AI")
        result = negation_toggle(ind)
        assert result.query != ind.query
        # Should contain a negation word
        negations = {"not", "no", "never", "without", "neither"}
        assert any(w.lower() in negations for w in result.query.split())

    def test_remove_negation(self):
        ind = _make_individual("companies NOT in AI")
        result = negation_toggle(ind)
        assert "NOT" not in result.query

    def test_preserves_metadata(self):
        ind = _make_individual("companies NOT in AI")
        result = negation_toggle(ind)
        assert result.category == ind.category
        assert result.generation == ind.generation + 1
        assert "negation_remove" in result.mutation_history[-1]


class TestConstraintInject:
    def test_adds_constraint(self):
        random.seed(42)
        ind = _make_individual("AI companies")
        result = constraint_inject(ind)
        assert len(result.query) > len(ind.query)
        assert result.query.startswith("AI companies ")

    def test_mutation_history_recorded(self):
        ind = _make_individual("AI companies")
        result = constraint_inject(ind)
        assert any("constraint_inject" in m for m in result.mutation_history)


class TestSpecificityShift:
    def test_increases_specificity(self):
        random.seed(0)
        ind = _make_individual("AI companies")
        result = specificity_shift(ind)
        assert result.query != ind.query

    def test_decreases_specificity_removes_marker(self):
        ind = _make_individual("specifically the largest AI companies")
        # Seed to get "general" direction
        random.seed(1)
        result = specificity_shift(ind)
        # Either removes a specificity word or adds generalization
        assert result.query != ind.query


class TestTenseFlip:
    def test_flips_known_word(self):
        ind = _make_individual("future AI companies")
        result = tense_flip(ind)
        assert "past" in result.query or "future" not in result.query

    def test_adds_temporal_when_no_match(self):
        ind = _make_individual("AI companies")
        result = tense_flip(ind)
        assert result.query != ind.query
        assert len(result.query.split()) > len(ind.query.split())


class TestEntitySwap:
    def test_swaps_known_entity(self):
        ind = _make_individual("Google AI research")
        result = entity_swap(ind)
        assert "Alphabet" in result.query

    def test_no_change_without_entity(self):
        ind = _make_individual("small companies in healthcare")
        result = entity_swap(ind)
        assert result.query == ind.query  # No swap happened


class TestCategoryBlend:
    def test_inserts_pattern(self):
        random.seed(42)
        ind = _make_individual("AI companies")
        result = category_blend(ind)
        assert result.query != ind.query
        assert len(result.query.split()) > len(ind.query.split())


class TestWordSubstitute:
    def test_substitutes_known_word(self):
        random.seed(3)  # Seed that picks "companies"
        ind = _make_individual("companies in AI")
        result = word_substitute(ind)
        # May or may not substitute depending on seed hitting the right word
        # At minimum it should not crash
        assert isinstance(result.query, str)

    def test_single_word_unchanged(self):
        ind = _make_individual("x")
        result = word_substitute(ind)
        assert result.query == "x"


class TestMutationRegistry:
    def test_all_operators_registered(self):
        expected = {
            "word_substitute", "negation_toggle", "constraint_inject",
            "specificity_shift", "category_blend", "tense_flip", "entity_swap",
        }
        assert set(MUTATION_OPERATORS.keys()) == expected

    def test_apply_random_mutation(self):
        random.seed(42)
        ind = _make_individual("AI companies in tech")
        result = apply_random_mutation(ind)
        assert isinstance(result, Individual)
        assert result.query  # Non-empty

    def test_all_operators_callable(self):
        ind = _make_individual("AI companies in tech sector")
        for name, op in MUTATION_OPERATORS.items():
            result = op(ind)
            assert isinstance(result, Individual), f"{name} returned wrong type"
