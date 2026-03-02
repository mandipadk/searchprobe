"""Tests for adversarial query taxonomy."""

import pytest

from searchprobe.queries.taxonomy import (
    AdversarialCategory,
    CATEGORY_METADATA,
    get_category_metadata,
)


def test_all_categories_have_metadata():
    """Every category should have metadata defined."""
    for category in AdversarialCategory:
        metadata = get_category_metadata(category)
        assert metadata is not None
        assert metadata.display_name
        assert metadata.description
        assert metadata.failure_hypothesis


def test_category_count():
    """Should have 13 adversarial categories."""
    assert len(AdversarialCategory) == 13


def test_category_enum_values():
    """Category enum values should be lowercase strings."""
    for category in AdversarialCategory:
        assert category.value == category.value.lower()
        assert "_" in category.value or category.value.isalpha()


def test_metadata_has_required_fields():
    """Each category metadata should have all required fields."""
    for category, metadata in CATEGORY_METADATA.items():
        assert metadata.display_name, f"Category {category} missing display_name"
        assert metadata.description, f"Category {category} missing description"
        assert metadata.failure_hypothesis, f"Category {category} missing failure_hypothesis"
        assert metadata.difficulty, f"Category {category} missing difficulty"


def test_difficulty_values():
    """Difficulty should be one of: easy, medium, hard."""
    valid_difficulties = {"easy", "medium", "hard"}

    for category, metadata in CATEGORY_METADATA.items():
        assert metadata.difficulty in valid_difficulties, \
            f"Invalid difficulty for {category}: {metadata.difficulty}"
