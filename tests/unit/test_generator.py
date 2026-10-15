"""Tests for LLM query generation (fake client, no network)."""

import asyncio

from searchprobe.queries.generator import QueryGenerator
from searchprobe.queries.taxonomy import AdversarialCategory
from tests.fakes import FakeAnthropic


def _item(text, **kw):
    base = {
        "text": text,
        "difficulty": "hard",
        "adversarial_reason": "r",
        "must_contain_keywords": [],
        "must_not_contain_keywords": [],
        "expected_entity": None,
        "numeric_min": None,
        "numeric_max": None,
        "date_min": None,
        "date_max": None,
        "expected_language": None,
        "expected_types": [],
        "verification_guidance": "check it",
    }
    base.update(kw)
    return base


def test_generated_queries_get_structured_ground_truth_and_are_deduplicated():
    payload = {
        "queries": [
            _item("Banks that are not owned by the government",
                  must_not_contain_keywords=["state-owned bank"]),
            _item("Skyscrapers between 300 and 350 meters", numeric_min=300, numeric_max=350),
            _item("Existing query"),
            _item("Bad date query", date_min="last week"),
        ]
    }
    gen = QueryGenerator(model="m", client=FakeAnthropic(lambda kw: payload))
    queries = asyncio.run(
        gen.agenerate_for_category(AdversarialCategory.NEGATION, 4, ["existing query"])
    )
    by_text = {q.text: q for q in queries}
    assert "Existing query" not in by_text
    neg = by_text["Banks that are not owned by the government"].ground_truth
    assert neg.strategy == "must_not_contain"
    num = by_text["Skyscrapers between 300 and 350 meters"].ground_truth
    assert (num.strategy, num.numeric_min, num.numeric_max) == ("numeric_range", 300, 350)
    assert by_text["Bad date query"].ground_truth.date_min is None  # unparseable dropped
