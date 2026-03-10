"""Tests for pipeline cost tracker."""

import pytest

from searchprobe.pipeline.cost_tracker import CostRecord, CostTracker


class TestCostRecord:
    def test_record_creation(self):
        r = CostRecord(provider="exa", operation="search_auto", cost_usd=0.005)
        assert r.provider == "exa"
        assert r.cost_usd == 0.005
        assert r.timestamp is not None

    def test_record_with_metadata(self):
        r = CostRecord(
            provider="exa", operation="search_auto", cost_usd=0.005,
            metadata={"query_id": "q1"},
        )
        assert r.metadata["query_id"] == "q1"


class TestCostTracker:
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.get_total() == 0.0
        assert tracker.get_total_by_provider() == {}
        assert tracker.get_total_by_operation() == {}
        assert len(tracker.records) == 0

    def test_record_single(self):
        tracker = CostTracker()
        tracker.record("exa", "search_auto", 0.005)
        assert tracker.get_total() == pytest.approx(0.005)
        assert tracker.get_total_by_provider() == {"exa": pytest.approx(0.005)}

    def test_record_multiple_providers(self):
        tracker = CostTracker()
        tracker.record("exa", "search_auto", 0.005)
        tracker.record("tavily", "search_basic", 0.01)
        assert tracker.get_total() == pytest.approx(0.015)
        by_provider = tracker.get_total_by_provider()
        assert by_provider["exa"] == pytest.approx(0.005)
        assert by_provider["tavily"] == pytest.approx(0.01)

    def test_record_accumulates_same_provider(self):
        tracker = CostTracker()
        tracker.record("exa", "search_auto", 0.005)
        tracker.record("exa", "search_auto", 0.005)
        tracker.record("exa", "content", 0.001)
        assert tracker.get_total() == pytest.approx(0.011)
        assert tracker.get_total_by_provider()["exa"] == pytest.approx(0.011)

    def test_get_total_by_operation(self):
        tracker = CostTracker()
        tracker.record("exa", "search_auto", 0.005)
        tracker.record("exa", "content", 0.001)
        tracker.record("exa", "search_auto", 0.005)
        by_op = tracker.get_total_by_operation()
        assert by_op["exa:search_auto"] == pytest.approx(0.010)
        assert by_op["exa:content"] == pytest.approx(0.001)

    def test_budget_not_exceeded_no_limit(self):
        tracker = CostTracker(budget_limit=None)
        tracker.record("exa", "search_auto", 100.0)
        assert tracker.is_budget_exceeded() is False

    def test_budget_not_exceeded_under_limit(self):
        tracker = CostTracker(budget_limit=1.0)
        tracker.record("exa", "search_auto", 0.5)
        assert tracker.is_budget_exceeded() is False

    def test_budget_exceeded_at_limit(self):
        tracker = CostTracker(budget_limit=0.01)
        tracker.record("exa", "search_auto", 0.01)
        assert tracker.is_budget_exceeded() is True

    def test_budget_exceeded_over_limit(self):
        tracker = CostTracker(budget_limit=0.01)
        tracker.record("exa", "search_auto", 0.02)
        assert tracker.is_budget_exceeded() is True

    def test_remaining_budget_no_limit(self):
        tracker = CostTracker(budget_limit=None)
        assert tracker.remaining_budget() is None

    def test_remaining_budget_with_limit(self):
        tracker = CostTracker(budget_limit=1.0)
        tracker.record("exa", "search_auto", 0.3)
        assert tracker.remaining_budget() == pytest.approx(0.7)

    def test_remaining_budget_floor_zero(self):
        tracker = CostTracker(budget_limit=0.01)
        tracker.record("exa", "search_auto", 0.05)
        assert tracker.remaining_budget() == 0.0

    def test_estimate_operation_cost_known(self):
        tracker = CostTracker()
        assert tracker.estimate_operation_cost("exa", "search_auto") == 0.005
        assert tracker.estimate_operation_cost("serpapi", "search_google") == 0.01

    def test_estimate_operation_cost_unknown(self):
        tracker = CostTracker()
        assert tracker.estimate_operation_cost("unknown", "search") == 0.005

    def test_estimate_remaining_queries_no_limit(self):
        tracker = CostTracker(budget_limit=None)
        assert tracker.estimate_remaining_queries("exa", "search_auto") is None

    def test_estimate_remaining_queries_with_limit(self):
        tracker = CostTracker(budget_limit=0.05)
        remaining = tracker.estimate_remaining_queries("exa", "search_auto")
        assert remaining == 10  # 0.05 / 0.005

    def test_to_dict(self):
        tracker = CostTracker(budget_limit=1.0)
        tracker.record("exa", "search_auto", 0.005)
        d = tracker.to_dict()
        assert d["total_cost"] == pytest.approx(0.005)
        assert d["budget_limit"] == 1.0
        assert d["remaining_budget"] == pytest.approx(0.995)
        assert d["record_count"] == 1
        assert "exa" in d["by_provider"]
