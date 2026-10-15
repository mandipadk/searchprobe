"""Tests for the list-level LLM-as-judge."""

import asyncio
import time

import pytest

from searchprobe.config import Settings
from searchprobe.evaluation.dimensions import EvaluationDimension
from searchprobe.evaluation.judge import (
    STATUS_JUDGE_ERROR,
    STATUS_NO_RESULTS,
    STATUS_OK,
    STATUS_SEARCH_ERROR,
    DimensionScore,
    EvaluationResult,
    SearchJudge,
    build_judge_schema,
)
from searchprobe.evaluation.prompts import build_evaluation_prompt, format_ground_truth
from searchprobe.providers.models import SearchResponse, SearchResult
from tests.fakes import FakeAnthropic, user_prompt


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, anthropic_api_key="test", **kw)


def _ok_payload(kwargs):
    dims = kwargs["output_config"]["format"]["schema"]["properties"]["scores"]["required"]
    return {
        "scores": {d: {"score": 0.8, "justification": "fine"} for d in dims},
        "failure_modes": ["partial_constraint", "partial_constraint"],
        "best_result_index": 1,
        "overall_assessment": "ok",
    }


def _response(n=2, error=None):
    results = [
        SearchResult(title=f"t{i}", url=f"https://e.com/{i}", snippet="s") for i in range(n)
    ]
    return SearchResponse(
        provider="exa", search_mode="neural", query="q", results=results, error=error
    )


class TestEvaluationResult:
    def test_to_dict_carries_mode_status_and_judge(self):
        result = EvaluationResult(
            query_id="q1",
            provider="exa",
            search_mode="auto",
            category="negation",
            dimension_scores={
                "relevance": DimensionScore(EvaluationDimension.RELEVANCE, 0.8, "Relevant")
            },
            weighted_score=0.8,
            failure_modes=[],
            best_result_index=1,
            overall_assessment="Good",
            judge_model="m",
        )
        d = result.to_dict()
        assert d["search_mode"] == "auto"
        assert d["status"] == STATUS_OK
        assert d["judge_model"] == "m"
        assert d["prompt_version"]
        assert d["dimension_scores"]["relevance"]["score"] == 0.8


class TestJudgeStatuses:
    def test_search_error_is_infra_not_quality(self):
        judge = SearchJudge(_settings(), client=FakeAnthropic(_ok_payload))
        r = asyncio.run(judge.evaluate("q1", "x", "negation", _response(0, error="boom")))
        assert r.status == STATUS_SEARCH_ERROR
        assert not r.counts_for_quality

    def test_empty_results_count_as_quality_zero(self):
        judge = SearchJudge(_settings(), client=FakeAnthropic(_ok_payload))
        r = asyncio.run(judge.evaluate("q1", "x", "negation", _response(0)))
        assert r.status == STATUS_NO_RESULTS
        assert r.counts_for_quality
        assert r.weighted_score == 0.0

    def test_judge_failure_is_infra(self):
        judge = SearchJudge(_settings(), client=FakeAnthropic(lambda kw: "not json"))
        r = asyncio.run(judge.evaluate("q1", "x", "negation", _response()))
        assert r.status == STATUS_JUDGE_ERROR
        assert "judge_failed" in r.failure_modes


class TestJudgeCall:
    def test_uses_configured_model_structured_output_and_no_temperature(self):
        client = FakeAnthropic(_ok_payload)
        judge = SearchJudge(_settings(judge_model="claude-haiku-5-5"), client=client)
        r = asyncio.run(judge.evaluate("q1", "x", "negation", _response()))
        call = client.messages.calls[0]
        assert call["model"] == "claude-haiku-5-5"
        assert call["output_config"]["format"]["type"] == "json_schema"
        assert "temperature" not in call
        assert r.status == STATUS_OK
        assert r.judge_model == "claude-haiku-5-5"
        assert r.failure_modes == ["partial_constraint"]  # deduplicated
        assert r.cost_usd > 0
        assert 0.79 < r.weighted_score < 0.81

    def test_scores_are_clamped_and_bad_best_index_dropped(self):
        def payload(kw):
            p = _ok_payload(kw)
            for v in p["scores"].values():
                v["score"] = 3.0
            p["best_result_index"] = 99
            return p

        judge = SearchJudge(_settings(), client=FakeAnthropic(payload))
        r = asyncio.run(judge.evaluate("q1", "x", "negation", _response()))
        assert all(s.score == 1.0 for s in r.dimension_scores.values())
        assert r.best_result_index is None

    def test_ground_truth_reaches_prompt(self):
        client = FakeAnthropic(_ok_payload)
        judge = SearchJudge(_settings(), client=client)
        gt = {"strategy": "must_not_contain", "must_not_contain_keywords": ["AI", "machine learning"]}
        asyncio.run(judge.evaluate("q1", "x", "negation", _response(), ground_truth=gt))
        assert "machine learning" in user_prompt(client.messages.calls[0])

    def test_batch_is_concurrent_and_ordered(self):
        client = FakeAnthropic(_ok_payload, delay=0.2)
        judge = SearchJudge(_settings(), client=client)
        batch = [
            {"query_id": f"q{i}", "query_text": "x", "category": "negation",
             "search_response": _response()}
            for i in range(5)
        ]
        start = time.perf_counter()
        results = asyncio.run(judge.evaluate_batch(batch, max_concurrent=5))
        elapsed = time.perf_counter() - start
        assert [r.query_id for r in results] == [f"q{i}" for i in range(5)]
        assert client.messages.max_in_flight == 5
        assert elapsed < 0.6  # serial would take >= 1.0s


class TestPrompts:
    def test_schema_requires_every_dimension(self):
        schema = build_judge_schema(["relevance", "negation_respect"])
        assert schema["properties"]["scores"]["required"] == ["relevance", "negation_respect"]

    @pytest.mark.parametrize(
        "gt,expected",
        [
            ({"must_not_contain_keywords": ["AI"]}, "AI"),
            ({"numeric_min": 45, "numeric_max": 55, "numeric_field": "employees"}, "[45, 55]"),
            ({"expected_entity": "Paris, Texas"}, "Paris, Texas"),
            ({"expected_language": "de"}, "de"),
            ({"date_min": "2025-01-01", "date_max": "2025-01-31"}, "2025-01-31"),
        ],
    )
    def test_format_ground_truth_uses_real_fields(self, gt, expected):
        assert expected in format_ground_truth(gt)
        assert "N/A" not in format_ground_truth(gt)

    def test_prompt_without_ground_truth_has_no_criteria_section(self):
        prompt = build_evaluation_prompt("q", "negation", [], ["relevance"], None)
        assert "Verification Criteria" not in prompt
