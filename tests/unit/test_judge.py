"""Tests for the LLM-as-judge evaluation module."""

from unittest.mock import MagicMock, patch

import pytest

from searchprobe.evaluation.judge import DimensionScore, EvaluationResult, SearchJudge
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class TestEvaluationResult:
    def test_to_dict(self):
        result = EvaluationResult(
            query_id="q1",
            provider="exa",
            search_mode="auto",
            category="negation",
            dimension_scores={},
            weighted_score=0.75,
            failure_modes=["partial_negation"],
            best_result_index=0,
            overall_assessment="Good results",
        )
        d = result.to_dict()
        assert d["query_id"] == "q1"
        assert d["provider"] == "exa"
        assert d["weighted_score"] == 0.75
        assert d["failure_modes"] == ["partial_negation"]
        assert "evaluated_at" in d

    def test_to_dict_with_dimension_scores(self):
        from searchprobe.evaluation.dimensions import EvaluationDimension

        result = EvaluationResult(
            query_id="q1",
            provider="exa",
            search_mode="auto",
            category="negation",
            dimension_scores={
                "relevance": DimensionScore(
                    dimension=EvaluationDimension.RELEVANCE,
                    score=0.8,
                    justification="Relevant results",
                )
            },
            weighted_score=0.8,
            failure_modes=[],
            best_result_index=0,
            overall_assessment="Good",
        )
        d = result.to_dict()
        assert d["dimension_scores"]["relevance"]["score"] == 0.8
        assert d["dimension_scores"]["relevance"]["justification"] == "Relevant results"


class TestSearchJudgeEvaluateFailedSearch:
    @patch("searchprobe.evaluation.judge.get_anthropic_client")
    @patch("searchprobe.evaluation.judge.get_settings")
    def test_evaluate_failed_search(self, mock_settings, mock_client):
        settings = MagicMock()
        settings.has_anthropic_configured.return_value = True
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()

        judge = SearchJudge(settings)

        response = SearchResponse(
            provider="exa", query="test", results=[], error="API error",
        )

        import asyncio

        result = asyncio.run(judge.evaluate(
            query_id="q1",
            query_text="test query",
            category="negation",
            search_response=response,
        ))

        assert result.weighted_score == 0.0
        assert "search_failed" in result.failure_modes

    @patch("searchprobe.evaluation.judge.get_anthropic_client")
    @patch("searchprobe.evaluation.judge.get_settings")
    def test_evaluate_empty_results(self, mock_settings, mock_client):
        settings = MagicMock()
        settings.has_anthropic_configured.return_value = True
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()

        judge = SearchJudge(settings)

        response = SearchResponse(
            provider="exa", query="test", results=[],
        )

        import asyncio

        result = asyncio.run(judge.evaluate(
            query_id="q1",
            query_text="test query",
            category="negation",
            search_response=response,
        ))

        assert result.weighted_score == 0.0
        assert "no_results" in result.failure_modes


class TestSearchJudgeParseResponse:
    @patch("searchprobe.evaluation.judge.get_anthropic_client")
    @patch("searchprobe.evaluation.judge.get_settings")
    def test_parse_valid_json(self, mock_settings, mock_client):
        settings = MagicMock()
        settings.has_anthropic_configured.return_value = True
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()

        judge = SearchJudge(settings)
        result = judge._parse_response('{"scores": {"relevance": {"score": 0.8}}}')
        assert result["scores"]["relevance"]["score"] == 0.8

    @patch("searchprobe.evaluation.judge.get_anthropic_client")
    @patch("searchprobe.evaluation.judge.get_settings")
    def test_parse_invalid_json(self, mock_settings, mock_client):
        settings = MagicMock()
        settings.has_anthropic_configured.return_value = True
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()

        judge = SearchJudge(settings)
        result = judge._parse_response("not json at all")
        assert "parse_error" in result["failure_modes"]


class TestSearchJudgeBatchConcurrency:
    @patch("searchprobe.evaluation.judge.get_anthropic_client")
    @patch("searchprobe.evaluation.judge.get_settings")
    def test_evaluate_batch_preserves_order(self, mock_settings, mock_client):
        settings = MagicMock()
        settings.has_anthropic_configured.return_value = True
        mock_settings.return_value = settings
        mock_client.return_value = MagicMock()

        judge = SearchJudge(settings)

        # All error responses to avoid actual LLM calls
        batch = [
            {
                "query_id": f"q{i}",
                "query_text": f"query {i}",
                "category": "negation",
                "search_response": SearchResponse(
                    provider="exa", query=f"query {i}", results=[], error="skip",
                ),
            }
            for i in range(5)
        ]

        import asyncio

        results = asyncio.run(judge.evaluate_batch(batch, max_concurrent=3))
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.query_id == f"q{i}"
