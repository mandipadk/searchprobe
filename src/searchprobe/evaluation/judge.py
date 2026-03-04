"""LLM-as-judge for evaluating search results."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from searchprobe.config import Settings, get_anthropic_client, get_settings
from searchprobe.evaluation.dimensions import (
    EvaluationDimension,
    get_active_dimensions,
    calculate_weighted_score,
)
from searchprobe.evaluation.prompts import JUDGE_SYSTEM_PROMPT, build_evaluation_prompt
from searchprobe.providers.models import SearchResponse


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    justification: str


@dataclass
class EvaluationResult:
    """Complete evaluation of search results for a query."""

    query_id: str
    provider: str
    search_mode: str | None
    category: str
    dimension_scores: dict[str, DimensionScore]
    weighted_score: float
    failure_modes: list[str]
    best_result_index: int | None
    overall_assessment: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_id": self.query_id,
            "provider": self.provider,
            "search_mode": self.search_mode,
            "category": self.category,
            "dimension_scores": {
                k: {"score": v.score, "justification": v.justification}
                for k, v in self.dimension_scores.items()
            },
            "weighted_score": self.weighted_score,
            "failure_modes": self.failure_modes,
            "best_result_index": self.best_result_index,
            "overall_assessment": self.overall_assessment,
            "evaluated_at": self.evaluated_at.isoformat(),
            "error": self.error,
        }


class SearchJudge:
    """LLM-based judge for evaluating search result quality.

    Uses Claude to assess search results across multiple dimensions,
    with category-specific weighting and failure mode detection.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        """Initialize the judge.

        Args:
            settings: Application settings
            model: Anthropic model to use for evaluation
        """
        self.settings = settings or get_settings()
        self.model = model

        if not self.settings.has_anthropic_configured():
            raise ValueError(
                "Anthropic credentials required for evaluation. "
                "Set SEARCHPROBE_ANTHROPIC_API_KEY in .env, or enable Vertex AI with "
                "SEARCHPROBE_USE_VERTEX_AI=true and SEARCHPROBE_VERTEX_PROJECT_ID"
            )

        self.client = get_anthropic_client(self.settings)

    async def evaluate(
        self,
        query_id: str,
        query_text: str,
        category: str,
        search_response: SearchResponse,
        ground_truth: dict | None = None,
    ) -> EvaluationResult:
        """Evaluate search results for a query.

        Args:
            query_id: Unique query identifier
            query_text: The search query
            category: Adversarial category
            search_response: Search results to evaluate
            ground_truth: Optional ground truth for calibration

        Returns:
            EvaluationResult with scores and analysis
        """
        # Handle failed searches
        if search_response.error or not search_response.results:
            return EvaluationResult(
                query_id=query_id,
                provider=search_response.provider,
                search_mode=search_response.search_mode,
                category=category,
                dimension_scores={},
                weighted_score=0.0,
                failure_modes=["search_failed" if search_response.error else "no_results"],
                best_result_index=None,
                overall_assessment=search_response.error or "No results returned",
                error=search_response.error,
            )

        # Get dimensions for this category
        dimensions = get_active_dimensions(category)
        dimension_names = [d.value for d in dimensions]

        # Convert results to dicts
        results_dicts = [
            {
                "title": r.title,
                "url": str(r.url),
                "snippet": r.snippet,
                "content": r.content,
            }
            for r in search_response.results
        ]

        # Build evaluation prompt
        prompt = build_evaluation_prompt(
            query=query_text,
            category=category,
            results=results_dicts,
            dimensions=dimension_names,
            ground_truth=ground_truth,
        )

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_response = response.content[0].text

            # Parse JSON response
            evaluation_data = self._parse_response(raw_response)

            # Build dimension scores
            dimension_scores = {}
            for dim_name, score_data in evaluation_data.get("scores", {}).items():
                try:
                    dim_enum = EvaluationDimension(dim_name)
                    dimension_scores[dim_name] = DimensionScore(
                        dimension=dim_enum,
                        score=float(score_data.get("score", 0.0)),
                        justification=score_data.get("justification", ""),
                    )
                except ValueError:
                    # Unknown dimension, skip
                    pass

            # Calculate weighted score
            scores_dict = {
                EvaluationDimension(k): v.score for k, v in dimension_scores.items()
            }
            weighted_score = calculate_weighted_score(scores_dict, category)

            return EvaluationResult(
                query_id=query_id,
                provider=search_response.provider,
                search_mode=search_response.search_mode,
                category=category,
                dimension_scores=dimension_scores,
                weighted_score=weighted_score,
                failure_modes=evaluation_data.get("failure_modes", []),
                best_result_index=evaluation_data.get("best_result_index"),
                overall_assessment=evaluation_data.get("overall_assessment", ""),
                raw_response=raw_response,
            )

        except Exception as e:
            return EvaluationResult(
                query_id=query_id,
                provider=search_response.provider,
                search_mode=search_response.search_mode,
                category=category,
                dimension_scores={},
                weighted_score=0.0,
                failure_modes=["evaluation_failed"],
                best_result_index=None,
                overall_assessment="",
                error=str(e),
            )

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response.

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed evaluation data
        """
        from searchprobe.utils.parsing import extract_json_from_llm_response

        try:
            return extract_json_from_llm_response(response_text)
        except ValueError:
            return {
                "scores": {},
                "failure_modes": ["parse_error"],
                "best_result_index": None,
                "overall_assessment": "Failed to parse evaluation response",
            }

    async def evaluate_batch(
        self,
        evaluations: list[dict],
    ) -> list[EvaluationResult]:
        """Evaluate multiple query results.

        Args:
            evaluations: List of dicts with query_id, query_text, category,
                        search_response, and optional ground_truth

        Returns:
            List of EvaluationResults
        """
        results = []
        for eval_item in evaluations:
            result = await self.evaluate(
                query_id=eval_item["query_id"],
                query_text=eval_item["query_text"],
                category=eval_item["category"],
                search_response=eval_item["search_response"],
                ground_truth=eval_item.get("ground_truth"),
            )
            results.append(result)
        return results
