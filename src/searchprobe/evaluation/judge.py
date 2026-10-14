"""List-level LLM-as-judge for search results.

This judge scores a whole top-k list per dimension. It is kept for the
legacy ``evaluate`` flow; the per-document constraint judge in
``searchprobe.judging`` is the primary instrument because it yields
document-level labels (nDCG, violation@k, pooling).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from searchprobe.config import Settings, get_settings
from searchprobe.core.exceptions import ConfigurationError
from searchprobe.evaluation.dimensions import (
    EvaluationDimension,
    calculate_weighted_score,
    get_active_dimensions,
)
from searchprobe.evaluation.prompts import (
    JUDGE_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_evaluation_prompt,
)
from searchprobe.intelligence.taxonomy import JUDGE_FAILURE_MODES
from searchprobe.llm import get_async_client, structured_call
from searchprobe.providers.models import SearchResponse

# Evaluation statuses. Only ``ok`` and ``no_results`` measure retrieval quality.
STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_SEARCH_ERROR = "search_error"
STATUS_JUDGE_ERROR = "judge_error"


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
    status: str = STATUS_OK
    judge_model: str | None = None
    prompt_version: str = PROMPT_VERSION
    cost_usd: float = 0.0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_response: str | None = None
    error: str | None = None

    @property
    def counts_for_quality(self) -> bool:
        """Whether this result measures retrieval quality (not infrastructure)."""
        return self.status in (STATUS_OK, STATUS_NO_RESULTS)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_id": self.query_id,
            "provider": self.provider,
            "search_mode": self.search_mode,
            "category": self.category,
            "status": self.status,
            "dimension_scores": {
                k: {"score": v.score, "justification": v.justification}
                for k, v in self.dimension_scores.items()
            },
            "weighted_score": self.weighted_score,
            "failure_modes": self.failure_modes,
            "best_result_index": self.best_result_index,
            "overall_assessment": self.overall_assessment,
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "cost_usd": self.cost_usd,
            "evaluated_at": self.evaluated_at.isoformat(),
            "error": self.error,
        }


def build_judge_schema(dimensions: list[str]) -> dict[str, Any]:
    """JSON schema for the list-level judge's structured output."""
    dim_schema = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "justification": {"type": "string"},
        },
        "required": ["score", "justification"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": dict.fromkeys(dimensions, dim_schema),
                "required": list(dimensions),
                "additionalProperties": False,
            },
            "failure_modes": {
                "type": "array",
                "items": {"type": "string", "enum": JUDGE_FAILURE_MODES},
            },
            "best_result_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "overall_assessment": {"type": "string"},
        },
        "required": ["scores", "failure_modes", "best_result_index", "overall_assessment"],
        "additionalProperties": False,
    }


class SearchJudge:
    """LLM-based judge for evaluating search result quality."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the judge.

        Args:
            settings: Application settings (``judge_model``/``judge_effort`` are honored).
            model: Override the judge model.
            client: Pre-built async client (tests inject fakes here).
        """
        self.settings = settings or get_settings()
        self.model = model or self.settings.judge_model
        self.effort = self.settings.judge_effort

        if client is not None:
            self.client = client
        else:
            if not self.settings.has_anthropic_configured():
                raise ConfigurationError(
                    "Anthropic credentials required for evaluation. "
                    "Set SEARCHPROBE_ANTHROPIC_API_KEY in .env, or enable Vertex AI with "
                    "SEARCHPROBE_USE_VERTEX_AI=true and SEARCHPROBE_VERTEX_PROJECT_ID"
                )
            self.client = get_async_client(self.settings)

    def _base_result(self, search_response: SearchResponse, query_id: str, category: str):
        return {
            "query_id": query_id,
            "provider": search_response.provider,
            "search_mode": search_response.search_mode,
            "category": category,
            "judge_model": self.model,
        }

    async def evaluate(
        self,
        query_id: str,
        query_text: str,
        category: str,
        search_response: SearchResponse,
        ground_truth: dict | None = None,
    ) -> EvaluationResult:
        """Evaluate search results for a query."""
        base = self._base_result(search_response, query_id, category)

        if search_response.error:
            return EvaluationResult(
                **base,
                dimension_scores={},
                weighted_score=0.0,
                failure_modes=["search_failed"],
                best_result_index=None,
                overall_assessment=search_response.error,
                status=STATUS_SEARCH_ERROR,
                error=search_response.error,
            )
        if not search_response.results:
            return EvaluationResult(
                **base,
                dimension_scores={},
                weighted_score=0.0,
                failure_modes=["no_results"],
                best_result_index=None,
                overall_assessment="No results returned",
                status=STATUS_NO_RESULTS,
            )

        dimensions = [d.value for d in get_active_dimensions(category)]
        results_dicts = [
            {"title": r.title, "url": str(r.url), "snippet": r.snippet, "content": r.content}
            for r in search_response.results
        ]
        prompt = build_evaluation_prompt(
            query=query_text,
            category=category,
            results=results_dicts,
            dimensions=dimensions,
            ground_truth=ground_truth,
        )

        try:
            result = await structured_call(
                self.client,
                model=self.model,
                system=JUDGE_SYSTEM_PROMPT,
                prompt=prompt,
                schema=build_judge_schema(dimensions),
                effort=self.effort,
            )
        except Exception as e:  # any judge failure (LLMError, API, network) is infra
            return EvaluationResult(
                **base,
                dimension_scores={},
                weighted_score=0.0,
                failure_modes=["judge_failed"],
                best_result_index=None,
                overall_assessment="",
                status=STATUS_JUDGE_ERROR,
                error=f"Evaluation failed for query {query_id}: {e}",
            )

        data = result.data
        dimension_scores: dict[str, DimensionScore] = {}
        for dim_name, score_data in data.get("scores", {}).items():
            try:
                dim_enum = EvaluationDimension(dim_name)
            except ValueError:
                continue
            score = min(1.0, max(0.0, float(score_data.get("score", 0.0))))
            dimension_scores[dim_name] = DimensionScore(
                dimension=dim_enum,
                score=score,
                justification=score_data.get("justification", ""),
            )

        weighted_score = calculate_weighted_score(
            {EvaluationDimension(k): v.score for k, v in dimension_scores.items()}, category
        )
        best = data.get("best_result_index")
        if not isinstance(best, int) or not 1 <= best <= len(results_dicts):
            best = None

        return EvaluationResult(
            **base,
            dimension_scores=dimension_scores,
            weighted_score=weighted_score,
            failure_modes=list(dict.fromkeys(data.get("failure_modes", []))),
            best_result_index=best,
            overall_assessment=data.get("overall_assessment", ""),
            status=STATUS_OK,
            cost_usd=result.cost_usd,
            raw_response=result.raw_text,
        )

    async def evaluate_batch(
        self,
        evaluations: list[dict],
        max_concurrent: int = 5,
        on_complete: Callable[[EvaluationResult, int], None] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple query results concurrently, preserving input order."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(index: int, item: dict) -> EvaluationResult:
            async with semaphore:
                result = await self.evaluate(
                    query_id=item["query_id"],
                    query_text=item["query_text"],
                    category=item["category"],
                    search_response=item["search_response"],
                    ground_truth=item.get("ground_truth"),
                )
            if on_complete:
                on_complete(result, index)
            return result

        return list(await asyncio.gather(*[_bounded(i, e) for i, e in enumerate(evaluations)]))
