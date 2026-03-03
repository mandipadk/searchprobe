"""Embedding gap analysis — aggregate metrics across categories."""

import numpy as np

from searchprobe.validation.models import EmbeddingGapAnalysis, ValidationResult


class EmbeddingGapAnalyzer:
    """Analyzes the embedding gap across categories and providers.

    The "embedding gap" measures how much relevance quality is lost by using
    bi-encoder (embedding similarity) search instead of cross-encoder (joint)
    scoring. A large gap means the category is poorly served by embedding search.
    """

    def analyze_by_category(
        self,
        results: list[ValidationResult],
    ) -> dict[str, dict[str, EmbeddingGapAnalysis]]:
        """Aggregate validation results by category and provider.

        Args:
            results: List of ValidationResult objects

        Returns:
            Nested dict: category -> provider -> EmbeddingGapAnalysis
        """
        # Group by (category, provider)
        grouped: dict[str, dict[str, list[ValidationResult]]] = {}

        for result in results:
            if result.category not in grouped:
                grouped[result.category] = {}
            if result.provider not in grouped[result.category]:
                grouped[result.category][result.provider] = []
            grouped[result.category][result.provider].append(result)

        # Compute aggregates
        analysis: dict[str, dict[str, EmbeddingGapAnalysis]] = {}

        for category, providers in grouped.items():
            analysis[category] = {}
            for provider, provider_results in providers.items():
                improvements = [r.ndcg_improvement for r in provider_results]
                taus = [r.kendall_tau for r in provider_results]

                mean_improvement = float(np.mean(improvements))
                median_improvement = float(np.median(improvements))
                mean_tau = float(np.mean(taus))

                analysis[category][provider] = EmbeddingGapAnalysis(
                    category=category,
                    provider=provider,
                    mean_ndcg_improvement=mean_improvement,
                    median_ndcg_improvement=median_improvement,
                    mean_kendall_tau=mean_tau,
                    n_queries=len(provider_results),
                    gap_severity=_classify_gap(mean_improvement),
                    validation_results=provider_results,
                )

        return analysis

    def get_category_ranking(
        self,
        results: list[ValidationResult],
    ) -> list[tuple[str, float]]:
        """Rank categories by embedding gap severity.

        Args:
            results: List of ValidationResult objects

        Returns:
            List of (category, mean_ndcg_improvement) sorted by gap size
        """
        category_improvements: dict[str, list[float]] = {}

        for result in results:
            if result.category not in category_improvements:
                category_improvements[result.category] = []
            category_improvements[result.category].append(result.ndcg_improvement)

        ranking = [
            (cat, float(np.mean(improvements)))
            for cat, improvements in category_improvements.items()
        ]

        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def get_provider_robustness(
        self,
        results: list[ValidationResult],
    ) -> dict[str, dict[str, float]]:
        """Compute per-provider robustness metrics.

        Args:
            results: List of ValidationResult objects

        Returns:
            Dict of provider -> {"mean_tau": ..., "mean_improvement": ..., "worst_category": ...}
        """
        provider_results: dict[str, list[ValidationResult]] = {}

        for result in results:
            if result.provider not in provider_results:
                provider_results[result.provider] = []
            provider_results[result.provider].append(result)

        robustness: dict[str, dict[str, float]] = {}
        for provider, prov_results in provider_results.items():
            taus = [r.kendall_tau for r in prov_results]
            improvements = [r.ndcg_improvement for r in prov_results]

            robustness[provider] = {
                "mean_kendall_tau": float(np.mean(taus)),
                "mean_ndcg_improvement": float(np.mean(improvements)),
                "max_ndcg_improvement": float(np.max(improvements)) if improvements else 0.0,
                "n_queries": len(prov_results),
            }

        return robustness


def _classify_gap(mean_improvement: float) -> str:
    """Classify the embedding gap severity.

    Args:
        mean_improvement: Mean NDCG improvement from reranking

    Returns:
        Severity classification
    """
    if mean_improvement >= 0.3:
        return "severe"
    elif mean_improvement >= 0.15:
        return "significant"
    elif mean_improvement >= 0.05:
        return "moderate"
    else:
        return "minimal"
