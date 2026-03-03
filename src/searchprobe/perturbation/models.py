"""Data models for perturbation analysis."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensitivityMap:
    """Word-level sensitivity map for a query.

    Shows which words are "load-bearing" for retrieval — removing or changing
    them causes the most disruption to search results.
    """

    query: str
    word_scores: dict[str, float]  # word -> sensitivity score [0, 1]
    provider: str
    category: str

    def get_most_sensitive_words(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Get the words with highest sensitivity (most load-bearing)."""
        sorted_words = sorted(
            self.word_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_words[:top_n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "word_scores": self.word_scores,
            "provider": self.provider,
            "category": self.category,
        }


@dataclass
class PerturbationAnalysis:
    """Analysis of a single perturbation operation on a query."""

    original_query: str
    perturbed_query: str
    perturbation_type: str
    perturbation_detail: str  # What specifically was changed
    provider: str
    category: str
    jaccard_similarity: float  # Result set overlap
    rbo_score: float  # Rank-Biased Overlap
    original_urls: list[str] = field(default_factory=list)
    perturbed_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "perturbed_query": self.perturbed_query,
            "perturbation_type": self.perturbation_type,
            "perturbation_detail": self.perturbation_detail,
            "provider": self.provider,
            "category": self.category,
            "jaccard_similarity": self.jaccard_similarity,
            "rbo_score": self.rbo_score,
        }


@dataclass
class PerturbationReport:
    """Aggregate perturbation analysis report."""

    provider: str
    analyses: list[PerturbationAnalysis] = field(default_factory=list)
    sensitivity_maps: list[SensitivityMap] = field(default_factory=list)
    mean_jaccard: float = 0.0
    mean_rbo: float = 0.0
    stability_by_operator: dict[str, float] = field(default_factory=dict)
    stability_by_category: dict[str, float] = field(default_factory=dict)
