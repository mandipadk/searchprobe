"""Data models for embedding geometry analysis."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EmbeddingPair:
    """A pair of queries for adversarial embedding analysis."""

    query_a: str
    query_b: str
    category: str
    expected_relationship: str  # "adversarial", "same_topic", "random"
    description: str = ""


@dataclass
class SimilarityResult:
    """Result of computing similarity between an embedding pair."""

    pair: EmbeddingPair
    cosine_similarity: float
    angular_distance: float
    model_name: str


@dataclass
class CategoryGeometryProfile:
    """Geometric profile for a single adversarial category."""

    category: str
    model_name: str
    adversarial_similarities: list[float] = field(default_factory=list)
    baseline_similarities: list[float] = field(default_factory=list)
    random_similarities: list[float] = field(default_factory=list)
    mean_adversarial_sim: float = 0.0
    mean_baseline_sim: float = 0.0
    mean_random_sim: float = 0.0
    collapse_ratio: float = 0.0  # adversarial_sim / baseline_sim (>1 = collapse)
    vulnerability_score: float = 0.0  # 0-1, higher = more vulnerable
    intrinsic_dimensionality: float = 0.0
    isotropy_score: float = 0.0
    pair_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "category": self.category,
            "model_name": self.model_name,
            "adversarial_similarity": self.mean_adversarial_sim,
            "baseline_similarity": self.mean_baseline_sim,
            "collapse_ratio": self.collapse_ratio,
            "vulnerability_score": self.vulnerability_score,
            "intrinsic_dimensionality": self.intrinsic_dimensionality,
            "isotropy_score": self.isotropy_score,
            "pair_details": {
                "adversarial": self.adversarial_similarities,
                "baseline": self.baseline_similarities,
                "random": self.random_similarities,
                "details": self.pair_details,
            },
        }


@dataclass
class GeometryReport:
    """Complete geometry analysis report across models and categories."""

    models: list[str]
    profiles: dict[str, dict[str, CategoryGeometryProfile]]  # model -> category -> profile
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_vulnerability_matrix(self) -> dict[str, dict[str, float]]:
        """Get model x category vulnerability scores as a matrix.

        Returns:
            Dict of model -> category -> vulnerability_score
        """
        matrix: dict[str, dict[str, float]] = {}
        for model, categories in self.profiles.items():
            matrix[model] = {}
            for category, profile in categories.items():
                matrix[model][category] = profile.vulnerability_score
        return matrix

    def get_most_vulnerable_categories(self, model: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Get the most vulnerable categories for a model.

        Args:
            model: Model name
            top_n: Number of top categories to return

        Returns:
            List of (category, vulnerability_score) tuples, sorted descending
        """
        if model not in self.profiles:
            return []
        categories = self.profiles[model]
        sorted_cats = sorted(
            categories.items(),
            key=lambda x: x[1].vulnerability_score,
            reverse=True,
        )
        return [(cat, prof.vulnerability_score) for cat, prof in sorted_cats[:top_n]]
