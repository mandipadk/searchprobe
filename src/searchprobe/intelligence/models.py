"""Shared data models for the intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalVector:
    """Multi-dimensional signal for a single adversarial category.

    Each dimension captures a different aspect of how a category performs
    across analysis modules. Used for cross-module correlation.
    """

    category: str
    vulnerability_score: float | None = None  # From geometry
    perturbation_stability: float | None = None  # From perturbation (Jaccard)
    embedding_gap: float | None = None  # From validation (NDCG improvement)
    evolution_fitness: float | None = None  # From evolution (best fitness)
    evaluation_score: float | None = None  # From LLM judge (mean score)
    failure_mode_counts: dict[str, int] = field(default_factory=dict)

    def to_array(self) -> list[float | None]:
        """Return numeric dimensions as a list (for correlation computation)."""
        return [
            self.vulnerability_score,
            self.perturbation_stability,
            self.embedding_gap,
            self.evolution_fitness,
            self.evaluation_score,
        ]

    @staticmethod
    def dimension_names() -> list[str]:
        return [
            "vulnerability_score",
            "perturbation_stability",
            "embedding_gap",
            "evolution_fitness",
            "evaluation_score",
        ]

    @property
    def completeness(self) -> float:
        """Fraction of dimensions that have values (0.0 to 1.0)."""
        values = self.to_array()
        return sum(1 for v in values if v is not None) / len(values)


@dataclass
class CategoryIntelligenceProfile:
    """Complete intelligence dossier for a single adversarial category.

    Synthesizes findings from all analysis modules into an actionable profile.
    """

    category: str
    signal_vector: SignalVector
    primary_failure_modes: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    correlations: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.risk_score >= 0.7:
            return "HIGH"
        elif self.risk_score >= 0.4:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "risk_score": round(self.risk_score, 3),
            "risk_level": self.risk_level(),
            "primary_failure_modes": self.primary_failure_modes,
            "signal_vector": {
                "vulnerability": self.signal_vector.vulnerability_score,
                "stability": self.signal_vector.perturbation_stability,
                "embedding_gap": self.signal_vector.embedding_gap,
                "evolution_fitness": self.signal_vector.evolution_fitness,
                "evaluation_score": self.signal_vector.evaluation_score,
            },
            "correlations": self.correlations,
            "recommendations": self.recommendations,
        }
