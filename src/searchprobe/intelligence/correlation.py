"""Cross-module correlation engine.

Discovers statistical patterns across analysis dimensions and generates
actionable CategoryIntelligenceProfiles with risk scores and recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from searchprobe.intelligence.models import CategoryIntelligenceProfile, SignalVector
from searchprobe.intelligence.taxonomy import FailureClassifier, FailureMode

logger = logging.getLogger(__name__)


class CorrelationEngine:
    """Computes cross-module correlations and generates intelligence profiles."""

    def __init__(self) -> None:
        self.classifier = FailureClassifier()

    def build_signal_vectors(
        self, context: Any  # SharedContext
    ) -> dict[str, SignalVector]:
        """Extract per-category metrics from all analysis results.

        Args:
            context: SharedContext with accumulated analysis results.

        Returns:
            Dict of category -> SignalVector.
        """
        vectors: dict[str, SignalVector] = {}
        results = context.results if hasattr(context, "results") else {}

        # Geometry: vulnerability scores
        if "geometry" in results:
            for detail in results["geometry"].details:
                cat = detail.get("category", "")
                if cat not in vectors:
                    vectors[cat] = SignalVector(category=cat)
                vectors[cat].vulnerability_score = detail.get("vulnerability_score")

        # Perturbation: stability scores
        if "perturbation" in results:
            stability_by_cat = results["perturbation"].summary.get("stability_by_category", {})
            for cat, stability in stability_by_cat.items():
                if cat not in vectors:
                    vectors[cat] = SignalVector(category=cat)
                vectors[cat].perturbation_stability = stability

        # Validation: embedding gap
        if "validation" in results:
            improvements = results["validation"].summary.get("improvements_by_category", {})
            for cat, improvement in improvements.items():
                if cat not in vectors:
                    vectors[cat] = SignalVector(category=cat)
                vectors[cat].embedding_gap = improvement

        # Evolution: best fitness per category
        if "evolution" in results:
            for detail in results["evolution"].details:
                cat = detail.get("category", "")
                if not cat:
                    continue
                if cat not in vectors:
                    vectors[cat] = SignalVector(category=cat)
                current = vectors[cat].evolution_fitness or 0.0
                vectors[cat].evolution_fitness = max(current, detail.get("fitness", 0.0))

        return vectors

    def compute_correlation_matrix(
        self, vectors: dict[str, SignalVector]
    ) -> dict[str, dict[str, float | None]]:
        """Compute pairwise Spearman correlations between signal dimensions.

        Returns:
            Nested dict: dimension_a -> dimension_b -> correlation coefficient.
            None values indicate insufficient data.
        """
        dim_names = SignalVector.dimension_names()
        n_dims = len(dim_names)

        # Build matrix: rows = categories, columns = dimensions
        cats = sorted(vectors.keys())
        if len(cats) < 3:
            # Not enough categories for meaningful correlation
            return {}

        matrix = np.full((len(cats), n_dims), np.nan)
        for i, cat in enumerate(cats):
            values = vectors[cat].to_array()
            for j, v in enumerate(values):
                if v is not None:
                    matrix[i, j] = v

        correlations: dict[str, dict[str, float | None]] = {}
        for a in range(n_dims):
            correlations[dim_names[a]] = {}
            for b in range(n_dims):
                if a == b:
                    correlations[dim_names[a]][dim_names[b]] = 1.0
                    continue

                # Get valid pairs
                valid = ~np.isnan(matrix[:, a]) & ~np.isnan(matrix[:, b])
                if valid.sum() < 3:
                    correlations[dim_names[a]][dim_names[b]] = None
                    continue

                from scipy.stats import spearmanr
                rho, p_value = spearmanr(matrix[valid, a], matrix[valid, b])
                if np.isnan(rho):
                    correlations[dim_names[a]][dim_names[b]] = None
                else:
                    correlations[dim_names[a]][dim_names[b]] = round(float(rho), 3)

        return correlations

    def generate_profiles(
        self, context: Any  # SharedContext
    ) -> list[CategoryIntelligenceProfile]:
        """Generate complete intelligence profiles for all categories.

        Args:
            context: SharedContext with accumulated analysis results.

        Returns:
            List of CategoryIntelligenceProfile sorted by risk score descending.
        """
        vectors = self.build_signal_vectors(context)
        corr_matrix = self.compute_correlation_matrix(vectors)

        profiles: list[CategoryIntelligenceProfile] = []

        for cat, vector in vectors.items():
            # Compute risk score (weighted combination of available signals)
            risk = self._compute_risk(vector)

            # Get failure modes from evaluation results if available
            failure_modes = self._get_failure_modes(context, cat)

            # Extract relevant correlations
            correlations = self._extract_correlations(vector, corr_matrix)

            # Generate recommendations
            recommendations = self._generate_recommendations(cat, vector, failure_modes)

            profiles.append(CategoryIntelligenceProfile(
                category=cat,
                signal_vector=vector,
                primary_failure_modes=[m.value for m in failure_modes],
                risk_score=risk,
                correlations=correlations,
                recommendations=recommendations,
            ))

        # Sort by risk score descending
        profiles.sort(key=lambda p: p.risk_score, reverse=True)
        return profiles

    def _compute_risk(self, vector: SignalVector) -> float:
        """Compute composite risk score from available signals."""
        weights = {
            "vulnerability_score": 0.3,
            "perturbation_stability": 0.2,  # Inverted (low stability = high risk)
            "embedding_gap": 0.2,
            "evolution_fitness": 0.15,
            "evaluation_score": 0.15,  # Inverted (low score = high risk)
        }

        total_weight = 0.0
        weighted_sum = 0.0

        if vector.vulnerability_score is not None:
            weighted_sum += vector.vulnerability_score * weights["vulnerability_score"]
            total_weight += weights["vulnerability_score"]

        if vector.perturbation_stability is not None:
            # Low stability = high risk
            weighted_sum += (1.0 - vector.perturbation_stability) * weights["perturbation_stability"]
            total_weight += weights["perturbation_stability"]

        if vector.embedding_gap is not None:
            weighted_sum += min(1.0, vector.embedding_gap) * weights["embedding_gap"]
            total_weight += weights["embedding_gap"]

        if vector.evolution_fitness is not None:
            weighted_sum += vector.evolution_fitness * weights["evolution_fitness"]
            total_weight += weights["evolution_fitness"]

        if vector.evaluation_score is not None:
            # Low eval score = high risk
            weighted_sum += (1.0 - vector.evaluation_score) * weights["evaluation_score"]
            total_weight += weights["evaluation_score"]

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 3)

    def _get_failure_modes(self, context: Any, category: str) -> list[FailureMode]:
        """Extract and classify failure modes for a category."""
        results = context.results if hasattr(context, "results") else {}
        # Look for evaluation data with failure modes
        # This would come from the evaluate stage in a full session
        # For now, return based on signals
        modes: list[FailureMode] = []
        signals = context.signals if hasattr(context, "signals") else []
        for signal in signals:
            if signal.category == category:
                # Classify based on signal data
                failure_texts = signal.data.get("failure_modes", [])
                for text in failure_texts:
                    modes.extend(self.classifier.classify(text, category))

        return list(dict.fromkeys(modes))[:5]  # Deduplicate, top 5

    def _extract_correlations(
        self, vector: SignalVector, corr_matrix: dict[str, dict[str, float | None]]
    ) -> dict[str, float]:
        """Extract notable correlations for a specific vector's active dimensions."""
        correlations: dict[str, float] = {}
        if not corr_matrix:
            return correlations

        dims = SignalVector.dimension_names()
        values = vector.to_array()
        active = [dims[i] for i, v in enumerate(values) if v is not None]

        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a in corr_matrix and b in corr_matrix[a]:
                    rho = corr_matrix[a][b]
                    if rho is not None and abs(rho) > 0.5:
                        correlations[f"{a} <-> {b}"] = rho

        return correlations

    def _generate_recommendations(
        self,
        category: str,
        vector: SignalVector,
        failure_modes: list[FailureMode],
    ) -> list[str]:
        """Generate plain-text recommendations based on signals."""
        recs: list[str] = []

        if vector.vulnerability_score is not None and vector.vulnerability_score > 0.7:
            recs.append(
                f"High embedding vulnerability ({vector.vulnerability_score:.2f}): "
                f"queries in '{category}' are nearly indistinguishable from their "
                f"adversarial counterparts in embedding space."
            )

        if vector.perturbation_stability is not None and vector.perturbation_stability < 0.3:
            recs.append(
                f"Low perturbation stability ({vector.perturbation_stability:.2f}): "
                f"minor query modifications cause large result set changes."
            )

        if vector.embedding_gap is not None and vector.embedding_gap > 0.2:
            recs.append(
                f"Significant embedding gap ({vector.embedding_gap:.2f}): "
                f"cross-encoder reranking substantially improves results, "
                f"indicating bi-encoder limitations."
            )

        if FailureMode.NEGATION_COLLAPSE in failure_modes:
            recs.append(
                "Negation collapse detected: consider post-retrieval filtering "
                "with must_not_contain ground truth checks."
            )

        if FailureMode.PARTIAL_CONSTRAINT in failure_modes:
            recs.append(
                "Partial constraint satisfaction: decompose multi-constraint queries "
                "and verify each constraint independently."
            )

        if not recs:
            recs.append(f"Category '{category}' shows no critical issues.")

        return recs
