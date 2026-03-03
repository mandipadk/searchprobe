"""Core embedding geometry analyzer using sentence-transformers."""

from typing import Any

import numpy as np

from searchprobe.geometry.metrics import (
    angular_distance,
    cosine_similarity,
    isotropy_score,
    local_intrinsic_dimensionality,
)
from searchprobe.geometry.models import (
    CategoryGeometryProfile,
    EmbeddingPair,
    GeometryReport,
    SimilarityResult,
)
from searchprobe.geometry.pairs import (
    get_adversarial_pairs,
    get_baseline_pairs,
    get_random_pairs,
)
from searchprobe.geometry.vulnerability import (
    compute_collapse_ratio,
    compute_vulnerability_score,
)
from searchprobe.queries.taxonomy import AdversarialCategory


DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]


class EmbeddingGeometryAnalyzer:
    """Analyzes geometric properties of embedding spaces to explain search failures.

    This is the core research module. Instead of just testing THAT search fails,
    it measures WHY by analyzing the geometric properties of the embedding space.

    Key insight: If cos(embed("companies in AI"), embed("companies NOT in AI")) = 0.96,
    then ANY embedding-based retrieval system will fail on negation.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            models: List of sentence-transformer model names to analyze.
                   Defaults to ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_names = models or DEFAULT_MODELS
        self.device = device
        self._models: dict[str, Any] = {}

    def _load_model(self, model_name: str) -> Any:
        """Lazy-load a sentence-transformer model.

        Args:
            model_name: HuggingFace model name

        Returns:
            SentenceTransformer model instance
        """
        if model_name not in self._models:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self._models[model_name] = SentenceTransformer(model_name, **kwargs)
        return self._models[model_name]

    def _encode(self, model_name: str, texts: list[str]) -> np.ndarray:
        """Encode texts using a specific model.

        Args:
            model_name: Model to use
            texts: List of texts to encode

        Returns:
            Embedding matrix of shape (len(texts), embedding_dim)
        """
        model = self._load_model(model_name)
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def compute_pair_similarity(
        self,
        pair: EmbeddingPair,
        model_name: str,
    ) -> SimilarityResult:
        """Compute similarity for a single embedding pair.

        Args:
            pair: The embedding pair to analyze
            model_name: Model to use for encoding

        Returns:
            SimilarityResult with cosine similarity and angular distance
        """
        embeddings = self._encode(model_name, [pair.query_a, pair.query_b])
        cos_sim = cosine_similarity(embeddings[0], embeddings[1])
        ang_dist = angular_distance(embeddings[0], embeddings[1])

        return SimilarityResult(
            pair=pair,
            cosine_similarity=cos_sim,
            angular_distance=ang_dist,
            model_name=model_name,
        )

    def analyze_category(
        self,
        category: str,
        model_name: str,
    ) -> CategoryGeometryProfile:
        """Analyze the geometric profile of a single category.

        Args:
            category: Adversarial category name
            model_name: Model to use

        Returns:
            CategoryGeometryProfile with all metrics
        """
        # Get pairs
        adversarial_pairs = get_adversarial_pairs(category)
        baseline_pairs = get_baseline_pairs(category)
        random_pairs = get_random_pairs()

        profile = CategoryGeometryProfile(
            category=category,
            model_name=model_name,
        )

        # Compute adversarial similarities
        all_adversarial_texts = []
        for pair in adversarial_pairs:
            result = self.compute_pair_similarity(pair, model_name)
            profile.adversarial_similarities.append(result.cosine_similarity)
            profile.pair_details.append({
                "query_a": pair.query_a,
                "query_b": pair.query_b,
                "similarity": result.cosine_similarity,
                "angular_distance": result.angular_distance,
                "relationship": pair.expected_relationship,
            })
            all_adversarial_texts.extend([pair.query_a, pair.query_b])

        # Compute baseline similarities
        for pair in baseline_pairs:
            result = self.compute_pair_similarity(pair, model_name)
            profile.baseline_similarities.append(result.cosine_similarity)

        # Compute random similarities
        for pair in random_pairs:
            result = self.compute_pair_similarity(pair, model_name)
            profile.random_similarities.append(result.cosine_similarity)

        # Compute aggregate metrics
        if profile.adversarial_similarities:
            profile.mean_adversarial_sim = float(np.mean(profile.adversarial_similarities))
        if profile.baseline_similarities:
            profile.mean_baseline_sim = float(np.mean(profile.baseline_similarities))
        if profile.random_similarities:
            profile.mean_random_sim = float(np.mean(profile.random_similarities))

        # Compute collapse ratio and vulnerability
        profile.collapse_ratio = compute_collapse_ratio(profile)
        profile.vulnerability_score = compute_vulnerability_score(profile)

        # Compute geometric properties from all adversarial query embeddings
        if all_adversarial_texts:
            embeddings = self._encode(model_name, all_adversarial_texts)
            profile.intrinsic_dimensionality = local_intrinsic_dimensionality(embeddings)
            profile.isotropy_score = isotropy_score(embeddings)

        return profile

    def analyze_all_categories(
        self,
        model_name: str | None = None,
        categories: list[str] | None = None,
        progress_callback: Any | None = None,
    ) -> dict[str, CategoryGeometryProfile]:
        """Analyze all adversarial categories for a model.

        Args:
            model_name: Model to use (default: first in self.model_names)
            categories: Categories to analyze (default: all 13)
            progress_callback: Optional callback(category, index, total)

        Returns:
            Dict of category -> CategoryGeometryProfile
        """
        if model_name is None:
            model_name = self.model_names[0]
        if categories is None:
            categories = [c.value for c in AdversarialCategory]

        profiles: dict[str, CategoryGeometryProfile] = {}
        total = len(categories)

        for i, category in enumerate(categories):
            if progress_callback:
                progress_callback(category, i, total)
            profiles[category] = self.analyze_category(category, model_name)

        return profiles

    def generate_report(
        self,
        categories: list[str] | None = None,
        progress_callback: Any | None = None,
    ) -> GeometryReport:
        """Generate a complete geometry analysis report across all models.

        Args:
            categories: Categories to analyze (default: all)
            progress_callback: Optional callback(model, category, model_idx, cat_idx, total)

        Returns:
            GeometryReport with all profiles
        """
        all_profiles: dict[str, dict[str, CategoryGeometryProfile]] = {}
        total_models = len(self.model_names)

        for model_idx, model_name in enumerate(self.model_names):
            if progress_callback:
                progress_callback(model_name, "", model_idx, 0, total_models)

            all_profiles[model_name] = self.analyze_all_categories(
                model_name=model_name,
                categories=categories,
            )

        return GeometryReport(
            models=self.model_names,
            profiles=all_profiles,
            metadata={
                "categories_analyzed": categories or [c.value for c in AdversarialCategory],
                "n_models": len(self.model_names),
            },
        )
