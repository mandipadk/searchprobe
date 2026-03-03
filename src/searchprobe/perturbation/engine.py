"""Perturbation engine — orchestrates perturbation, search, and stability measurement."""

import asyncio
from typing import Any, Callable

import numpy as np

from searchprobe.perturbation.models import PerturbationAnalysis, PerturbationReport, SensitivityMap
from searchprobe.perturbation.operators import PerturbationType, apply_perturbation
from searchprobe.perturbation.stability import (
    compute_sensitivity_map,
    jaccard_similarity,
    rank_biased_overlap,
)
from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse


class PerturbationEngine:
    """Orchestrates perturbation analysis for search robustness testing.

    For each query:
    1. Search with original query
    2. Apply perturbation operators to generate variants
    3. Search with each variant
    4. Measure result stability (Jaccard, RBO)
    5. Build sensitivity maps
    """

    def __init__(
        self,
        provider: SearchProvider,
        operators: list[PerturbationType] | None = None,
        max_variants_per_operator: int = 5,
        num_results: int = 10,
    ) -> None:
        """Initialize the perturbation engine.

        Args:
            provider: Search provider to test
            operators: Perturbation operators to apply (default: all)
            max_variants_per_operator: Max variants per operator
            num_results: Number of results per query
        """
        self.provider = provider
        self.operators = operators or [
            PerturbationType.WORD_DELETE,
            PerturbationType.WORD_SWAP,
            PerturbationType.SYNONYM_REPLACE,
        ]
        self.max_variants = max_variants_per_operator
        self.num_results = num_results

    async def _search(self, query: str, mode: str | None = None) -> SearchResponse:
        """Execute a search query."""
        request = SearchRequest(
            query=query,
            num_results=self.num_results,
            include_content=False,
        )
        if mode:
            request.search_mode = mode
        return await self.provider.search(request)

    def _extract_urls(self, response: SearchResponse) -> list[str]:
        """Extract URLs from a search response."""
        return [str(r.url) for r in response.results]

    async def analyze_query(
        self,
        query: str,
        category: str,
        search_mode: str | None = None,
    ) -> list[PerturbationAnalysis]:
        """Analyze a single query with all perturbation operators.

        Args:
            query: Original query text
            category: Adversarial category
            search_mode: Optional search mode

        Returns:
            List of PerturbationAnalysis results
        """
        # Search with original query
        original_response = await self._search(query, search_mode)
        original_urls = self._extract_urls(original_response)

        analyses = []

        for operator in self.operators:
            variants = apply_perturbation(query, operator, self.max_variants)

            for perturbed_query, detail in variants:
                if not perturbed_query.strip():
                    continue

                # Search with perturbed query
                perturbed_response = await self._search(perturbed_query, search_mode)
                perturbed_urls = self._extract_urls(perturbed_response)

                # Compute stability metrics
                jaccard = jaccard_similarity(
                    set(original_urls), set(perturbed_urls)
                )
                rbo = rank_biased_overlap(original_urls, perturbed_urls)

                analyses.append(
                    PerturbationAnalysis(
                        original_query=query,
                        perturbed_query=perturbed_query,
                        perturbation_type=operator.value,
                        perturbation_detail=detail,
                        provider=self.provider.name,
                        category=category,
                        jaccard_similarity=jaccard,
                        rbo_score=rbo,
                        original_urls=original_urls,
                        perturbed_urls=perturbed_urls,
                    )
                )

        return analyses

    async def analyze_queries(
        self,
        queries: list[dict[str, Any]],
        search_mode: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> PerturbationReport:
        """Analyze multiple queries and produce an aggregate report.

        Args:
            queries: List of query dicts with 'text' and 'category'
            search_mode: Optional search mode
            progress_callback: Optional callback(query_text, index, total)

        Returns:
            PerturbationReport with all analyses and sensitivity maps
        """
        all_analyses: list[PerturbationAnalysis] = []
        sensitivity_maps: list[SensitivityMap] = []
        total = len(queries)

        for i, query_dict in enumerate(queries):
            query_text = query_dict["text"]
            category = query_dict.get("category", "unknown")

            if progress_callback:
                progress_callback(query_text, i, total)

            analyses = await self.analyze_query(query_text, category, search_mode)
            all_analyses.extend(analyses)

            # Build sensitivity map from word_delete perturbations
            word_delete_analyses = [
                a for a in analyses
                if a.perturbation_type == PerturbationType.WORD_DELETE.value
            ]
            if word_delete_analyses:
                sm = compute_sensitivity_map(query_text, word_delete_analyses)
                sensitivity_maps.append(sm)

        # Compute aggregate metrics
        jaccards = [a.jaccard_similarity for a in all_analyses]
        rbos = [a.rbo_score for a in all_analyses]

        # Stability by operator
        stability_by_operator: dict[str, float] = {}
        for op in self.operators:
            op_analyses = [a for a in all_analyses if a.perturbation_type == op.value]
            if op_analyses:
                stability_by_operator[op.value] = float(
                    np.mean([a.jaccard_similarity for a in op_analyses])
                )

        # Stability by category
        stability_by_category: dict[str, float] = {}
        categories = set(a.category for a in all_analyses)
        for cat in categories:
            cat_analyses = [a for a in all_analyses if a.category == cat]
            if cat_analyses:
                stability_by_category[cat] = float(
                    np.mean([a.jaccard_similarity for a in cat_analyses])
                )

        return PerturbationReport(
            provider=self.provider.name,
            analyses=all_analyses,
            sensitivity_maps=sensitivity_maps,
            mean_jaccard=float(np.mean(jaccards)) if jaccards else 0.0,
            mean_rbo=float(np.mean(rbos)) if rbos else 0.0,
            stability_by_operator=stability_by_operator,
            stability_by_category=stability_by_category,
        )
