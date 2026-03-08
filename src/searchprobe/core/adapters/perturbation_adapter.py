"""Adapter wrapping PerturbationEngine to conform to the Analyzer protocol."""

from __future__ import annotations

from typing import Any

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType
from searchprobe.providers.base import SearchProvider


class PerturbationAdapter:
    """Wraps PerturbationEngine as an Analyzer-protocol-compatible module."""

    name = "perturbation"

    def __init__(
        self,
        provider: SearchProvider,
        operators: list[str] | None = None,
        num_results: int = 10,
    ) -> None:
        self.provider = provider
        self.operators = operators
        self.num_results = num_results

    async def analyze(
        self,
        run_id: str | None = None,
        categories: list[str] | None = None,
        context: Any | None = None,
    ) -> AnalysisResult:
        from searchprobe.perturbation.engine import PerturbationEngine
        from searchprobe.perturbation.operators import PerturbationType

        # Build operator list
        ops = None
        if self.operators:
            ops = [PerturbationType(o) for o in self.operators]

        engine = PerturbationEngine(
            provider=self.provider,
            operators=ops,
            num_results=self.num_results,
        )

        # Build queries from context or DB
        queries: list[dict[str, Any]] = []
        if context and hasattr(context, "metadata") and "queries" in context.metadata:
            queries = context.metadata["queries"]
        else:
            # Fallback: use taxonomy example queries
            from searchprobe.queries.taxonomy import CATEGORY_METADATA, AdversarialCategory
            for cat in AdversarialCategory:
                if categories and cat.value not in categories:
                    continue
                meta = CATEGORY_METADATA[cat]
                for q in meta.example_queries[:2]:
                    queries.append({"text": q, "category": cat.value})

        report = await engine.analyze_queries(queries)

        # Build signals
        signals: list[Signal] = []
        for cat, stability in report.stability_by_category.items():
            signals.append(Signal(
                type=SignalType.STABILITY_MEASURED,
                source=self.name,
                category=cat,
                data={
                    "mean_jaccard": stability,
                    "provider": report.provider,
                },
            ))

        # Find least stable operator
        least_stable_op = min(
            report.stability_by_operator.items(),
            key=lambda x: x[1],
        )[0] if report.stability_by_operator else None

        details = [
            {
                "original_query": a.original_query,
                "perturbed_query": a.perturbed_query,
                "operator": a.perturbation_type,
                "category": a.category,
                "jaccard": a.jaccard_similarity,
                "rbo": a.rbo_score,
            }
            for a in report.analyses
        ]

        return AnalysisResult(
            analysis_type=self.name,
            run_id=run_id,
            categories=list(report.stability_by_category.keys()),
            summary={
                "total_perturbations": len(report.analyses),
                "mean_jaccard": round(report.mean_jaccard, 3),
                "mean_rbo": round(report.mean_rbo, 3),
                "least_stable_operator": least_stable_op,
                "stability_by_operator": report.stability_by_operator,
                "stability_by_category": report.stability_by_category,
            },
            details=details,
            signals=signals,
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_provider": True,
            "operators": self.operators or ["word_delete", "word_swap", "synonym_replace"],
        }
