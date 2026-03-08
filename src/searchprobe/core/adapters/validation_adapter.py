"""Adapter wrapping CrossEncoderValidator to conform to the Analyzer protocol."""

from __future__ import annotations

from typing import Any

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType
from searchprobe.providers.base import SearchProvider


class ValidationAdapter:
    """Wraps CrossEncoderValidator as an Analyzer-protocol-compatible module."""

    name = "validation"

    def __init__(
        self,
        provider: SearchProvider,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.device = device

    async def analyze(
        self,
        run_id: str | None = None,
        categories: list[str] | None = None,
        context: Any | None = None,
    ) -> AnalysisResult:
        from searchprobe.providers.models import SearchRequest
        from searchprobe.validation.cross_encoder import CrossEncoderValidator

        validator = CrossEncoderValidator(
            model_name=self.model_name, device=self.device
        )

        # Build queries from context or taxonomy
        queries: list[dict[str, Any]] = []
        if context and hasattr(context, "metadata") and "queries" in context.metadata:
            queries = context.metadata["queries"]
        else:
            from searchprobe.queries.taxonomy import CATEGORY_METADATA, AdversarialCategory
            for cat in AdversarialCategory:
                if categories and cat.value not in categories:
                    continue
                meta = CATEGORY_METADATA[cat]
                for q in meta.example_queries[:2]:
                    queries.append({"text": q, "category": cat.value, "id": q[:8]})

        details: list[dict[str, Any]] = []
        signals: list[Signal] = []
        improvements_by_cat: dict[str, list[float]] = {}
        provider_improvements: list[float] = []

        for q in queries:
            # Search
            request = SearchRequest(
                query=q["text"], num_results=10, include_content=True
            )
            response = await self.provider.search(request)
            if not response.success:
                continue

            results_dicts = [
                {"title": r.title, "url": str(r.url), "snippet": r.snippet, "content": r.content}
                for r in response.results
            ]

            result = validator.validate_search_results(
                query_id=q.get("id", q["text"][:8]),
                query_text=q["text"],
                category=q.get("category", "unknown"),
                provider=response.provider,
                results=results_dicts,
            )

            cat = q.get("category", "unknown")
            improvements_by_cat.setdefault(cat, []).append(result.ndcg_improvement)
            provider_improvements.append(result.ndcg_improvement)

            details.append({
                "query": q["text"],
                "category": cat,
                "provider": response.provider,
                "original_ndcg": result.original_ndcg,
                "reranked_ndcg": result.reranked_ndcg,
                "ndcg_improvement": result.ndcg_improvement,
                "kendall_tau": result.kendall_tau,
            })

            if result.ndcg_improvement > 0.1:
                signals.append(Signal(
                    type=SignalType.EMBEDDING_GAP_FOUND,
                    source=self.name,
                    category=cat,
                    data={
                        "ndcg_improvement": result.ndcg_improvement,
                        "query": q["text"],
                        "provider": response.provider,
                    },
                ))

        mean_improvement = (
            sum(provider_improvements) / len(provider_improvements)
            if provider_improvements else 0.0
        )
        max_improvement = max(provider_improvements) if provider_improvements else 0.0

        return AnalysisResult(
            analysis_type=self.name,
            run_id=run_id,
            categories=list(improvements_by_cat.keys()),
            summary={
                "mean_ndcg_improvement": round(mean_improvement, 3),
                "max_ndcg_improvement": round(max_improvement, 3),
                "queries_validated": len(details),
                "improvements_by_category": {
                    cat: round(sum(imps) / len(imps), 3)
                    for cat, imps in improvements_by_cat.items()
                },
            },
            details=details,
            signals=signals,
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_provider": True,
            "requires_ml": True,
            "cross_encoder_model": self.model_name,
        }
