"""Adapter wrapping EmbeddingGeometryAnalyzer to conform to the Analyzer protocol."""

from __future__ import annotations

from typing import Any

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType


class GeometryAdapter:
    """Wraps EmbeddingGeometryAnalyzer as an Analyzer-protocol-compatible module."""

    name = "geometry"

    def __init__(
        self,
        models: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.models = models
        self.device = device
        self._analyzer: Any = None

    def _get_analyzer(self) -> Any:
        if self._analyzer is None:
            from searchprobe.geometry.analyzer import EmbeddingGeometryAnalyzer
            self._analyzer = EmbeddingGeometryAnalyzer(
                models=self.models, device=self.device
            )
        return self._analyzer

    async def analyze(
        self,
        run_id: str | None = None,
        categories: list[str] | None = None,
        context: Any | None = None,
    ) -> AnalysisResult:
        analyzer = self._get_analyzer()
        report = analyzer.generate_report(categories=categories)

        # Build summary and signals
        all_vuln: list[float] = []
        details: list[dict[str, Any]] = []
        signals: list[Signal] = []
        most_vulnerable: str | None = None
        max_vuln = 0.0

        for model_name, profiles in report.profiles.items():
            for cat_name, profile in profiles.items():
                vuln = profile.vulnerability_score or 0.0
                all_vuln.append(vuln)
                details.append({
                    "model": model_name,
                    "category": cat_name,
                    "vulnerability_score": vuln,
                    "collapse_ratio": profile.collapse_ratio,
                    "intrinsic_dimensionality": profile.intrinsic_dimensionality,
                    "isotropy_score": profile.isotropy_score,
                    "mean_adversarial_sim": profile.mean_adversarial_sim,
                })

                if vuln > max_vuln:
                    max_vuln = vuln
                    most_vulnerable = cat_name

                if vuln > 0.5:
                    signals.append(Signal(
                        type=SignalType.VULNERABILITY_DETECTED,
                        source=self.name,
                        category=cat_name,
                        data={
                            "vulnerability_score": vuln,
                            "model": model_name,
                            "collapse_ratio": profile.collapse_ratio,
                        },
                    ))

        mean_vuln = sum(all_vuln) / len(all_vuln) if all_vuln else 0.0
        analyzed_cats = list({d["category"] for d in details})

        return AnalysisResult(
            analysis_type=self.name,
            run_id=run_id,
            categories=analyzed_cats,
            summary={
                "models_analyzed": len(report.models),
                "mean_vulnerability": round(mean_vuln, 3),
                "most_vulnerable_category": most_vulnerable,
                "categories_analyzed": len(analyzed_cats),
            },
            details=details,
            signals=signals,
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": self.models or ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            "requires_ml": True,
        }
