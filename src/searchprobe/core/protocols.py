"""Analysis protocol and shared result types.

Defines the unified interface that all analysis modules (geometry, perturbation,
validation, evolution) conform to via adapters. Uses typing.Protocol for
structural subtyping -- no inheritance changes to existing modules required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from searchprobe.core.signals import Signal


@dataclass
class AnalysisResult:
    """Standardized result from any analysis module."""

    analysis_type: str  # "geometry", "perturbation", "validation", "evolution"
    run_id: str | None = None
    categories: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get a metric from the summary dict."""
        return self.summary.get(key, default)

    def get_category_detail(self, category: str) -> dict[str, Any] | None:
        """Get detail dict for a specific category."""
        for detail in self.details:
            if detail.get("category") == category:
                return detail
        return None


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for analysis modules.

    Any class with a matching ``name`` attribute and ``analyze`` / ``get_capabilities``
    methods satisfies this protocol -- no inheritance required.
    """

    name: str

    async def analyze(
        self,
        run_id: str | None = None,
        categories: list[str] | None = None,
        context: Any | None = None,  # SharedContext, but avoid circular import
    ) -> AnalysisResult: ...

    def get_capabilities(self) -> dict[str, Any]: ...
