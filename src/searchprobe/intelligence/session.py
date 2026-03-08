"""Research session DAG -- chains analyses with data flowing between stages.

This is the core "intelligent pipeline" feature. Instead of running analyses
independently, a ResearchSession chains them with a SharedContext that
accumulates findings. Geometry identifies vulnerable categories, evolution
targets them, perturbation validates robustness, and correlation discovers
cross-module patterns.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalBus, SignalType

logger = logging.getLogger(__name__)


@dataclass
class SharedContext:
    """Accumulates results across analysis stages."""

    run_id: str | None = None
    signals: list[Signal] = field(default_factory=list)
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    vulnerable_categories: list[str] = field(default_factory=list)
    stable_categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, name: str, result: AnalysisResult) -> None:
        """Add an analysis result and update category intelligence from signals."""
        self.results[name] = result
        self.signals.extend(result.signals)

        # Update vulnerable/stable category lists from signals
        for signal in result.signals:
            if signal.type == SignalType.VULNERABILITY_DETECTED and signal.category:
                vuln_score = signal.data.get("vulnerability_score", 0)
                if vuln_score > 0.5 and signal.category not in self.vulnerable_categories:
                    self.vulnerable_categories.append(signal.category)

            if signal.type == SignalType.STABILITY_MEASURED and signal.category:
                stability = signal.data.get("mean_jaccard", 0)
                if stability > 0.7 and signal.category not in self.stable_categories:
                    self.stable_categories.append(signal.category)


class AnalysisStage(Protocol):
    """Protocol for a stage in the research session DAG."""

    name: str
    depends_on: list[str]

    async def execute(self, context: SharedContext) -> AnalysisResult: ...


class ResearchSession:
    """Orchestrates a DAG of analysis stages.

    Usage::

        session = ResearchSession.from_profile(profile, settings)
        context = await session.run(progress_callback=my_callback)
        # context.results contains all analysis results
        # context.vulnerable_categories populated by geometry
    """

    def __init__(self, signal_bus: SignalBus | None = None) -> None:
        self.stages: dict[str, AnalysisStage] = {}
        self.signal_bus = signal_bus or SignalBus()
        self.context = SharedContext()

    def add_stage(self, stage: AnalysisStage) -> None:
        """Add a stage to the session."""
        self.stages[stage.name] = stage

    def _topological_sort(self) -> list[str]:
        """Resolve execution order from dependency graph.

        Returns:
            Stage names in valid execution order.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        in_degree: dict[str, int] = defaultdict(int)
        adjacency: dict[str, list[str]] = defaultdict(list)

        all_stages = set(self.stages.keys())
        for name, stage in self.stages.items():
            for dep in stage.depends_on:
                if dep in all_stages:
                    adjacency[dep].append(name)
                    in_degree[name] += 1

        # BFS (Kahn's algorithm)
        queue = [name for name in self.stages if in_degree[name] == 0]
        order: list[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.stages):
            missing = set(self.stages.keys()) - set(order)
            raise ValueError(f"Circular dependency detected involving: {missing}")

        return order

    async def run(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> SharedContext:
        """Execute all stages in topological order.

        Args:
            progress_callback: Optional callback(stage_name, completed, total).

        Returns:
            SharedContext with all accumulated results.
        """
        order = self._topological_sort()
        total = len(order)

        for i, stage_name in enumerate(order):
            stage = self.stages[stage_name]

            if progress_callback:
                progress_callback(stage_name, i, total)

            self.signal_bus.emit(Signal(
                type=SignalType.STAGE_STARTED,
                source="session",
                data={"stage": stage_name, "index": i, "total": total},
            ))

            logger.info("Executing stage: %s (%d/%d)", stage_name, i + 1, total)

            try:
                result = await stage.execute(self.context)
                self.context.add_result(stage_name, result)

                self.signal_bus.emit(Signal(
                    type=SignalType.STAGE_COMPLETED,
                    source="session",
                    data={
                        "stage": stage_name,
                        "summary": result.summary,
                        "signal_count": len(result.signals),
                    },
                ))

                logger.info(
                    "Stage %s completed: %d signals emitted",
                    stage_name,
                    len(result.signals),
                )

            except Exception:
                logger.exception("Stage %s failed", stage_name)
                raise

        if progress_callback:
            progress_callback("done", total, total)

        return self.context

    @classmethod
    def from_profile(
        cls,
        profile: Any,  # SearchProbeProfile
        settings: Any,  # Settings
    ) -> ResearchSession:
        """Build a session from a TOML profile.

        Only includes stages enabled in the profile.
        Automatically wires up dependencies.
        """
        from searchprobe.core.adapters import (
            EvolutionAdapter,
            GeometryAdapter,
            PerturbationAdapter,
            ValidationAdapter,
        )
        from searchprobe.providers.registry import ProviderRegistry
        from searchprobe.providers.resilient import ResilientProvider

        session = cls()
        session.context.metadata["profile"] = profile.to_dict() if hasattr(profile, "to_dict") else {}

        # Get a provider for modules that need one
        provider = None
        if profile.providers:
            raw_provider = ProviderRegistry.get_provider(profile.providers[0], settings)
            provider = ResilientProvider(raw_provider)

        # Geometry
        if profile.run_geometry:
            geo_adapter = GeometryAdapter(
                models=profile.geometry_models,
            )
            geo_stage = _AdapterStage(
                name="geometry",
                depends_on=[],
                adapter=geo_adapter,
                categories=profile.categories or None,
            )
            session.add_stage(geo_stage)

        # Perturbation (depends on geometry for context)
        if profile.run_perturbation and provider:
            perturb_adapter = PerturbationAdapter(
                provider=provider.provider,
                operators=profile.perturbation_operators,
                num_results=profile.num_results,
            )
            deps = ["geometry"] if profile.run_geometry else []
            perturb_stage = _AdapterStage(
                name="perturbation",
                depends_on=deps,
                adapter=perturb_adapter,
                categories=profile.categories or None,
            )
            session.add_stage(perturb_stage)

        # Validation (depends on geometry)
        if profile.run_validation and provider:
            val_adapter = ValidationAdapter(
                provider=provider.provider,
            )
            deps = ["geometry"] if profile.run_geometry else []
            val_stage = _AdapterStage(
                name="validation",
                depends_on=deps,
                adapter=val_adapter,
                categories=profile.categories or None,
            )
            session.add_stage(val_stage)

        # Evolution (depends on geometry for vulnerable categories)
        if profile.run_evolution:
            evo_adapter = EvolutionAdapter(
                provider=provider.provider if provider else None,
                fitness_mode=profile.evolution_fitness_mode,
                generations=profile.evolution_generations,
                population_size=profile.evolution_population,
                budget=profile.evolution_budget,
            )
            deps = ["geometry"] if profile.run_geometry else []
            evo_stage = _AdapterStage(
                name="evolution",
                depends_on=deps,
                adapter=evo_adapter,
                categories=profile.categories or None,
            )
            session.add_stage(evo_stage)

        # Correlation (always last, depends on everything)
        all_stage_names = list(session.stages.keys())
        if all_stage_names:
            corr_stage = _CorrelationStage(
                name="correlation",
                depends_on=all_stage_names,
            )
            session.add_stage(corr_stage)

        return session


@dataclass
class _AdapterStage:
    """Wraps an adapter as an AnalysisStage."""

    name: str
    depends_on: list[str]
    adapter: Any  # Analyzer protocol
    categories: list[str] | None = None

    async def execute(self, context: SharedContext) -> AnalysisResult:
        return await self.adapter.analyze(
            run_id=context.run_id,
            categories=self.categories,
            context=context,
        )


@dataclass
class _CorrelationStage:
    """Runs the correlation engine as the final stage."""

    name: str = "correlation"
    depends_on: list[str] = field(default_factory=list)

    async def execute(self, context: SharedContext) -> AnalysisResult:
        from searchprobe.intelligence.correlation import CorrelationEngine

        engine = CorrelationEngine()
        profiles = engine.generate_profiles(context)

        details = [p.to_dict() for p in profiles]

        return AnalysisResult(
            analysis_type="correlation",
            run_id=context.run_id,
            categories=[p.category for p in profiles],
            summary={
                "total_categories": len(profiles),
                "high_risk_categories": len([p for p in profiles if p.risk_level() == "HIGH"]),
                "medium_risk_categories": len([p for p in profiles if p.risk_level() == "MEDIUM"]),
                "low_risk_categories": len([p for p in profiles if p.risk_level() == "LOW"]),
            },
            details=details,
        )
