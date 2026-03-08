"""Adapter wrapping AdversarialQueryOptimizer to conform to the Analyzer protocol."""

from __future__ import annotations

from typing import Any

from searchprobe.core.protocols import AnalysisResult
from searchprobe.core.signals import Signal, SignalType


class EvolutionAdapter:
    """Wraps AdversarialQueryOptimizer as an Analyzer-protocol-compatible module."""

    name = "evolution"

    def __init__(
        self,
        provider: Any | None = None,
        judge: Any | None = None,
        fitness_mode: str = "embedding_sim",
        generations: int = 20,
        population_size: int = 30,
        budget: float = 5.0,
        seed_queries: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.judge = judge
        self.fitness_mode = fitness_mode
        self.generations = generations
        self.population_size = population_size
        self.budget = budget
        self.seed_queries = seed_queries or []

    async def analyze(
        self,
        run_id: str | None = None,
        categories: list[str] | None = None,
        context: Any | None = None,
    ) -> AnalysisResult:
        from searchprobe.adversarial.fitness import FitnessEvaluator
        from searchprobe.adversarial.models import EvolutionConfig
        from searchprobe.adversarial.optimizer import AdversarialQueryOptimizer

        # Use context to focus on vulnerable categories
        target_categories = categories or []
        if context and hasattr(context, "vulnerable_categories"):
            target_categories = target_categories or context.vulnerable_categories

        # Get seed queries
        seeds = list(self.seed_queries)
        if not seeds:
            from searchprobe.queries.taxonomy import get_example_queries
            seeds = get_example_queries()[:self.population_size]

        config = EvolutionConfig(
            population_size=self.population_size,
            generations=self.generations,
            budget_limit=self.budget,
            fitness_mode=self.fitness_mode,
            seed_queries=seeds,
            target_categories=target_categories,
        )

        fitness_evaluator = FitnessEvaluator(
            mode=self.fitness_mode,
            provider=self.provider,
            judge=self.judge,
        )

        optimizer = AdversarialQueryOptimizer(config, fitness_evaluator)
        result = await optimizer.optimize()

        # Build details and signals
        details = [ind.to_dict() for ind in result.best_individuals[:10]]

        signals = [Signal(
            type=SignalType.EVOLUTION_COMPLETED,
            source=self.name,
            data={
                "generations": result.generations_completed,
                "total_evaluations": result.total_evaluations,
                "best_fitness": result.best_individuals[0].fitness if result.best_individuals else 0.0,
                "fitness_mode": self.fitness_mode,
            },
        )]

        best_fitness = result.best_individuals[0].fitness if result.best_individuals else 0.0

        return AnalysisResult(
            analysis_type=self.name,
            run_id=run_id,
            categories=target_categories,
            summary={
                "generations": result.generations_completed,
                "total_evaluations": result.total_evaluations,
                "best_fitness": round(best_fitness, 3),
                "fitness_mode": self.fitness_mode,
                "fitness_history": result.fitness_history,
            },
            details=details,
            signals=signals,
        )

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fitness_modes": ["embedding_sim", "llm_judge", "cross_encoder"],
            "requires_provider": self.fitness_mode != "embedding_sim",
        }
