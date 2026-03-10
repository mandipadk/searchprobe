"""Tests for adversarial query optimizer."""

import pytest

from searchprobe.adversarial.models import EvolutionConfig, Individual, Population
from searchprobe.adversarial.optimizer import AdversarialQueryOptimizer


class FakeFitnessEvaluator:
    """Fake fitness evaluator that scores by query length."""

    def __init__(self):
        self.eval_count = 0

    async def evaluate_population(self, individuals: list[Individual]) -> list[Individual]:
        self.eval_count += len(individuals)
        for ind in individuals:
            ind.fitness = len(ind.query) / 100.0
        return individuals


class TestOptimizerInit:
    def test_initializes_population(self):
        config = EvolutionConfig(
            population_size=5,
            generations=1,
            seed_queries=["test query one", "test query two"],
            target_categories=["negation"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)
        pop = optimizer._initialize_population()
        assert pop.size == 5
        assert pop.generation == 0

    def test_initializes_with_default_seed(self):
        config = EvolutionConfig(
            population_size=3,
            generations=1,
            seed_queries=[],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)
        pop = optimizer._initialize_population()
        assert pop.size == 3


class TestOptimizerSelection:
    def test_tournament_select(self):
        config = EvolutionConfig(
            population_size=5,
            generations=1,
            tournament_size=3,
            seed_queries=["a", "b", "c", "d", "e"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)

        individuals = [
            Individual(query=f"query_{i}", fitness=float(i))
            for i in range(5)
        ]
        pop = Population(individuals=individuals, generation=0)
        selected = optimizer._tournament_select(pop)
        assert isinstance(selected, Individual)


class TestOptimizerEvolution:
    @pytest.mark.asyncio
    async def test_optimize_runs_generations(self):
        config = EvolutionConfig(
            population_size=5,
            generations=3,
            mutation_rate=0.5,
            crossover_rate=0.3,
            elitism_count=1,
            tournament_size=2,
            seed_queries=["AI companies", "healthcare startups", "tech firms"],
            target_categories=["negation"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)

        result = await optimizer.optimize()

        assert result.generations_completed == 3
        assert result.total_evaluations > 0
        assert len(result.best_individuals) > 0
        assert len(result.fitness_history) == 4  # 3 gens + final
        assert fitness.eval_count > 5  # initial pop + offspring each gen

    @pytest.mark.asyncio
    async def test_optimize_with_progress_callback(self):
        config = EvolutionConfig(
            population_size=3,
            generations=2,
            seed_queries=["test query"],
            target_categories=["negation"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)

        progress_calls = []

        def on_progress(gen, total, best, mean):
            progress_calls.append((gen, total, best, mean))

        result = await optimizer.optimize(progress_callback=on_progress)
        assert len(progress_calls) == 2  # One per generation

    @pytest.mark.asyncio
    async def test_fitness_improves_or_stable(self):
        config = EvolutionConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.8,
            elitism_count=2,
            seed_queries=["AI companies", "healthcare startups"],
            target_categories=["negation"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)

        result = await optimizer.optimize()
        # With elitism, max fitness should never decrease
        max_fitnesses = [h["max"] for h in result.fitness_history]
        for i in range(1, len(max_fitnesses)):
            assert max_fitnesses[i] >= max_fitnesses[i - 1] - 0.01  # Small tolerance

    @pytest.mark.asyncio
    async def test_bandit_updated(self):
        config = EvolutionConfig(
            population_size=5,
            generations=2,
            mutation_rate=1.0,  # Always mutate
            crossover_rate=0.0,
            seed_queries=["test query one"],
            target_categories=["negation"],
        )
        fitness = FakeFitnessEvaluator()
        optimizer = AdversarialQueryOptimizer(config, fitness)

        await optimizer.optimize()
        # Bandit should have been updated with some pulls
        total_pulls = sum(optimizer.bandit.counts.values())
        assert total_pulls > 0
