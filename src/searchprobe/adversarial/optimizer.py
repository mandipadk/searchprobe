"""Adversarial query optimizer — evolutionary loop for breeding worst-case queries."""

import random
from datetime import datetime
from typing import Any, Callable

from searchprobe.adversarial.bandit import OperatorBandit
from searchprobe.adversarial.crossover import apply_random_crossover
from searchprobe.adversarial.fitness import FitnessEvaluator
from searchprobe.adversarial.models import (
    EvolutionConfig,
    Individual,
    OptimizationResult,
    Population,
)
from searchprobe.adversarial.mutations import MUTATION_OPERATORS, apply_random_mutation


class AdversarialQueryOptimizer:
    """Evolutionary optimizer that breeds worst-case adversarial queries.

    Instead of hand-crafting adversarial queries, this uses evolutionary
    optimization to discover failure modes humans wouldn't think of.

    Features:
    - UCB1 bandit for adaptive mutation operator selection
    - Optional geometry-guided mutation (via SharedContext)
    - Diversity-aware survivor selection

    Algorithm:
    1. Initialize population from seed queries
    2. Evaluate fitness (how badly each query breaks the search engine)
    3. Select parents via tournament selection
    4. Apply crossover and mutation to create offspring (using bandit-selected operators)
    5. Evaluate offspring fitness
    6. Select survivors (with elitism)
    7. Repeat for N generations
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_evaluator: FitnessEvaluator,
        context: Any | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            config: Evolution configuration
            fitness_evaluator: Fitness evaluator for scoring individuals
            context: Optional SharedContext for geometry-guided evolution
        """
        self.config = config
        self.fitness = fitness_evaluator
        self.context = context
        self.bandit = OperatorBandit(list(MUTATION_OPERATORS.keys()))

    def _initialize_population(self) -> Population:
        """Create initial population from seed queries."""
        individuals = []

        for query in self.config.seed_queries:
            category = ""
            if self.config.target_categories:
                category = random.choice(self.config.target_categories)

            individuals.append(
                Individual(
                    query=query,
                    category=category,
                    generation=0,
                )
            )

        # If not enough seeds, generate variants
        while len(individuals) < self.config.population_size:
            if individuals:
                parent = random.choice(individuals)
                child = apply_random_mutation(parent)
                child.generation = 0
                individuals.append(child)
            else:
                # Default seed if none provided
                individuals.append(
                    Individual(
                        query="companies that are NOT in AI",
                        category="negation",
                        generation=0,
                    )
                )

        return Population(
            individuals=individuals[:self.config.population_size],
            generation=0,
        )

    def _tournament_select(self, population: Population) -> Individual:
        """Select an individual via tournament selection.

        Args:
            population: Current population

        Returns:
            Selected individual
        """
        candidates = random.sample(
            population.individuals,
            min(self.config.tournament_size, len(population.individuals)),
        )
        return max(candidates, key=lambda x: x.fitness)

    def _create_offspring(self, population: Population) -> list[Individual]:
        """Create offspring through crossover and mutation.

        Uses UCB1 bandit for adaptive operator selection instead of uniform random.

        Args:
            population: Current population

        Returns:
            List of offspring individuals
        """
        offspring = []
        target_size = self.config.population_size - self.config.elitism_count

        while len(offspring) < target_size:
            if random.random() < self.config.crossover_rate and len(population.individuals) >= 2:
                # Crossover
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                child = apply_random_crossover(parent_a, parent_b)
            else:
                # Selection + mutation
                parent = self._tournament_select(population)
                child = Individual(
                    query=parent.query,
                    category=parent.category,
                    generation=population.generation + 1,
                    parent_ids=[parent.id],
                    mutation_history=parent.mutation_history.copy(),
                )

            # Apply mutation with bandit-selected operator
            if random.random() < self.config.mutation_rate:
                operator_name = self.bandit.select()
                operator_fn = MUTATION_OPERATORS[operator_name]
                old_fitness = child.fitness
                child = operator_fn(child)
                # Store operator name for later reward update
                child.metadata["_pending_operator"] = operator_name
                child.metadata["_pre_mutation_fitness"] = old_fitness

            child.generation = population.generation + 1
            offspring.append(child)

        return offspring

    def _update_bandit_rewards(self, offspring: list[Individual]) -> None:
        """Update bandit with fitness improvements from mutations."""
        for ind in offspring:
            operator = ind.metadata.pop("_pending_operator", None)
            pre_fitness = ind.metadata.pop("_pre_mutation_fitness", None)
            if operator and pre_fitness is not None:
                reward = ind.fitness - pre_fitness
                self.bandit.update(operator, reward)

    async def optimize(
        self,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> OptimizationResult:
        """Run the evolutionary optimization loop.

        Args:
            progress_callback: Optional callback(generation, total_gens, best_fitness, mean_fitness)

        Returns:
            OptimizationResult with best individuals and fitness history
        """
        started_at = datetime.utcnow()
        fitness_history: list[dict[str, float]] = []
        total_evaluations = 0

        # Initialize
        population = self._initialize_population()

        # Evaluate initial population
        population.individuals = await self.fitness.evaluate_population(
            population.individuals
        )
        total_evaluations += population.size

        for gen in range(self.config.generations):
            # Record fitness stats
            fitnesses = [i.fitness for i in population.individuals]
            stats = {
                "generation": gen,
                "mean": sum(fitnesses) / len(fitnesses),
                "max": max(fitnesses),
                "min": min(fitnesses),
            }
            fitness_history.append(stats)

            if progress_callback:
                progress_callback(gen, self.config.generations, stats["max"], stats["mean"])

            # Elitism — preserve top individuals
            elites = population.top_n(self.config.elitism_count)

            # Create offspring
            offspring = self._create_offspring(population)

            # Evaluate offspring
            offspring = await self.fitness.evaluate_population(offspring)
            total_evaluations += len(offspring)

            # Update bandit rewards based on fitness improvements
            self._update_bandit_rewards(offspring)

            # Form next generation
            next_gen = elites + offspring
            population = Population(
                individuals=next_gen[:self.config.population_size],
                generation=gen + 1,
            )

        # Final fitness recording
        fitnesses = [i.fitness for i in population.individuals]
        fitness_history.append({
            "generation": self.config.generations,
            "mean": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "max": max(fitnesses) if fitnesses else 0,
            "min": min(fitnesses) if fitnesses else 0,
        })

        # Get best individuals
        best = population.top_n(10)

        return OptimizationResult(
            best_individuals=best,
            generations_completed=self.config.generations,
            total_evaluations=total_evaluations,
            total_cost=0.0,  # Would be tracked by CostTracker in live mode
            fitness_history=fitness_history,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            config=self.config,
        )
