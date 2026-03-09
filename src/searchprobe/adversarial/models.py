"""Data models for the adversarial query optimizer."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Individual:
    """A single query in the evolutionary population."""

    query: str
    fitness: float = 0.0
    category: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    mutation_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def __lt__(self, other: "Individual") -> bool:
        """Compare by fitness for sorting (higher fitness = better adversarial)."""
        return self.fitness < other.fitness

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "fitness": self.fitness,
            "category": self.category,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_history": self.mutation_history,
            "metadata": self.metadata,
        }


@dataclass
class Population:
    """A population of query individuals."""

    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def best(self) -> Individual | None:
        return max(self.individuals, key=lambda x: x.fitness) if self.individuals else None

    @property
    def mean_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return sum(i.fitness for i in self.individuals) / len(self.individuals)

    def top_n(self, n: int) -> list[Individual]:
        """Get top N individuals by fitness."""
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:n]


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary optimizer."""

    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    elitism_count: int = 3  # Always keep top N
    tournament_size: int = 3
    budget_limit: float = 5.0  # USD
    fitness_mode: str = "llm_judge"  # "llm_judge", "cross_encoder", "embedding_sim"
    seed_queries: list[str] = field(default_factory=list)
    target_categories: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of an evolutionary optimization run."""

    best_individuals: list[Individual]
    generations_completed: int
    total_evaluations: int
    total_cost: float
    fitness_history: list[dict[str, float]]  # generation -> {mean, max, min}
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    config: EvolutionConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_individuals": [i.to_dict() for i in self.best_individuals],
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "total_cost": self.total_cost,
            "fitness_history": self.fitness_history,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
