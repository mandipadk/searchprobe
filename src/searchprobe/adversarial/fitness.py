"""Fitness evaluation for adversarial query individuals."""

import asyncio
from contextvars import ContextVar
from typing import Any

from searchprobe.adversarial.models import Individual

# Spend of the evaluation running in the current asyncio task. Population members are
# evaluated concurrently, so a per-instance attribute would mix their costs.
_task_cost: ContextVar[float] = ContextVar("fitness_task_cost", default=0.0)
# Set when the current task's evaluation hit an infrastructure failure: its 0.0 is
# not a measurement and must not be cached.
_task_failed: ContextVar[bool] = ContextVar("fitness_task_failed", default=False)


class FitnessEvaluator:
    """Evaluates how effectively a query breaks the search engine.

    Supports multiple fitness modes:
    - llm_judge: Use LLM judge (expensive but accurate)
    - cross_encoder: Use cross-encoder gap (moderate cost)
    - embedding_sim: Use embedding similarity to known-bad target (cheap)
    """

    def __init__(
        self,
        mode: str = "embedding_sim",
        provider: Any | None = None,
        judge: Any | None = None,
        cross_encoder: Any | None = None,
        search_mode: str | None = None,
        cost_tracker: Any | None = None,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize the fitness evaluator.

        Args:
            mode: Fitness evaluation mode
            provider: Search provider for live testing
            judge: SearchJudge for LLM-based evaluation
            cross_encoder: CrossEncoderValidator for gap analysis
        """
        self.mode = mode
        self.provider = provider
        self.judge = judge
        self.cross_encoder = cross_encoder
        self.search_mode = search_mode
        self.cost_tracker = cost_tracker
        self.max_concurrent = max_concurrent
        self.total_cost = 0.0
        self.budget_exhausted = False
        self._cache: dict[str, float] = {}
        self._embedding_model: Any = None

    async def evaluate(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual.

        Higher fitness = more adversarial (worse search results).

        Args:
            individual: Individual to evaluate

        Returns:
            Fitness score in [0, 1]
        """
        if self.mode == "llm_judge":
            return await self._evaluate_llm_judge(individual)
        elif self.mode == "cross_encoder":
            return await self._evaluate_cross_encoder(individual)
        elif self.mode in ("heuristic", "embedding_sim"):
            return self._evaluate_embedding_sim(individual)
        raise ValueError(f"Unknown fitness mode: {self.mode}")

    async def evaluate_population(self, individuals: list[Individual]) -> list[Individual]:
        """Evaluate fitness for a population concurrently, with caching.

        Identical queries (case/whitespace-normalized) are evaluated once. When
        a cost tracker is attached, live evaluations reserve budget first; once
        the budget is exhausted, remaining individuals get fitness 0.0 and
        ``budget_exhausted`` is set.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        live = self.mode in ("llm_judge", "cross_encoder") and self.provider is not None
        inflight: dict[str, asyncio.Future[float]] = {}

        def cache_key(ind: Individual) -> str:
            query = " ".join(ind.query.lower().split())
            if not live:  # heuristic fitness also depends on mutation depth
                return f"{query}|{len(ind.mutation_history)}"
            # The judge scores with category-specific dimensions and weights.
            return f"{query}|{ind.category}" if self.mode == "llm_judge" else query

        async def _one(ind: Individual) -> None:
            key = cache_key(ind)
            if key in self._cache:
                ind.fitness = self._cache[key]
                return
            if key in inflight:  # a duplicate in this population: share its evaluation
                # Shielded: cancelling this waiter must not cancel the shared result.
                ind.fitness = await asyncio.shield(inflight[key])
                return
            future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
            inflight[key] = future
            fitness = 0.0
            try:
                async with semaphore:
                    reservation = None
                    if live and self.cost_tracker is not None:
                        reservation = await self.cost_tracker.reserve(self._estimate_cost())
                        if reservation is None:
                            self.budget_exhausted = True
                            return
                    try:
                        fitness, cost, failed = await self._evaluate_with_cost(ind)
                    except BaseException:  # cancellation bypasses the evaluators' handlers
                        if self.cost_tracker is not None:
                            # Spend so far is unknown to the tracker beyond this task's tally.
                            self.cost_tracker.commit(
                                reservation, "evolution", self.mode, _task_cost.get()
                            )
                        raise
                    if self.cost_tracker is not None:
                        self.cost_tracker.commit(reservation, "evolution", self.mode, cost)
                    self.total_cost += cost
                if not failed:  # a transient failure is retried in a later generation
                    self._cache[key] = fitness
            finally:
                ind.fitness = fitness
                if not future.done():
                    future.set_result(fitness)
                inflight.pop(key, None)

        await asyncio.gather(*[_one(ind) for ind in individuals])
        return individuals

    def _estimate_cost(self) -> float:
        cost = 0.0
        if self.provider is not None and hasattr(self.provider, "estimate_cost"):
            from searchprobe.providers.models import SearchRequest

            cost += self.provider.estimate_cost(
                SearchRequest(query="x", num_results=5, search_mode=self.search_mode)
            )
        if self.mode == "llm_judge":
            cost += 0.02  # conservative per-judgment estimate
        return cost

    async def _evaluate_with_cost(self, individual: Individual) -> tuple[float, float, bool]:
        """Fitness, this evaluation's cost, and whether it failed for infrastructure reasons."""
        _task_cost.set(0.0)
        _task_failed.set(False)
        fitness = await self.evaluate(individual)
        return fitness, _task_cost.get(), _task_failed.get()

    @staticmethod
    def _failed() -> float:
        _task_failed.set(True)
        return 0.0

    @staticmethod
    def _add_cost(cost: float) -> None:
        _task_cost.set(_task_cost.get() + cost)

    async def _evaluate_llm_judge(self, individual: Individual) -> float:
        """Evaluate using LLM judge — most accurate but expensive."""
        if self.provider is None or self.judge is None:
            return 0.0

        from searchprobe.providers.models import SearchRequest

        try:
            request = SearchRequest(
                query=individual.query,
                num_results=5,
                include_content=True,
                search_mode=self.search_mode,
            )
            response = await self.provider.search(request)
            self._add_cost(response.cost_usd)

            if response.error:
                return self._failed()  # infrastructure failure is not an adversarial success

            evaluation = await self.judge.evaluate(
                query_id=individual.id,
                query_text=individual.query,
                category=individual.category,
                search_response=response,
            )

            self._add_cost(getattr(evaluation, "cost_usd", 0.0))
            if not evaluation.counts_for_quality:
                return self._failed()
            # Fitness = 1 - score (lower relevance = higher adversarial fitness)
            return 1.0 - evaluation.weighted_score

        except Exception:
            return self._failed()  # evaluation error: not evidence of an adversarial query

    async def _evaluate_cross_encoder(self, individual: Individual) -> float:
        """Evaluate using cross-encoder gap."""
        if self.provider is None or self.cross_encoder is None:
            return 0.0

        from searchprobe.providers.models import SearchRequest

        try:
            request = SearchRequest(
                query=individual.query,
                num_results=5,
                include_content=True,
                search_mode=self.search_mode,
            )
            response = await self.provider.search(request)
            self._add_cost(response.cost_usd)

            if response.error:
                return self._failed()

            results_dicts = [
                {
                    "title": r.title,
                    "url": str(r.url),
                    "snippet": r.snippet,
                    "content": r.content,
                }
                for r in response.results
            ]

            validation = self.cross_encoder.validate_search_results(
                query_id=individual.id,
                query_text=individual.query,
                category=individual.category,
                provider=response.provider,
                results=results_dicts,
            )

            # Fitness = NDCG improvement (higher gap = more adversarial)
            return min(1.0, validation.ndcg_improvement * 2)

        except Exception:
            return self._failed()  # evaluation error: not evidence of an adversarial query

    def _evaluate_embedding_sim(self, individual: Individual) -> float:
        """Structural heuristic ("heuristic"/legacy "embedding_sim" mode).

        WARNING: this never queries a search engine. It scores query *shape*
        (negations, digits, length), so evolving against it produces long,
        operator-stuffed queries regardless of how any engine behaves. Use it
        only for offline smoke tests.

        Uses structural features of the query to estimate adversarial potential.
        """
        score = 0.0
        query_lower = individual.query.lower()
        words = query_lower.split()

        # Feature 1: Has negation (0-0.2)
        negation_words = {"not", "no", "never", "without", "neither", "nor", "none"}
        has_negation = any(w in negation_words for w in words)
        if has_negation:
            score += 0.2

        # Feature 2: Has numeric constraints (0-0.15)
        has_numbers = any(c.isdigit() for c in query_lower)
        if has_numbers:
            score += 0.15

        # Feature 3: Multi-constraint (0-0.2)
        constraint_markers = {"and", "with", "that", "which", "where", "having", "including"}
        n_constraints = sum(1 for w in words if w in constraint_markers)
        score += min(0.2, n_constraints * 0.05)

        # Feature 4: Query length/complexity (0-0.15)
        word_count = len(words)
        if word_count > 10:
            score += 0.15
        elif word_count > 6:
            score += 0.1

        # Feature 5: Has temporal markers (0-0.1)
        temporal_words = {
            "before",
            "after",
            "since",
            "until",
            "between",
            "during",
            "recent",
            "latest",
        }
        if any(w in temporal_words for w in words):
            score += 0.1

        # Feature 6: Boolean operators (0-0.1)
        n_boolean = sum(1 for w in words if w.upper() in {"AND", "OR", "NOT"})
        if n_boolean >= 2:
            score += 0.1

        # Feature 7: Mutation depth (0-0.1)
        mutation_depth = len(individual.mutation_history)
        score += min(0.1, mutation_depth * 0.02)

        return min(1.0, score)
