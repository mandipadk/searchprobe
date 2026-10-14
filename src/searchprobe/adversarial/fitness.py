"""Fitness evaluation for adversarial query individuals."""

import asyncio
from typing import Any

from searchprobe.adversarial.models import Individual


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

        async def _one(ind: Individual) -> None:
            key = " ".join(ind.query.lower().split())
            if key in self._cache:
                ind.fitness = self._cache[key]
                return
            async with semaphore:
                reservation = None
                live = self.mode in ("llm_judge", "cross_encoder") and self.provider is not None
                if live and self.cost_tracker is not None:
                    reservation = self.cost_tracker.try_reserve(self._estimate_cost())
                    if reservation is None:
                        self.budget_exhausted = True
                        ind.fitness = 0.0
                        return
                fitness, cost = await self._evaluate_with_cost(ind)
                if self.cost_tracker is not None:
                    self.cost_tracker.commit(reservation, "evolution", self.mode, cost)
                self.total_cost += cost
            ind.fitness = fitness
            self._cache[key] = fitness

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

    async def _evaluate_with_cost(self, individual: Individual) -> tuple[float, float]:
        self._last_cost = 0.0
        fitness = await self.evaluate(individual)
        return fitness, self._last_cost

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
            self._last_cost = getattr(self, "_last_cost", 0.0) + response.cost_usd

            if response.error:
                return 0.0  # infrastructure failure is not an adversarial success

            evaluation = await self.judge.evaluate(
                query_id=individual.id,
                query_text=individual.query,
                category=individual.category,
                search_response=response,
            )

            self._last_cost += getattr(evaluation, "cost_usd", 0.0)
            if not evaluation.counts_for_quality:
                return 0.0
            # Fitness = 1 - score (lower relevance = higher adversarial fitness)
            return 1.0 - evaluation.weighted_score

        except Exception:
            return 0.0  # evaluation error: not evidence of an adversarial query

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
            self._last_cost = getattr(self, "_last_cost", 0.0) + response.cost_usd

            if response.error:
                return 0.0

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
            return 0.0  # evaluation error: not evidence of an adversarial query

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
