"""Adversarial discovery: MAP-Elites over test cases with differential fitness.

Fitness of a candidate case for a target system T against a reference system R:

    fitness = violation@k(T) - violation@k(R)      if R answers the case
              0                                    otherwise

so a case scores only if the target fails *and* the query is demonstrably
answerable (the reference returns at least one fully satisfying document).
An unanswerable or nonsensical query cannot win, which stops the search from
drifting toward impossible queries.

The archive keeps the best case per cell (category x mutation family x
answer-set size), so the output is a diverse set of failures rather than many
copies of one. Elites are re-evaluated at the end to counter the winner's
curse (a single lucky evaluation).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from searchprobe.cases.models import TestCase
from searchprobe.discovery.mutations import RULE_MUTATIONS, llm_rephrase
from searchprobe.metrics.ir import ListMetrics, list_metrics
from searchprobe.providers.models import SearchRequest

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    fitness: float
    target: ListMetrics | None
    reference: ListMetrics | None
    answerable: bool


@dataclass
class Elite:
    case: TestCase
    evaluation: Evaluation
    cell: tuple[str, str, str]
    evaluations: list[float] = field(default_factory=list)

    @property
    def mean_fitness(self) -> float:
        return sum(self.evaluations) / len(self.evaluations) if self.evaluations else 0.0


def _size_bucket(case: TestCase) -> str:
    n = len(case.answer_set)
    return "open" if n == 0 else "small" if n <= 5 else "medium" if n <= 30 else "large"


def _family(case: TestCase) -> str:
    notes = case.provenance.notes or ""
    last = notes.split("|")[-1].strip()
    return last.split("(")[0] or "seed"


def cell_of(case: TestCase) -> tuple[str, str, str]:
    return (case.category.value, _family(case), _size_bucket(case))


LLM_MUTATION_ESTIMATE = 0.01  # conservative per-call estimate for budget reservation


class CaseEvaluator:
    """Search + judge one case on the target and reference systems."""

    def __init__(
        self,
        providers: dict[str, Any],
        target: str,
        reference: str | None,
        judge: Any | None,
        fetcher: Any | None = None,
        k: int = 5,
        cost_tracker: Any | None = None,
    ) -> None:
        self.providers = providers
        self.target = target
        self.reference = reference
        self.judge = judge
        self.fetcher = fetcher
        self.k = k
        self.cost_tracker = cost_tracker
        self.budget_exhausted = False

    def _reserve(self, estimate: float) -> Any:
        if self.cost_tracker is None:
            return 0
        r = self.cost_tracker.try_reserve(estimate)
        if r is None:
            self.budget_exhausted = True
        return r

    async def _system_metrics(self, system: str, case: TestCase) -> ListMetrics | None:
        from searchprobe.judging.checkers import run_checkers
        from searchprobe.judging.engine import checker_only_judgment
        from searchprobe.judging.llm_judge import evidence_from

        provider = self.providers[system]
        mode = system.split(":", 1)[1] if ":" in system else None
        req = SearchRequest(query=case.text, num_results=self.k, search_mode=mode)
        est = provider.estimate_cost(req) if hasattr(provider, "estimate_cost") else 0.0
        reservation = self._reserve(est)
        if reservation is None:
            return None
        resp = await provider.search(req)
        if self.cost_tracker is not None:
            self.cost_tracker.commit(reservation, system, "discovery", resp.cost_usd)
        if resp.error:
            return list_metrics(case, system, [], [], k=self.k, error=resp.error)
        urls = [str(r.url) for r in resp.results[: self.k]]
        snaps = await self.fetcher.fetch_many(urls) if self.fetcher else {}
        judgments = []
        for r in resp.results[: self.k]:
            url = str(r.url)
            ev = evidence_from(url, snaps.get(url), r.title, r.content or r.snippet)
            verdicts = run_checkers(case, url, snaps.get(url))
            j = checker_only_judgment(case, url, ev, verdicts)
            if j is None and self.judge is not None:
                reservation = self._reserve(0.01)
                if reservation is None:
                    judgments.append(None)
                    continue
                try:
                    j = await self.judge.judge(case, ev, decided=verdicts)
                except Exception as e:  # refusal / truncation: leave unjudged, keep going
                    logger.warning("Judging failed for %s: %s", url, e)
                    if self.cost_tracker is not None:
                        self.cost_tracker.release(reservation)
                    judgments.append(None)
                    continue
                if self.cost_tracker is not None:
                    self.cost_tracker.commit(reservation, "judge", "discovery", j.cost_usd)
            judgments.append(j)
        pool = [j for j in judgments if j is not None]
        return list_metrics(case, system, judgments, pool, k=self.k)

    async def evaluate(self, case: TestCase) -> Evaluation:
        target = await self._system_metrics(self.target, case)
        reference = await self._system_metrics(self.reference, case) if self.reference else None
        if target is None or target.violation_at_k is None:
            return Evaluation(0.0, target, reference, False)
        if self.reference is None:
            # Without a reference, require the target itself to show the case is
            # answerable (at least one satisfying document) while still violating.
            answerable = (target.satisfied_at_k or 0) > 0
            fit = target.violation_at_k if answerable else 0.0
            return Evaluation(fit, target, None, answerable)
        if reference is None or reference.violation_at_k is None:
            return Evaluation(0.0, target, reference, False)
        answerable = (reference.satisfied_at_k or 0) > 0
        fit = max(0.0, target.violation_at_k - reference.violation_at_k) if answerable else 0.0
        return Evaluation(fit, target, reference, answerable)


class MapElites:
    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, str], Elite] = {}
        self.seen: set[str] = set()

    def consider(self, case: TestCase, evaluation: Evaluation) -> bool:
        key = " ".join(case.text.lower().split())
        self.seen.add(key)
        cell = cell_of(case)
        incumbent = self.cells.get(cell)
        if incumbent is None or evaluation.fitness > incumbent.evaluation.fitness:
            self.cells[cell] = Elite(case, evaluation, cell, [evaluation.fitness])
            return True
        return False

    def is_new(self, case: TestCase) -> bool:
        return " ".join(case.text.lower().split()) not in self.seen

    def elites(self) -> list[Elite]:
        return sorted(self.cells.values(), key=lambda e: e.evaluation.fitness, reverse=True)


@dataclass
class DiscoveryResult:
    archive: MapElites
    evaluated: int
    accepted: int


async def discover(
    seeds: list[TestCase],
    evaluator: CaseEvaluator,
    iterations: int = 50,
    reevaluate: int = 2,
    llm: tuple[Any, str] | None = None,
    seed: int = 0,
    progress: Callable[[int, int, float], None] | None = None,
) -> DiscoveryResult:
    """Run MAP-Elites from seed cases."""
    rng = random.Random(seed)
    archive = MapElites()
    evaluated = accepted = 0

    for case in seeds:
        if evaluator.budget_exhausted:
            break
        archive.consider(case, await evaluator.evaluate(case))
        evaluated += 1

    mutation_names = list(RULE_MUTATIONS) + (["llm_rephrase"] if llm else [])
    for it in range(iterations):
        if evaluator.budget_exhausted or not archive.cells:
            break
        parent = rng.choice(list(archive.cells.values())).case
        name = rng.choice(mutation_names)
        if name == "llm_rephrase":
            client, model = llm  # type: ignore[misc]
            reservation = evaluator._reserve(LLM_MUTATION_ESTIMATE)
            if reservation is None:
                break
            cost = 0.0
            try:
                child, cost = await llm_rephrase(parent, client, model)
            except Exception as e:
                logger.warning("LLM mutation failed: %s", e)
                child = None
            if evaluator.cost_tracker is not None:
                evaluator.cost_tracker.commit(reservation, "mutation", model, cost)
        else:
            child = RULE_MUTATIONS[name](parent, rng)
        if child is None or not archive.is_new(child):
            continue
        ev = await evaluator.evaluate(child)
        evaluated += 1
        accepted += archive.consider(child, ev)
        if progress:
            best = max((e.evaluation.fitness for e in archive.cells.values()), default=0.0)
            progress(it + 1, iterations, best)

    # Re-evaluate elites to counter selection on noise.
    for elite in archive.elites():
        if elite.evaluation.fitness <= 0:
            continue
        for _ in range(reevaluate):
            if evaluator.budget_exhausted:
                break
            evaluation = await evaluator.evaluate(elite.case)
            if evaluator.budget_exhausted:  # incomplete evaluation: not a measurement
                break
            elite.evaluations.append(evaluation.fitness)
    return DiscoveryResult(archive, evaluated, accepted)


def promote(result: DiscoveryResult, min_fitness: float = 0.2) -> list[TestCase]:
    """Elites whose re-evaluated mean fitness clears the bar, as new test cases."""
    from searchprobe.cases.twins import attach_twins

    out = []
    for elite in result.archive.elites():
        if elite.mean_fitness >= min_fitness and elite.case.provenance.generator == "discovered":
            note = (
                f"discovered: mean differential fitness {elite.mean_fitness:.2f} over "
                f"{len(elite.evaluations)} evaluations"
            )
            out.append(elite.case.model_copy(update={"notes": note}))
    return attach_twins(out)
