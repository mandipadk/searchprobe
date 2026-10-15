"""Run metamorphic tests: base query, repeats (noise floor), twins (DIR), paraphrases (INV)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from searchprobe.cases.models import TestCase
from searchprobe.metamorphic.obi import VariantResult
from searchprobe.metamorphic.variants import Variant
from searchprobe.perturbation.stability import normalize_url
from searchprobe.providers.models import SearchRequest

logger = logging.getLogger(__name__)


@dataclass
class DirectionResult:
    """Judged direction for one DIR variant: did results move toward the twin's answers?"""

    system: str
    case_id: str
    relation: str
    satisfied_variant: float  # share of variant results satisfying the twin's constraints
    satisfied_base: float  # share of base results satisfying the twin's constraints
    n: int

    @property
    def direction(self) -> float:
        return self.satisfied_variant - self.satisfied_base


@dataclass
class MetamorphicRun:
    results: list[VariantResult] = field(default_factory=list)
    directions: list[DirectionResult] = field(default_factory=list)
    searches: int = 0
    skipped_for_budget: int = 0
    cost_usd: float = 0.0
    # (system, case_id, relation) -> (base response, variant response), for judging.
    raw: dict[tuple[str, str, str], tuple[Any, Any]] = field(default_factory=dict, repr=False)


class MetamorphicEngine:
    def __init__(
        self,
        providers: dict[str, Any],
        num_results: int = 10,
        cost_tracker: Any | None = None,
        concurrency: int = 5,
    ) -> None:
        """
        Args:
            providers: system key -> provider (``search(SearchRequest)``); the key's
                ``:mode`` suffix is sent as ``search_mode``.
        """
        self.providers = providers
        self.num_results = num_results
        self.cost_tracker = cost_tracker
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _search(self, system: str, text: str, run: MetamorphicRun):
        provider = self.providers[system]
        mode = system.split(":", 1)[1] if ":" in system else None
        request = SearchRequest(
            query=text, num_results=self.num_results, include_content=False, search_mode=mode
        )
        reservation = None
        if self.cost_tracker is not None:
            estimate = (
                provider.estimate_cost(request) if hasattr(provider, "estimate_cost") else 0.0
            )
            reservation = await self.cost_tracker.reserve(estimate)
            if reservation is None:
                run.skipped_for_budget += 1
                return None, "budget exhausted"
        async with self.semaphore:
            try:
                response = await provider.search(request)
            except Exception as e:
                response = None
                error = f"{type(e).__name__}: {e}"
        cost = response.cost_usd if response is not None else 0.0
        if self.cost_tracker is not None:
            self.cost_tracker.commit(reservation, system, "metamorphic", cost)
        run.cost_usd += cost
        run.searches += 1
        if response is None:
            return None, error
        if response.error:
            return None, response.error
        if not response.results:  # RBO against an empty list is not an observation
            return None, "empty results"
        return response, None

    async def run(
        self,
        cases: list[TestCase],
        variants: list[Variant],
        progress: Callable[[int, int], None] | None = None,
    ) -> MetamorphicRun:
        run = MetamorphicRun()
        by_case = {c.id: c for c in cases}
        base_ids = sorted({v.case_id for v in variants})
        total = len(self.providers) * (len(base_ids) + len(variants))
        done = 0

        def tick() -> None:
            nonlocal done
            done += 1
            if progress:
                progress(done, total)

        base_results: dict[tuple[str, str], Any] = {}

        async def base(system: str, cid: str) -> None:
            resp, err = await self._search(system, by_case[cid].text, run)
            base_results[(system, cid)] = (resp, err)
            tick()

        await asyncio.gather(*[base(s, cid) for s in self.providers for cid in base_ids])

        async def one(system: str, v: Variant) -> None:
            base_resp, base_err = base_results[(system, v.case_id)]
            resp, err = await self._search(system, v.text, run)
            tick()
            case = by_case[v.case_id]
            run.results.append(
                VariantResult(
                    system=system,
                    case_id=v.case_id,
                    category=case.category.value,
                    kind=v.kind,
                    relation=v.relation,
                    text=v.text,
                    base_urls=[normalize_url(str(r.url)) for r in base_resp.results]
                    if base_resp
                    else [],
                    variant_urls=[normalize_url(str(r.url)) for r in resp.results] if resp else [],
                    error=base_err or err,
                )
            )
            if v.kind == "DIR" and resp is not None and base_resp is not None:
                run.raw[(system, v.case_id, v.relation)] = (base_resp, resp)

        await asyncio.gather(*[one(s, v) for s in self.providers for v in variants])
        return run

    async def judge_directions(
        self,
        run: MetamorphicRun,
        cases: list[TestCase],
        variants: list[Variant],
        judge: Any,
        fetcher: Any | None = None,
        k: int = 5,
        judge_cost_estimate: float = 0.01,
    ) -> list[DirectionResult]:
        """For DIR variants, judge base and variant results against the *twin's* constraints.

        LLM judge calls are reserved from the run's budget like searches; a pair whose
        lists cannot be judged completely (budget, judge error) is left out.
        """
        from searchprobe.judging.checkers import run_checkers
        from searchprobe.judging.engine import checker_only_judgment
        from searchprobe.judging.llm_judge import evidence_from

        by_case = {c.id: c for c in cases}
        out: list[DirectionResult] = []

        async def judge_doc(twin: TestCase, url: str, ev: Any, snap: Any) -> Any | None:
            verdicts = run_checkers(twin, url, snap)
            decided = checker_only_judgment(twin, url, ev, verdicts)
            if decided is not None:
                return decided
            reservation = None
            if self.cost_tracker is not None:
                reservation = await self.cost_tracker.reserve(judge_cost_estimate)
                if reservation is None:
                    run.skipped_for_budget += 1
                    return None
            try:
                j = await judge.judge(twin, ev, decided=verdicts)
            except Exception as e:  # infra failure: leave the pair out, never guess
                logger.warning("Direction judging failed for %s: %s", url, e)
                if self.cost_tracker is not None:
                    self.cost_tracker.release(reservation)
                return None
            if self.cost_tracker is not None:
                self.cost_tracker.commit(reservation, "judge", "metamorphic", j.cost_usd)
            run.cost_usd += j.cost_usd
            return j

        for (system, case_id, relation), (base_resp, var_resp) in run.raw.items():
            v = next((x for x in variants if x.case_id == case_id and x.relation == relation), None)
            if v is None or v.twin_case_id not in by_case:
                continue
            twin = by_case[v.twin_case_id]

            async def share(resp: Any, twin: TestCase = twin) -> tuple[float, int] | None:
                results = resp.results[:k]
                if not results:
                    return 0.0, 0
                urls = [str(r.url) for r in results]
                snaps = await fetcher.fetch_many(urls) if fetcher else {}
                sat = 0
                for r in results:
                    url = str(r.url)
                    ev = evidence_from(url, snaps.get(url), r.title, r.content or r.snippet)
                    j = await judge_doc(twin, url, ev, snaps.get(url))
                    if j is None:
                        return None
                    sat += j.is_fully_satisfying(twin)
                return sat / len(results), len(results)

            var_share = await share(var_resp)
            base_share = await share(base_resp) if var_share is not None else None
            if var_share is None or base_share is None:
                continue
            out.append(
                DirectionResult(
                    system, case_id, relation, var_share[0], base_share[0], var_share[1]
                )
            )
        run.directions = out
        return out
