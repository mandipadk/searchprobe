"""ConstraintAwareSearch: a verify-and-rerank layer around any search provider.

Pipeline for one query:
    1. Parse the query into typed constraints (the same schema as SearchProbe
       test cases) plus retrieval rewrites, with an LLM. The first rewrite is
       the query's positive core (negations removed, nothing excluded named).
    2. Over-retrieve: run the original query and the rewrites against the base
       provider; fuse with reciprocal rank fusion.
    3. Fetch every candidate page once (shared evidence layer).
    4. Verify each candidate: deterministic checkers first, then an LLM
       verifier that returns a verdict per constraint.
    5. Rerank: drop (or demote) violating documents, put documents that
       satisfy every hard constraint first, then by relevance, then by fused rank.

The wrapper is itself a ``SearchProvider``, so SearchProbe evaluates it as
just another system (``aris+exa:neural``) on the same cases with the same
judge. The verifier model defaults to a different model from the evaluation
judge so the system is not graded by the same model that filtered it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from searchprobe.cases.generators.llm import CONSTRAINT_SCHEMA, _to_constraint
from searchprobe.cases.models import FreeformConstraint, TestCase
from searchprobe.judging.checkers import run_checkers
from searchprobe.judging.llm_judge import DocumentJudge, evidence_from
from searchprobe.judging.models import DocJudgment
from searchprobe.llm import structured_call
from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

PARSE_SYSTEM = """You turn a web search query into (a) typed constraints that define a correct result and (b) up to 3 rewrites optimized for a keyword/embedding search engine.

Constraint types: topic, exclude (results must NOT BE this), numeric, date ("published" = page date, "event" = when it happened; YYYY-MM-DD), entity (with confusable names in "wrong"), relation, doc_type, language (ISO 639-1), freeform. Fill unused fields with null or empty lists; ids c1, c2, ...

Rewrites: the first is the positive core of the query - what correct results are about - with every negation and excluded concept REMOVED (never name the excluded thing; search engines match it). Further rewrites may target individual constraints (e.g. a specific year, a synonym of the topic). Do not invent requirements."""

PARSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "constraints": {"type": "array", "items": CONSTRAINT_SCHEMA},
        "rewrites": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["constraints", "rewrites"],
    "additionalProperties": False,
}


@dataclass
class ParsedQuery:
    case: TestCase
    rewrites: list[str]
    cost_usd: float = 0.0


@dataclass
class Candidate:
    result: SearchResult
    fused: float
    judgment: DocJudgment | None = None
    sources: list[str] = field(default_factory=list)


def rrf_fuse(lists: list[list[SearchResult]], k: int = 60) -> list[Candidate]:
    """Reciprocal rank fusion keyed by URL (first-seen result object is kept)."""
    from searchprobe.perturbation.stability import normalize_url

    scores: dict[str, float] = {}
    first: dict[str, SearchResult] = {}
    for results in lists:
        for rank, r in enumerate(results):
            key = normalize_url(str(r.url))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(key, r)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [Candidate(first[u], scores[u]) for u in ordered]


class ConstraintAwareSearch(SearchProvider):
    """Wrap a base provider with constraint parsing, verification and reranking."""

    NAME: ClassVar[str] = "aris"
    SUPPORTED_MODES: ClassVar[list[str]] = []

    def __init__(
        self,
        base: Any,
        client: Any | None,
        parser_model: str,
        verifier_model: str,
        effort: str | None = "low",
        fanout: int = 2,
        pool_size: int = 20,
        drop_violations: bool = True,
        fetcher: Any | None = None,
        max_concurrent_verifications: int = 8,
    ) -> None:
        super().__init__(api_key="")
        self.base = base
        self.client = client
        self.parser_model = parser_model
        self.verifier = DocumentJudge(client, verifier_model, effort) if client else None
        self.effort = effort
        self.fanout = fanout
        self.pool_size = pool_size
        self.drop_violations = drop_violations
        self.fetcher = fetcher
        self._sem = asyncio.Semaphore(max_concurrent_verifications)
        self._parse_cache: dict[str, ParsedQuery] = {}
        base_name = getattr(base, "NAME", getattr(base, "name", "base"))
        self.NAME = f"aris+{base_name}"  # type: ignore[misc]
        self.SUPPORTED_MODES = list(getattr(base, "SUPPORTED_MODES", []))  # type: ignore[misc]

    @property
    def name(self) -> str:
        return self.NAME

    def estimate_cost(self, request: SearchRequest) -> float:
        base_req = request.model_copy(update={"num_results": self.pool_size})
        per_search = (
            self.base.estimate_cost(base_req) if hasattr(self.base, "estimate_cost") else 0.0
        )
        verify = 0.015 * self.pool_size  # conservative per-document verification estimate
        return per_search * (1 + self.fanout) + verify + 0.02

    async def parse(self, query: str) -> ParsedQuery:
        if query in self._parse_cache:
            return self._parse_cache[query]
        if self.client is None:
            case = TestCase(
                text=query,
                category="multi_constraint",
                constraints=[
                    FreeformConstraint(id="c1", requirement=f"The result satisfies: '{query}'")
                ],
            )
            parsed = ParsedQuery(case, [])
        else:
            result = await structured_call(
                self.client,
                model=self.parser_model,
                system=PARSE_SYSTEM,
                prompt=f'Query: "{query}"',
                schema=PARSE_SCHEMA,
                effort=self.effort,
            )
            constraints = [c for c in (_to_constraint(r) for r in result.data["constraints"]) if c]
            seen = set()
            unique = []
            for c in constraints:  # the model occasionally repeats an id
                if c.id in seen:
                    c = c.model_copy(update={"id": f"c{len(seen) + 1}x"})
                seen.add(c.id)
                unique.append(c)
            if not any(c.hard for c in unique):
                unique.append(
                    FreeformConstraint(
                        id=f"c{len(unique) + 1}", requirement=f"The result satisfies: '{query}'"
                    )
                )
            case = TestCase(text=query, category="multi_constraint", constraints=unique)
            rewrites = [r.strip() for r in result.data.get("rewrites", []) if r.strip()]
            parsed = ParsedQuery(
                case, [r for r in rewrites if r.lower() != query.lower()], result.cost_usd
            )
        self._parse_cache[query] = parsed
        return parsed

    async def _verify(self, case: TestCase, cand: Candidate, snapshots: dict) -> float:
        url = str(cand.result.url)
        snap = snapshots.get(url)
        verdicts = run_checkers(case, url, snap)
        if self.verifier is None:
            return 0.0
        ev = evidence_from(url, snap, cand.result.title, cand.result.content or cand.result.snippet)
        async with self._sem:
            try:
                cand.judgment = await self.verifier.judge(case, ev, decided=verdicts)
            except Exception as e:  # verification failure: keep the doc, unranked by checks
                logger.warning("Aris verification failed for %s: %s", url, e)
                return 0.0
        return cand.judgment.cost_usd

    def _sort_key(self, case: TestCase, cand: Candidate) -> tuple:
        j = cand.judgment
        if j is None:
            return (1, 1, 0, -cand.fused)
        violating = j.is_violating(case)
        satisfying = j.is_fully_satisfying(case)
        return (int(violating), int(not satisfying), -j.relevance, -cand.fused)

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.perf_counter()
        cost = 0.0
        try:
            parsed = await self.parse(request.query)
            cost += parsed.cost_usd
            queries = [request.query, *parsed.rewrites[: self.fanout]]
            base_requests = [
                SearchRequest(
                    query=q,
                    num_results=self.pool_size,
                    include_content=False,
                    search_mode=request.search_mode,
                )
                for q in queries
            ]
            responses = await asyncio.gather(*[self.base.search(r) for r in base_requests])
            cost += sum(r.cost_usd for r in responses)
            ok = [r for r in responses if not r.error]
            if not ok:
                return SearchResponse(
                    provider=self.NAME,
                    search_mode=request.search_mode,
                    query=request.query,
                    error=responses[0].error,
                    cost_usd=cost,
                )
            candidates = rrf_fuse([r.results for r in ok])[: self.pool_size]
            urls = [str(c.result.url) for c in candidates]
            snapshots = await self.fetcher.fetch_many(urls) if self.fetcher else {}
            costs = await asyncio.gather(
                *[self._verify(parsed.case, c, snapshots) for c in candidates]
            )
            cost += sum(costs)
            ranked = sorted(candidates, key=lambda c: self._sort_key(parsed.case, c))
            if self.drop_violations:
                kept = [
                    c
                    for c in ranked
                    if c.judgment is None or not c.judgment.is_violating(parsed.case)
                ]
            else:
                kept = ranked
            results = []
            for pos, cand in enumerate(kept[: request.num_results]):
                snap = snapshots.get(str(cand.result.url))
                content = (
                    snap.text[: request.max_content_chars]
                    if snap is not None and snap.ok
                    else cand.result.content
                )
                results.append(
                    cand.result.model_copy(
                        update={
                            "position": pos,
                            "content": content if request.include_content else None,
                            "provider_raw": {
                                "aris_verdicts": {
                                    v.constraint_id: v.verdict
                                    for v in (cand.judgment.verdicts if cand.judgment else [])
                                },
                                "aris_relevance": cand.judgment.relevance
                                if cand.judgment
                                else None,
                            },
                        }
                    )
                )
            return SearchResponse(
                provider=self.NAME,
                search_mode=request.search_mode,
                query=request.query,
                results=results,
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=cost,
                timestamp=datetime.now(UTC),
                metadata={
                    "constraints": [c.describe() for c in parsed.case.constraints],
                    "rewrites": queries[1:],
                    "pool": len(candidates),
                    "dropped_violations": len(ranked) - len(kept),
                },
            )
        except Exception as e:
            return SearchResponse(
                provider=self.NAME,
                search_mode=request.search_mode,
                query=request.query,
                error=f"{type(e).__name__}: {e}",
                cost_usd=cost,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def close(self) -> None:
        if hasattr(self.base, "close"):
            await self.base.close()
        if self.fetcher is not None:
            await self.fetcher.close()
