"""Judge every (case, document) pair in a benchmark run.

Pipeline per run:
    search results -> unique URLs -> evidence snapshots (fetched once)
    -> unique (case, content) pairs -> checkers -> LLM judge (cached)
    -> ranked, judged lists per (case, system)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from searchprobe.cases.convert import db_query_to_case
from searchprobe.cases.models import TestCase
from searchprobe.core.systems import system_key
from searchprobe.evidence.fetcher import EvidenceFetcher
from searchprobe.evidence.store import Snapshot
from searchprobe.judging.checkers import match_entity, run_checkers
from searchprobe.judging.llm_judge import (
    PROMPT_VERSION,
    DocumentEvidence,
    DocumentJudge,
    evidence_from,
)
from searchprobe.judging.models import ConstraintVerdict, DocJudgment
from searchprobe.judging.store import JudgmentStore
from searchprobe.storage import Database

logger = logging.getLogger(__name__)

CHECKER_JUDGE_ID = "checker"


@dataclass
class JudgedList:
    """One system's ranked results for one case, with a judgment per position."""

    case: TestCase
    system: str
    query_row_id: str
    judgments: list[DocJudgment | None]
    error: str | None = None

    @property
    def n_returned(self) -> int:
        return len(self.judgments)


@dataclass
class RunJudgments:
    run_id: str
    lists: list[JudgedList] = field(default_factory=list)
    judged_pairs: int = 0
    cached_pairs: int = 0
    checker_only_pairs: int = 0
    failed_pairs: int = 0
    skipped_for_budget: int = 0
    cost_usd: float = 0.0

    def cases(self) -> dict[str, TestCase]:
        return {jl.case.id: jl.case for jl in self.lists}

    def systems(self) -> list[str]:
        return sorted({jl.system for jl in self.lists})


def checker_only_judgment(
    case: TestCase, url: str, evidence: DocumentEvidence, verdicts: list
) -> DocJudgment | None:
    """A complete judgment from checkers alone, when the page is a known (non-)answer.

    - Known negative entity: one violated hard constraint already makes the
      document violating (and its nDCG gain 0), so no LLM call is needed;
      undecided constraints are recorded as unaddressed.
    - Known answer entity: only when the checkers decided every hard constraint.
    """
    hard_ids = {c.id for c in case.hard_constraints}
    decided = {v.constraint_id: v for v in verdicts}
    if not hard_ids:
        return None
    if match_entity(url, case.negative_set) is not None and any(
        decided.get(cid) is not None and decided[cid].verdict == "violated" for cid in hard_ids
    ):
        relevance = 1  # on-topic but wrong
        filled = list(verdicts) + [
            ConstraintVerdict(constraint_id=c.id, verdict="unaddressed", source="checker")
            for c in case.constraints
            if c.id not in decided
        ]
    elif match_entity(url, case.answer_set) is not None and hard_ids <= set(decided):
        relevance = 3 if all(v.verdict == "satisfied" for v in verdicts) else 1
        filled = list(verdicts)
    else:
        return None
    return DocJudgment(
        case_id=case.id,
        content_hash=evidence.content_hash,
        url=url,
        judge_id=CHECKER_JUDGE_ID,
        prompt_version="checker-v1",
        relevance=relevance,
        verdicts=filled,
        evidence_source=evidence.source,  # type: ignore[arg-type]
        rationale="Decided by known answer/negative set membership",
    )


class JudgingEngine:
    """Orchestrates evidence fetching and per-document judging for runs."""

    def __init__(
        self,
        db: Database,
        store: JudgmentStore,
        judge: DocumentJudge | None,
        fetcher: EvidenceFetcher | None = None,
        cost_tracker: Any | None = None,
        concurrency: int = 8,
        estimated_cost_per_call: float = 0.01,
        allow_llm: bool = True,
    ) -> None:
        self.db = db
        self.store = store
        self.judge = judge
        self.fetcher = fetcher
        self.cost_tracker = cost_tracker
        self.semaphore = asyncio.Semaphore(concurrency)
        self.estimated_cost_per_call = estimated_cost_per_call
        # With allow_llm=False only cached judgments and checkers are used
        # (e.g. recomputing a scorecard without spending anything).
        self.allow_llm = allow_llm

    def load_run(self, run_id: str, systems: list[str] | None = None, k: int = 10):
        """Return (cases by query row id, search result rows) for a run."""
        rows = self.db.get_search_results(run_id)
        query_ids = {r["query_id"] for r in rows}
        cases: dict[str, TestCase] = {}
        for qid in query_ids:
            q = self.db.get_query(qid)
            if q is not None:
                cases[qid] = db_query_to_case(q)
        selected = []
        for r in rows:
            sys_key = system_key(r["provider"], r["search_mode"])
            if systems and sys_key not in systems:
                continue
            if r["query_id"] not in cases:
                continue
            r = dict(r)
            r["system"] = sys_key
            r["results"] = (r["results"] or [])[:k]
            selected.append(r)
        return cases, selected

    async def judge_run(
        self,
        run_id: str,
        systems: list[str] | None = None,
        k: int = 10,
        refresh_evidence: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> RunJudgments:
        cases, rows = self.load_run(run_id, systems, k)
        out = RunJudgments(run_id=run_id)

        urls = [res["url"] for r in rows for res in r["results"]]
        snapshots: dict[str, Snapshot] = {}
        if self.fetcher is not None and urls:
            snapshots = await self.fetcher.fetch_many(urls, refresh=refresh_evidence)

        # Build unique (case, content) work items.
        work: dict[tuple[str, str], tuple[TestCase, str, DocumentEvidence]] = {}
        layout: list[tuple[TestCase, str, str, list[tuple[str, str]], str | None]] = []
        for r in rows:
            case = cases[r["query_id"]]
            keys = []
            for res in r["results"]:
                ev = evidence_from(
                    res["url"],
                    snapshots.get(res["url"]),
                    fallback_title=res.get("title") or "",
                    fallback_text=res.get("content") or res.get("snippet") or "",
                )
                key = (case.id, ev.content_hash)
                work.setdefault(key, (case, res["url"], ev))
                keys.append(key)
            layout.append((case, r["system"], r["query_id"], keys, r.get("error")))

        results: dict[tuple[str, str], DocJudgment | None] = {}
        total = len(work)
        done = 0

        async def _one(key: tuple[str, str]) -> None:
            nonlocal done
            case, url, ev = work[key]
            results[key] = await self._judge_pair(case, url, ev, snapshots.get(url), out)
            done += 1
            if progress:
                progress(done, total)

        await asyncio.gather(*[_one(key) for key in work])

        for case, system, qid, keys, error in layout:
            out.lists.append(
                JudgedList(
                    case=case,
                    system=system,
                    query_row_id=qid,
                    judgments=[results.get(k) for k in keys],
                    error=error,
                )
            )
        return out

    async def _judge_pair(
        self,
        case: TestCase,
        url: str,
        ev: DocumentEvidence,
        snapshot: Snapshot | None,
        out: RunJudgments,
    ) -> DocJudgment | None:
        if self.judge is not None:
            cached = self.store.get(case.id, ev.content_hash, self.judge.judge_id, PROMPT_VERSION)
            if cached is not None:
                out.cached_pairs += 1
                return cached
        verdicts = run_checkers(case, url, snapshot)
        decided = checker_only_judgment(case, url, ev, verdicts)
        if decided is not None:
            out.checker_only_pairs += 1
            return decided
        if self.judge is None or not self.allow_llm:
            return None

        reservation = None
        if self.cost_tracker is not None:
            reservation = self.cost_tracker.try_reserve(self.estimated_cost_per_call)
            if reservation is None:
                out.skipped_for_budget += 1
                return None
        try:
            async with self.semaphore:
                judgment = await self.judge.judge(case, ev, decided=verdicts)
        except Exception as e:  # infra failure: leave unjudged, never invent a label
            logger.warning("Judging failed for %s / %s: %s", case.id, url, e)
            out.failed_pairs += 1
            if self.cost_tracker is not None:
                self.cost_tracker.release(reservation)
            return None
        if self.cost_tracker is not None:
            self.cost_tracker.commit(reservation, "judge", self.judge.model, judgment.cost_usd)
        out.cost_usd += judgment.cost_usd
        out.judged_pairs += 1
        self.store.put(judgment)
        return judgment
