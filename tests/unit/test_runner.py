"""Regression tests for the benchmark runner and evaluation storage."""

import asyncio

from searchprobe.evaluation.statistics import align_paired, compare_all_systems
from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
from searchprobe.providers.models import SearchResponse
from tests.fakes import FakeProvider


def _runner(temp_db, provider, budget=100.0, providers=("fake",), modes=("auto",)):
    cfg = BenchmarkConfig(
        providers=list(providers), exa_modes=list(modes), budget_limit=budget, max_concurrent=5
    )
    runner = BenchmarkRunner(cfg, db=temp_db)
    runner._get_provider = lambda name: provider
    return runner


def _queries(temp_db, n):
    qs_id = temp_db.create_query_set("t")
    return [
        {
            "id": temp_db.add_query(qs_id, f"q{i}", "negation"),
            "text": f"q{i}",
            "category": "negation",
            "query_set_id": qs_id,
        }
        for i in range(n)
    ]


class TestBudget:
    def test_concurrent_tasks_cannot_overspend(self, temp_db):
        provider = FakeProvider(lambda q: ["https://a.com"], cost=1.0, delay=0.01)
        result = asyncio.run(_runner(temp_db, provider, budget=2.0).run(_queries(temp_db, 20)))
        assert result.total_cost <= 2.0
        assert len(provider.calls) == 2
        assert result.skipped_for_budget == 18
        assert result.status == "partial"


class TestStorage:
    def test_empty_result_sets_are_stored(self, temp_db):
        provider = FakeProvider(lambda q: [])
        runner = _runner(temp_db, provider)
        result = asyncio.run(runner.run(_queries(temp_db, 3)))
        assert result.empty_queries == 3
        assert len(temp_db.get_search_results(result.run_id)) == 3

    def test_provider_exceptions_are_stored_as_errors(self, temp_db):
        class Boom(FakeProvider):
            async def search(self, request):
                raise RuntimeError("kaput")

        result = asyncio.run(_runner(temp_db, Boom()).run(_queries(temp_db, 2)))
        rows = temp_db.get_search_results(result.run_id)
        assert len(rows) == 2 and all("kaput" in r["error"] for r in rows)

    def test_skip_evaluated_is_mode_aware(self, temp_db):
        q = _queries(temp_db, 1)[0]
        run_id = temp_db.create_run(q["query_set_id"])
        for mode in ("auto", "neural"):
            temp_db.add_search_result(
                run_id, q["id"], SearchResponse(provider="exa", search_mode=mode, query="q")
            )
        temp_db.add_evaluation(
            run_id,
            {"query_id": q["id"], "provider": "exa", "search_mode": "auto", "weighted_score": 1},
        )
        pending = temp_db.get_search_results_for_evaluation(run_id, skip_evaluated=True)
        assert [p["search_mode"] for p in pending] == ["neural"]

    def test_aggregates_separate_modes_and_exclude_infra_errors(self, temp_db):
        q = _queries(temp_db, 1)[0]
        run_id = temp_db.create_run(q["query_set_id"])
        for mode, score, status in (
            ("auto", 0.9, "ok"),
            ("neural", 0.1, "ok"),
            ("neural", 0.0, "judge_error"),
        ):
            temp_db.add_evaluation(
                run_id,
                {"query_id": q["id"], "provider": "exa", "search_mode": mode,
                 "weighted_score": score, "status": status},
            )
        temp_db.materialize_aggregate_scores(run_id)
        rows = {
            r["provider"]: r
            for r in temp_db.get_aggregate_scores(run_id)
            if r["dimension"] == "overall"
        }
        assert rows["exa:auto"]["mean_score"] == 0.9
        assert rows["exa:neural"]["mean_score"] == 0.1
        assert rows["exa:neural"]["failure_rate"] == 0.5
        evals = temp_db.get_evaluations(run_id)
        assert {e["system"] for e in evals} == {"exa:auto", "exa:neural"}


class TestPairedStats:
    def test_alignment_uses_query_ids_and_skips_errors(self):
        evals = [
            {"query_id": "1", "provider": "a", "weighted_score": 0.9},
            {"query_id": "2", "provider": "a", "weighted_score": 0.8},
            {"query_id": "3", "provider": "a", "weighted_score": 0.7, "status": "search_error"},
            {"query_id": "2", "provider": "b", "weighted_score": 0.1},
            {"query_id": "1", "provider": "b", "weighted_score": 0.2},
            {"query_id": "3", "provider": "b", "weighted_score": 0.3},
        ]
        a, b, ids = align_paired(evals, "a", "b")
        assert ids == ["1", "2"] and a == [0.9, 0.8] and b == [0.2, 0.1]

    def test_compare_all_systems_reports_real_ci(self):
        evals = []
        for i in range(30):
            evals.append({"query_id": str(i), "provider": "a", "weighted_score": 0.8})
            evals.append({"query_id": str(i), "provider": "b", "weighted_score": 0.2 + i * 0.001})
        (comp,) = compare_all_systems(evals)
        assert comp.significant
        assert comp.ci.lower > 0.5 and comp.ci.upper < 0.65
