"""Tests for metamorphic testing and the Operator Blindness Index."""

import asyncio

import pytest

from searchprobe.cases.models import ExcludeConstraint, TestCase, TopicConstraint
from searchprobe.cases.twins import attach_twins
from searchprobe.metamorphic.engine import MetamorphicEngine
from searchprobe.metamorphic.obi import compute_obi, normalized, summarize
from searchprobe.metamorphic.report import render_markdown, save_run
from searchprobe.metamorphic.variants import build_variants, typo_variant
from searchprobe.pipeline.cost_tracker import CostTracker
from tests.fakes import FakeProvider

TOPICS = ["banks", "airlines", "bakeries", "law firms", "hospitals", "breweries"]


def _cases():
    bases = [
        TestCase(
            text=f"{t.capitalize()} that are NOT owned by the government",
            category="negation",
            operators=["negation"],
            constraints=[TopicConstraint(id="c1", subject=t),
                         ExcludeConstraint(id="c2", concept="government-owned")],
        )
        for t in TOPICS
    ]
    return attach_twins(bases)


def _topic(query: str) -> str:
    q = query.lower()
    return next(t for t in TOPICS if t in q)


def blind_engine(query: str) -> list[str]:
    """Ignores negation entirely: results depend on topic only."""
    t = _topic(query).replace(" ", "-")
    return [f"https://{t}.example/{i}" for i in range(10)]


def aware_engine(query: str) -> list[str]:
    """Returns different documents when the negation is removed."""
    t = _topic(query).replace(" ", "-")
    polarity = "private" if "not" in query.lower() else "state"
    return [f"https://{t}.example/{polarity}/{i}" for i in range(10)]


def test_typo_variant_is_deterministic_and_small():
    import random

    a = typo_variant("Companies headquartered in Switzerland", random.Random(1))
    b = typo_variant("Companies headquartered in Switzerland", random.Random(1))
    assert a == b and a != "Companies headquartered in Switzerland"
    assert len(a) == len("Companies headquartered in Switzerland")


def test_normalized_handles_degenerate_noise():
    assert normalized(0.5, noise=0.2, floor=0.2) != normalized(0.5, noise=0.2, floor=0.2)  # NaN
    assert normalized(1.0, noise=1.0, floor=0.0) == 1.0
    assert normalized(0.0, noise=1.0, floor=0.0) == 0.0


def test_obi_separates_blind_from_aware_engines(tmp_path):
    cases = _cases()
    variants = build_variants(cases, repeats=1, typos=False)
    assert {v.kind for v in variants} == {"REPEAT", "DIR"}
    providers = {
        "blind": FakeProvider(blind_engine, name="blind"),
        "aware": FakeProvider(aware_engine, name="aware"),
    }
    engine = MetamorphicEngine(providers, cost_tracker=CostTracker(budget_limit=100))
    run = asyncio.run(engine.run(cases, variants))
    dir_rows, _ = compute_obi(run.results)
    overall = {s.system: s for s in summarize(dir_rows) if s.group == "ALL"}
    assert overall["blind"].obi.mean == pytest.approx(1.0)
    assert overall["aware"].obi.mean == pytest.approx(0.0)
    assert overall["blind"].n == len(TOPICS)

    md = render_markdown(run)
    assert "Operator Blindness Index" in md and "negation_flip" in md
    run_id = save_run(tmp_path / "m.db", run)
    assert run_id


def test_budget_stops_searches():
    cases = _cases()
    variants = build_variants(cases, repeats=1, typos=False)
    provider = FakeProvider(blind_engine, cost=1.0, name="p")
    engine = MetamorphicEngine({"p": provider}, cost_tracker=CostTracker(budget_limit=3))
    run = asyncio.run(engine.run(cases, variants))
    assert len(provider.calls) == 3
    assert run.skipped_for_budget > 0
