"""Regression tests for bugs found in code review of the measurement redesign."""

import asyncio
from datetime import date

import httpx
import pytest

from searchprobe.adversarial.fitness import FitnessEvaluator
from searchprobe.adversarial.models import Individual
from searchprobe.cases.models import (
    DateConstraint,
    KnownEntity,
    NumericConstraint,
    TestCase,
    TopicConstraint,
)
from searchprobe.config import Settings
from searchprobe.pipeline.cost_tracker import CostTracker
from searchprobe.providers.models import SearchRequest, SearchResponse
from tests.fakes import FakeAnthropic, FakeProvider

# --- cost accounting -------------------------------------------------------------


def test_tavily_estimate_uses_the_mode_it_runs_by_default():
    from searchprobe.providers.tavily import TavilyProvider

    tavily = TavilyProvider("key")
    assert tavily.estimate_cost(SearchRequest(query="q")) == tavily.COST_PER_QUERY["advanced"]


def test_aris_cached_parse_is_not_charged_again():
    from aris.wrapper import ConstraintAwareSearch

    def llm(kwargs):
        return {"constraints": [{"id": "c1", "type": "topic", "subject": "a company",
                                 "hard": True, "description": ""}], "rewrites": []}

    aris = ConstraintAwareSearch(FakeProvider(), FakeAnthropic(llm), "claude-opus-5-5", "v")
    first = asyncio.run(aris.parse("companies"))
    again = asyncio.run(aris.parse("companies"))
    assert first.cost_usd > 0
    assert again.cost_usd == 0.0 and again.case == first.case


def test_discovery_llm_mutations_are_charged_to_the_budget():
    from searchprobe.discovery.search import discover

    class Evaluator:
        cost_tracker = CostTracker(budget_limit=0.025)
        budget_exhausted = False

        def _reserve(self, estimate):
            r = self.cost_tracker.try_reserve(estimate)
            if r is None:
                self.budget_exhausted = True
            return r

        async def evaluate(self, case):
            from searchprobe.discovery.search import Evaluation

            return Evaluation(fitness=0.5, target=None, reference=None, answerable=True)

    calls = []

    def llm(kwargs):
        calls.append(1)
        return {"query": f"rephrased {len(calls)}", "change": "x"}

    case = TestCase(text="companies", category="negation",
                    constraints=[TopicConstraint(id="c1", subject="a company")])
    ev = Evaluator()
    asyncio.run(discover([case], ev, iterations=20, llm=(FakeAnthropic(llm), "m"),
                         reevaluate=0, seed=0))
    assert ev.budget_exhausted
    assert ev.cost_tracker.get_total() <= 0.025
    assert len(calls) == 2  # 0.01 reserved per call; the third does not fit


def test_duplicate_queries_in_one_population_are_evaluated_once():
    class Provider:
        calls = 0

        async def search(self, request):
            Provider.calls += 1
            await asyncio.sleep(0.01)
            return SearchResponse(provider="p", query=request.query, cost_usd=1.0)

    class Judge:
        async def evaluate(self, query_id, query_text, category, search_response):
            class E:
                cost_usd = 0.0
                counts_for_quality = True
                weighted_score = 0.3

            return E()

    ev = FitnessEvaluator(mode="llm_judge", provider=Provider(), judge=Judge(),
                          cost_tracker=CostTracker(budget_limit=100), max_concurrent=4)
    population = [Individual(query="Same query"), Individual(query="same  QUERY")]
    asyncio.run(ev.evaluate_population(population))
    assert Provider.calls == 1
    assert population[0].fitness == population[1].fitness == pytest.approx(0.7)


def test_heuristic_fitness_cache_respects_mutation_depth():
    ev = FitnessEvaluator(mode="embedding_sim")
    shallow = Individual(query="companies not in ai")
    deep = Individual(query="companies not in ai", mutation_history=["a", "b", "c", "d"])
    asyncio.run(ev.evaluate_population([shallow]))
    asyncio.run(ev.evaluate_population([deep]))
    assert deep.fitness != shallow.fitness


# --- pipeline ----------------------------------------------------------------------


def test_runner_survives_a_provider_that_cannot_be_created(tmp_path, monkeypatch):
    from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
    from searchprobe.providers import registry
    from searchprobe.storage import Database

    fake = FakeProvider({"q": ["https://a.example/"]}, name="brave")

    def get_provider(cls, name, settings=None):
        if name == "tavily":
            raise ValueError("TAVILY_API_KEY not set")
        return fake

    monkeypatch.setattr(registry.ProviderRegistry, "get_provider", classmethod(get_provider))
    db = Database(str(tmp_path / "db.sqlite"))
    qs = db.create_query_set(name="t")
    qid = db.add_query(qs, "q", "negation")
    runner = BenchmarkRunner(BenchmarkConfig(providers=["brave", "tavily"]),
                             settings=Settings(_env_file=None), db=db)
    result = asyncio.run(runner.run(db.get_queries(query_set_id=qs)))
    assert result.status == "completed"
    errors = [r for r in result.results if r.error]
    assert len(errors) == 1 and "provider unavailable" in errors[0].error
    assert db.get_run(result.run_id)["status"] == "completed"
    assert qid


def test_runner_rejects_an_empty_query_list(tmp_path):
    from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
    from searchprobe.storage import Database

    runner = BenchmarkRunner(BenchmarkConfig(providers=["brave"]),
                             settings=Settings(_env_file=None),
                             db=Database(str(tmp_path / "db.sqlite")))
    with pytest.raises(ValueError):
        asyncio.run(runner.run([]))


def test_bare_exa_systems_match_the_stored_auto_mode():
    from searchprobe.pipeline.research import ResearchProfile

    profile = ResearchProfile(name="t", systems=["exa", "aris+exa", "exa:neural", "brave"])
    assert profile.canonical_systems() == ["exa:auto", "aris+exa:auto", "exa:neural", "brave"]


def test_metamorphic_treats_empty_results_as_failed_searches():
    from searchprobe.metamorphic.engine import MetamorphicEngine, MetamorphicRun

    engine = MetamorphicEngine({"p": FakeProvider({}, name="p")})
    response, error = asyncio.run(engine._search("p", "anything", MetamorphicRun()))
    assert response is None and error == "empty results"


# --- cases and evidence ------------------------------------------------------------


def test_v1_conversion_always_keeps_the_query_as_a_hard_criterion():
    from searchprobe.cases.convert import query_to_case

    case = query_to_case("Machine learning tutorials in German", "cross_lingual",
                         {"expected_language": "de"})
    assert [c.type for c in case.hard_constraints] == ["language", "freeform"]


def test_llm_twin_identical_to_its_base_is_rejected():
    from searchprobe.cases.generators.llm import build_cases
    from searchprobe.queries.taxonomy import AdversarialCategory

    def flat(**kw):
        base = {k: None for k in ["subject", "concept", "attribute", "op", "value",
                                  "value_max", "unit", "field", "start", "end", "name",
                                  "predicate", "object", "code", "requirement"]}
        base.update({"wrong": [], "allowed": [], "disallowed": [], "hard": True,
                     "description": ""})
        base.update(kw)
        return base

    constraints = [flat(id="c1", type="topic", subject="a river"),
                   flat(id="c2", type="numeric", attribute="length", op="gt", value=100)]
    item = {"text": "Rivers longer than 100 km", "difficulty": "hard",
            "adversarial_reason": "r", "constraints": constraints,
            "twin": {"text": "rivers  longer than 100 KM", "relation": "llm_twin",
                     "constraints": constraints}}
    cases, rejected = build_cases([item], AdversarialCategory.NUMERIC_PRECISION, "m")
    base = next(c for c in cases if c.text == "Rivers longer than 100 km")
    assert base.id not in base.twins.values()
    assert any("twin" in r for r in rejected)


def test_negatives_of_twins_are_decided_by_checkers():
    from searchprobe.cases.generators.wikidata import _violating
    from searchprobe.judging.checkers import run_checkers

    france = KnownEntity(label="France", qid="Q142",
                         wikipedia_url="https://en.wikipedia.org/wiki/France")
    twin = TestCase(text="European countries that are members of the EU", category="negation",
                    constraints=[TopicConstraint(id="c1", subject="a European country"),
                                 TopicConstraint(id="c2", subject="an EU member state")],
                    negative_set=_violating([france], "c2"))
    verdicts = run_checkers(twin, "https://en.wikipedia.org/wiki/France", None)
    assert [(v.constraint_id, v.verdict) for v in verdicts] == [("c2", "violated")]


def test_numeric_description_states_the_band_check_accepts():
    c = NumericConstraint(id="c1", attribute="height", op="eq", value=100, tolerance=0.1,
                          unit="m")
    assert c.check(105)
    assert "(accepted: 90 to 110 m)" in c.describe()


def test_date_shift_never_shifts_a_year_twice():
    from searchprobe.cases.twins import date_shift

    case = TestCase(text="Reports published between 2024 and 2025",
                    category="temporal_constraint",
                    constraints=[DateConstraint(id="c1", field="published",
                                                start=date(2024, 1, 1), end=date(2025, 12, 31))])
    twin = date_shift(case, years=1)
    assert twin.text == "Reports published between 2025 and 2026"
    assert twin.constraint("c1").start == date(2025, 1, 1)


def test_empty_generator_selection_runs_nothing(monkeypatch):
    from searchprobe.cases.generators import wikidata
    from searchprobe.cases.generators.wikidata import generate

    async def no_problems(wd, expected=None):
        return []

    monkeypatch.setattr(wikidata, "verify_constants", no_problems)
    monkeypatch.setattr(wikidata, "verify_memberships", no_problems)
    cases, failures = asyncio.run(generate([], client=object()))  # type: ignore[arg-type]
    assert cases == [] and failures == []


def test_same_text_with_different_dates_gets_different_snapshots():
    from searchprobe.evidence.store import content_hash

    assert content_hash("body", meta="Title|2024-01-01") != content_hash(
        "body", meta="Title|2025-01-01")
    assert content_hash("body", meta="Title|2024-01-01") == content_hash(
        "body", meta="Title|2024-01-01")


# --- calibration, Aris ----------------------------------------------------------


def _judgment(case_id, chash, judge_id):
    from searchprobe.judging.models import DocJudgment

    return DocJudgment(case_id=case_id, content_hash=chash, url="https://x.example/",
                       judge_id=judge_id, prompt_version="v", relevance=2, verdicts=[],
                       evidence_source="page")


def test_calibration_samples_each_pair_once_and_refuses_mixed_judges():
    from searchprobe.calibration.store import agreement_report, stratified_sample

    case = TestCase(text="q", category="negation",
                    constraints=[TopicConstraint(id="c1", subject="x")])
    judgments = [_judgment(case.id, "h1", "a@x"), _judgment(case.id, "h1", "b@x")]
    sample = stratified_sample(judgments, {case.id: case}, n=10)
    assert len(sample) == 1
    with pytest.raises(ValueError):
        agreement_report({}, judgments, {case.id: case})


def _web(handler, addresses):
    from aris.sources.web import WebSource

    web = WebSource(public_only=True, resolve=lambda host: addresses[host])
    web._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return web


def test_index_fetch_refuses_redirects_to_private_hosts():
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["host"]))
        return httpx.Response(302, headers={"location": "http://intranet.example/admin"})

    web = _web(handler, {"public.example": ["93.184.216.34"], "intranet.example": ["10.0.0.5"]})
    assert asyncio.run(web.fetch_url("http://public.example/")) is None
    # The public hop was pinned to the checked IP; the private hop was never requested.
    assert seen == [("http://93.184.216.34/", "public.example")]


def test_index_fetch_refuses_hosts_with_any_private_address():
    def handler(request):
        raise AssertionError("must not connect")

    web = _web(handler, {"rebind.example": ["93.184.216.34", "127.0.0.1"]})
    assert asyncio.run(web.fetch_url("https://rebind.example/")) is None


def test_aris_parses_leading_and_trailing_decimal_points():
    from aris.core.numbers import to_float

    assert to_float(".5") == 0.5 and to_float("5.") == 5.0 and to_float(".") is None


# --- third review round -------------------------------------------------------------


def _neg(text, *excludes):
    from searchprobe.cases.models import ExcludeConstraint, FreeformConstraint

    return TestCase(text=text, category="negation", constraints=[
        TopicConstraint(id="c1", subject="a company"),
        *[ExcludeConstraint(id=f"x{i}", concept=e) for i, e in enumerate(excludes)],
        FreeformConstraint(id="q", requirement=f"The result directly satisfies the query: '{text}'"),
    ])


def test_twin_rewrites_the_query_requirement_for_its_own_text():
    from searchprobe.cases.twins import negation_flip

    twin = negation_flip(_neg("Companies that are NOT in the AI industry", "an AI company"))
    assert twin.constraint("q").requirement == (
        "The result directly satisfies the query: 'Companies that are in the AI industry'")


def test_negation_flip_refuses_cases_with_several_exclusions():
    from searchprobe.cases.twins import negation_flip

    assert negation_flip(_neg("Companies NOT in AI or crypto", "AI", "crypto")) is None


def test_twin_with_an_unrelated_free_text_requirement_is_rejected():
    from searchprobe.cases.models import ExcludeConstraint, FreeformConstraint
    from searchprobe.cases.twins import negation_flip

    case = TestCase(text="Companies NOT in AI", category="negation", constraints=[
        ExcludeConstraint(id="c1", concept="AI"),
        FreeformConstraint(id="c2", requirement="Results must not mention machine learning")])
    assert negation_flip(case) is None


def test_numeric_scale_scales_both_bounds_once():
    from searchprobe.cases.twins import numeric_scale

    case = TestCase(text="Cities with between 10 and 100 parks", category="numeric_precision",
                    constraints=[NumericConstraint(id="c1", attribute="parks", op="between",
                                                   value=10, value_max=100)])
    assert numeric_scale(case).text == "Cities with between 100 and 1,000 parks"


def test_argument_swap_refuses_overlapping_names():
    from searchprobe.cases.models import RelationConstraint
    from searchprobe.cases.twins import argument_swap

    case = TestCase(text="Companies that York acquired from New York", category="compositional",
                    constraints=[RelationConstraint(id="c1", subject="York",
                                                    predicate="acquired from",
                                                    object="New York")])
    assert argument_swap(case) is None


def test_judge_fitness_is_cached_per_category_and_failures_are_retried():
    class Provider:
        calls = 0
        fail = True

        async def search(self, request):
            Provider.calls += 1
            if Provider.fail:
                return SearchResponse(provider="p", query=request.query, error="503")
            return SearchResponse(provider="p", query=request.query)

    class Judge:
        async def evaluate(self, query_id, query_text, category, search_response):
            class E:
                cost_usd = 0.0
                counts_for_quality = True
                weighted_score = 0.2 if category == "negation" else 0.6

            return E()

    ev = FitnessEvaluator(mode="llm_judge", provider=Provider(), judge=Judge())
    first = [Individual(query="q", category="negation")]
    asyncio.run(ev.evaluate_population(first))
    assert first[0].fitness == 0.0 and Provider.calls == 1
    Provider.fail = False
    again = [Individual(query="q", category="negation"), Individual(query="q", category="numeric")]
    asyncio.run(ev.evaluate_population(again))
    assert Provider.calls == 3  # the failure was not cached; categories are scored apart
    assert [i.fitness for i in again] == [pytest.approx(0.8), pytest.approx(0.4)]


def test_url_normalization_keeps_content_selecting_parameters():
    from searchprobe.perturbation.stability import normalize_url

    assert normalize_url("https://e.org/p?reference=1") != normalize_url("https://e.org/p?reference=2")
    assert normalize_url("https://e.org/p?ref=twitter&utm_source=x") == "e.org/p"


def test_refetching_unchanged_content_keeps_the_url_fresh(tmp_path):
    from datetime import UTC, datetime, timedelta

    from searchprobe.evidence.store import EvidenceStore, Snapshot

    store = EvidenceStore(tmp_path / "e.db")
    old = datetime.now(UTC) - timedelta(days=30)
    snap = Snapshot(content_hash="h", url="https://e.org", final_url="https://e.org", status="ok",
                    http_status=200, title="t", text="body", fetched_at=old)
    store.put(snap, "e.org")
    assert store.get_for_url("e.org", max_age_days=7) is None
    store.put(snap.model_copy(update={"fetched_at": datetime.now(UTC)}), "e.org")
    assert store.get_for_url("e.org", max_age_days=7) is not None


def test_discovery_continues_when_the_judge_fails():
    from searchprobe.discovery.search import CaseEvaluator

    class FailingJudge:
        model = "m"

        async def judge(self, case, evidence, decided=None):
            raise RuntimeError("refusal")

    case = TestCase(text="companies", category="negation",
                    constraints=[TopicConstraint(id="c1", subject="a company")])
    providers = {"t": FakeProvider(lambda q: ["https://a.example/"], name="t")}
    tracker = CostTracker(budget_limit=1)
    ev = CaseEvaluator(providers, "t", None, judge=FailingJudge(), k=1, cost_tracker=tracker)
    result = asyncio.run(ev.evaluate(case))
    assert result.fitness == 0.0
    assert tracker.reserved_total == 0


def test_numeric_wikidata_values_come_from_best_ranked_statements():
    from searchprobe.cases.generators.wikidata import AREA_KM2, HEIGHT_M

    assert "wikibase:BestRank" in AREA_KM2 and "wikibase:BestRank" in HEIGHT_M


def test_calibration_saves_a_pair_only_when_fully_labeled(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import searchprobe.cli.calibrate as cal
    from searchprobe.calibration.store import HumanLabelStore

    case = TestCase(text="q", category="negation",
                    constraints=[TopicConstraint(id="c1", subject="x"),
                                 TopicConstraint(id="c2", subject="y")])
    labels = HumanLabelStore(tmp_path / "l.db")
    monkeypatch.setattr(cal, "_load", lambda settings, judge_id=None: (
        None, [_judgment(case.id, "h1", "a@x")], {case.id: case}, labels))
    monkeypatch.setattr(cal, "get_settings",
                        lambda: Settings(_env_file=None, database_path=str(tmp_path / "x.db")))
    CliRunner().invoke(cal.app, ["label", "--n", "1"], input="s\nq\n")
    assert labels.labeled_pairs("human") == set()  # quit mid-pair: nothing saved
    CliRunner().invoke(cal.app, ["label", "--n", "1"], input="s\nv\n2\n")
    assert labels.all("human") == {(case.id, "h1", "c1"): "satisfied",
                                   (case.id, "h1", "c2"): "violated",
                                   (case.id, "h1", "__relevance__"): "2"}
