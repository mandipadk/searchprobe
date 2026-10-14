"""End-to-end tests for the measurement core (no network, fake judge)."""

import asyncio
import json
from datetime import date

import httpx
import pytest

from searchprobe.calibration.agreement import cohen_kappa, weighted_kappa
from searchprobe.calibration.store import HumanLabelStore, agreement_report, stratified_sample
from searchprobe.cases.models import (
    DateConstraint,
    ExcludeConstraint,
    KnownEntity,
    LanguageConstraint,
    NumericConstraint,
    TestCase,
    TopicConstraint,
)
from searchprobe.evidence import EvidenceFetcher, EvidenceStore
from searchprobe.evidence.extract import builtin_extract
from searchprobe.evidence.language import detect_language
from searchprobe.judging.checkers import run_checkers
from searchprobe.judging.engine import JudgingEngine
from searchprobe.judging.llm_judge import DocumentJudge
from searchprobe.judging.models import ConstraintVerdict, DocJudgment
from searchprobe.judging.store import JudgmentStore
from searchprobe.metrics.ir import list_metrics
from searchprobe.metrics.render import scorecard_markdown
from searchprobe.metrics.scorecard import build_scorecard, compute_list_metrics
from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
from tests.fakes import FakeAnthropic, FakeProvider, user_prompt

PAGES = {
    "https://acme-plumbing.example/": "<html lang='en'><title>Acme Plumbing</title><body><p>"
    "Acme is a family plumbing business. The founder said he will maintain his email list."
    "</p></body></html>",
    "https://deepmind.example/": "<html lang='en'><title>DeepMind</title><body><p>DeepMind is "
    "an artificial intelligence research laboratory.</p></body></html>",
    "https://bakery.example/": "<html><title>Rosa's Bakery</title><body><p>A bakery selling "
    "bread and pastries since 1990.</p></body></html>",
}


def _transport():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in PAGES:
            return httpx.Response(200, html=PAGES[url])
        return httpx.Response(404, text="nope")

    return httpx.MockTransport(handler)


def _not_ai_case() -> TestCase:
    return TestCase(
        text="Companies that are NOT in the AI industry",
        category="negation",
        operators=["negation"],
        constraints=[
            TopicConstraint(id="c1", subject="a company"),
            ExcludeConstraint(id="c2", concept="an AI company"),
        ],
        negative_set=[
            KnownEntity(label="OpenAI", wikipedia_url="https://en.wikipedia.org/wiki/OpenAI")
        ],
    )


def _fake_judge_payload(kwargs):
    """Judge like a careful human: AI companies violate, others satisfy."""
    prompt = user_prompt(kwargs)
    doc = prompt.split("<document>")[1]
    ids = kwargs["output_config"]["format"]["schema"]["properties"]["verdicts"]["items"][
        "properties"]["constraint_id"]["enum"]
    is_ai = "artificial intelligence" in doc
    verdicts = []
    for cid in ids:
        if cid == "c2":
            verdicts.append({"constraint_id": cid, "verdict": "violated" if is_ai else "satisfied",
                             "quote": "artificial intelligence" if is_ai else ""})
        else:
            verdicts.append({"constraint_id": cid, "verdict": "satisfied", "quote": ""})
    return {"relevance": 1 if is_ai else 3, "verdicts": verdicts, "rationale": "fake"}


class TestEvidence:
    def test_builtin_extract_finds_text_date_and_language(self):
        html = (
            "<html lang='de'><head><title>T</title>"
            "<meta property='article:published_time' content='2025-01-15T10:00:00Z'>"
            "<script>var x=1;</script></head><body><nav>menu</nav>"
            "<p>Der Hund ist nicht in dem Haus und die Katze ist auch nicht da.</p></body></html>"
        )
        ex = builtin_extract(html)
        assert "var x" not in ex.text and "menu" not in ex.text
        assert ex.published_date == date(2025, 1, 15)
        assert ex.language == "de"

    @pytest.mark.parametrize(
        "text,lang",
        [
            ("The quick brown fox is in the house and it was on the table for the day", "en"),
            ("Le chat est dans la maison et le chien est sur la table pour une heure", "fr"),
            ("日本のスタートアップのエコシステムについて", "ja"),
            ("Статьи о машинном обучении и нейронных сетях", "ru"),
        ],
    )
    def test_detect_language(self, text, lang):
        assert detect_language(text)[0] == lang

    def test_fetcher_caches_and_handles_errors(self, tmp_path):
        store = EvidenceStore(tmp_path / "e.db")

        async def go():
            async with httpx.AsyncClient(transport=_transport()) as client:
                f = EvidenceFetcher(store, client=client)
                snaps = await f.fetch_many(
                    ["https://acme-plumbing.example/", "https://missing.example/"]
                )
                again = await f.fetch("https://www.acme-plumbing.example")  # normalized hit
                return snaps, again

        snaps, again = asyncio.run(go())
        ok = snaps["https://acme-plumbing.example/"]
        assert ok.ok and "plumbing" in ok.text
        assert snaps["https://missing.example/"].status == "http_error"
        assert again.content_hash == ok.content_hash


class TestCheckers:
    def test_negative_set_wikipedia_page_violates_exclusion(self):
        case = _not_ai_case()
        verdicts = run_checkers(case, "https://en.wikipedia.org/wiki/OpenAI", None)
        assert [(v.constraint_id, v.verdict) for v in verdicts] == [("c2", "violated")]

    def test_known_numeric_value_is_checked_deterministically(self):
        case = TestCase(
            text="Buildings between 300 and 350 meters tall",
            category="numeric_precision",
            constraints=[NumericConstraint(id="c1", attribute="height", op="between",
                                           value=300, value_max=350, unit="m")],
            answer_set=[
                KnownEntity(label="Tower A", wikipedia_url="https://en.wikipedia.org/wiki/Tower_A",
                            values={"height": 330}),
                KnownEntity(label="Tower B", wikipedia_url="https://en.wikipedia.org/wiki/Tower_B",
                            values={"height": 500}),
            ],
        )
        a = run_checkers(case, "https://en.wikipedia.org/wiki/Tower_A", None)
        b = run_checkers(case, "https://en.wikipedia.org/wiki/Tower_B", None)
        assert a[0].verdict == "satisfied" and b[0].verdict == "violated"

    def test_language_and_published_date_from_snapshot(self, tmp_path):
        from datetime import UTC, datetime

        from searchprobe.evidence.store import Snapshot

        case = TestCase(
            text="ML tutorials auf Deutsch from 2025",
            category="cross_lingual",
            constraints=[
                LanguageConstraint(id="c1", code="de"),
                DateConstraint(id="c2", field="published", start=date(2025, 1, 1)),
            ],
        )
        snap = Snapshot(content_hash="h", url="u", text="x", language="en",
                        language_confidence=0.9, published_date=date(2024, 6, 1),
                        fetched_at=datetime.now(UTC))
        verdicts = {v.constraint_id: v.verdict for v in run_checkers(case, "u", snap)}
        assert verdicts == {"c1": "violated", "c2": "violated"}


class TestMetrics:
    def _j(self, chash, rel, violated):
        return DocJudgment(
            case_id="x", content_hash=chash, url=chash, judge_id="j", prompt_version="p",
            relevance=rel,
            verdicts=[
                ConstraintVerdict(constraint_id="c1", verdict="satisfied"),
                ConstraintVerdict(constraint_id="c2",
                                  verdict="violated" if violated else "satisfied"),
            ],
        )

    def test_violation_and_pooled_ndcg(self):
        case = _not_ai_case()
        good, bad = self._j("g", 3, False), self._j("b", 1, True)
        m = list_metrics(case, "s", [bad, good, None], pool=[good, bad], k=10)
        assert m.violation_at_k == pytest.approx(0.5)
        assert m.coverage == pytest.approx(2 / 3)
        assert m.reciprocal_rank == pytest.approx(0.5)
        perfect = list_metrics(case, "s", [good], pool=[good, bad], k=10)
        assert perfect.ndcg_at_k == pytest.approx(1.0)
        assert 0 < m.ndcg_at_k < 1

    def test_empty_list_is_failure_not_missing(self):
        m = list_metrics(_not_ai_case(), "s", [], pool=[], k=10)
        assert m.satisfied_at_k == 0.0 and m.ndcg_at_k == 0.0 and m.error is None

    def test_search_error_has_no_metrics(self):
        m = list_metrics(_not_ai_case(), "s", [], pool=[], k=10, error="boom")
        assert m.violation_at_k is None and m.error == "boom"


class TestEndToEnd:
    def test_run_judge_scorecard_and_offline_reproduction(self, temp_db):
        case = _not_ai_case()
        qs_id = temp_db.add_cases([case], name="t")
        queries = temp_db.get_queries(query_set_id=qs_id)

        good = FakeProvider(lambda q: ["https://acme-plumbing.example/", "https://bakery.example/"],
                            name="good")
        bad = FakeProvider(lambda q: ["https://deepmind.example/",
                                      "https://en.wikipedia.org/wiki/OpenAI",
                                      "https://acme-plumbing.example/"], name="bad")
        runner = BenchmarkRunner(BenchmarkConfig(providers=["good", "bad"]), db=temp_db)
        runner._get_provider = lambda name: {"good": good, "bad": bad}[name]
        result = asyncio.run(runner.run(queries))

        db_path = temp_db.db_path
        judge_client = FakeAnthropic(_fake_judge_payload)

        async def judge_run(allow_llm):
            async with httpx.AsyncClient(transport=_transport()) as http:
                fetcher = EvidenceFetcher(EvidenceStore(db_path), client=http,
                                          offline=not allow_llm, max_age_days=None)
                engine = JudgingEngine(
                    temp_db, JudgmentStore(db_path),
                    DocumentJudge(judge_client if allow_llm else None, "fake-judge", "low"),
                    fetcher=fetcher, allow_llm=allow_llm,
                )
                return await engine.judge_run(result.run_id)

        run = asyncio.run(judge_run(True))
        # acme page is shared by both systems: judged once. Wikipedia OpenAI page
        # 404s in the mock but is decided by the negative-set checker.
        assert run.judged_pairs == 3
        assert run.checker_only_pairs == 1
        assert len(judge_client.messages.calls) == 3

        card = build_scorecard(compute_list_metrics(run))
        assert card.by_system["good"]["violation_at_k"].mean == 0.0
        assert card.by_system["bad"]["violation_at_k"].mean == pytest.approx(2 / 3)
        md = scorecard_markdown(card, run)
        assert "Failure exhibits" in md and "deepmind.example" in md

        # Offline recompute: no LLM calls, identical numbers.
        offline = asyncio.run(judge_run(False))
        card2 = build_scorecard(compute_list_metrics(offline))
        assert len(judge_client.messages.calls) == 3
        assert json.dumps(card.to_dict(), sort_keys=True) == json.dumps(
            card2.to_dict(), sort_keys=True
        )

        # Calibration: a human agreeing with the judge gives kappa 1.
        labels = HumanLabelStore(db_path)
        judgments = [j for j in JudgmentStore(db_path).all()]
        sample = stratified_sample(judgments, {case.id: case}, n=10)
        assert sample
        for item in sample:
            for c in case.constraints:
                labels.put(case.id, item.judgment.content_hash, c.id,
                           item.judgment.verdict_for(c.id))
        rep = agreement_report(labels.all("human"), judgments, {case.id: case})
        assert rep.n_pairs == len(sample)
        assert rep.overall_verdict_kappa == pytest.approx(1.0)


def test_kappa_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0]) < 0
