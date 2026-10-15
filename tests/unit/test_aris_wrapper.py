"""Tests for Aris's ConstraintAwareSearch wrapper (fake base provider and LLM)."""

import asyncio

import httpx

from aris.wrapper import ConstraintAwareSearch, rrf_fuse
from searchprobe.config import Settings
from searchprobe.evidence import EvidenceFetcher, EvidenceStore
from searchprobe.providers.models import SearchRequest, SearchResult
from searchprobe.providers.registry import ProviderRegistry
from tests.fakes import FakeAnthropic, FakeProvider, user_prompt

PAGES = {
    "https://deepmind.example/": "<p>DeepMind is an artificial intelligence lab.</p>",
    "https://openai.example/": "<p>OpenAI builds artificial intelligence models.</p>",
    "https://bakery.example/": "<p>Rosa's Bakery has sold bread since 1990.</p>",
    "https://plumbing.example/": "<p>Acme Plumbing fixes pipes across the county.</p>",
}


def _flat(**kw):
    base = {k: None for k in ["subject", "concept", "attribute", "op", "value", "value_max",
                              "unit", "field", "start", "end", "name", "predicate", "object",
                              "code", "requirement"]}
    base.update({"wrong": [], "allowed": [], "disallowed": [], "hard": True, "description": ""})
    base.update(kw)
    return base


def llm(kwargs):
    schema = kwargs["output_config"]["format"]["schema"]
    if "rewrites" in schema["properties"]:  # parse call
        return {
            "constraints": [_flat(id="c1", type="topic", subject="a company"),
                            _flat(id="c2", type="exclude", concept="an AI company")],
            "rewrites": ["small companies"],
        }
    doc = user_prompt(kwargs).split("<document>")[1]
    ai = "artificial intelligence" in doc
    ids = schema["properties"]["verdicts"]["items"]["properties"]["constraint_id"]["enum"]
    return {
        "relevance": 1 if ai else 3,
        "verdicts": [{"constraint_id": cid,
                      "verdict": "violated" if (cid == "c2" and ai) else "satisfied",
                      "quote": ""} for cid in ids],
        "rationale": "",
    }


def base_engine(query):
    if query == "small companies":
        return ["https://plumbing.example/", "https://bakery.example/"]
    # The base engine is negation-blind: AI companies first.
    return ["https://deepmind.example/", "https://openai.example/", "https://bakery.example/"]


def _transport():
    def handler(request):
        url = str(request.url)
        if url in PAGES:
            return httpx.Response(200, html=f"<html><body>{PAGES[url]}</body></html>")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_rrf_fuse_rewards_agreement():
    a = [SearchResult(title="", url="https://x.com"), SearchResult(title="", url="https://y.com")]
    b = [SearchResult(title="", url="https://y.com")]
    fused = rrf_fuse([a, b])
    assert str(fused[0].result.url).startswith("https://y.com")


def test_wrapper_drops_violations_and_promotes_satisfying(tmp_path):
    base = FakeProvider(base_engine, cost=0.01, name="fake")
    client = FakeAnthropic(llm)

    async def go():
        async with httpx.AsyncClient(transport=_transport()) as http:
            fetcher = EvidenceFetcher(EvidenceStore(tmp_path / "e.db"), client=http)
            aris = ConstraintAwareSearch(base, client, "parser", "verifier", fetcher=fetcher,
                                         fanout=1, pool_size=10)
            return await aris.search(SearchRequest(query="Companies NOT in AI", num_results=3))

    resp = asyncio.run(go())
    urls = [str(r.url) for r in resp.results]
    assert resp.error is None
    assert resp.provider == "aris+fake"
    assert "https://deepmind.example/" not in urls and "https://openai.example/" not in urls
    assert set(urls) == {"https://plumbing.example/", "https://bakery.example/"}
    assert resp.metadata["dropped_violations"] == 2
    assert resp.metadata["rewrites"] == ["small companies"]
    assert resp.cost_usd > 0.02  # base searches + LLM calls
    assert [r.query for r in base.calls] == ["Companies NOT in AI", "small companies"]


def test_registry_wraps_base_provider(monkeypatch):
    settings = Settings(_env_file=None, exa_api_key="x", anthropic_api_key="y",
                        database_path=":memory:")
    provider = ProviderRegistry.get_provider("aris+exa", settings)
    assert provider.name == "aris+exa"
    assert "neural" in provider.SUPPORTED_MODES
