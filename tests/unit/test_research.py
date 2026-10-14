"""End-to-end test of `run_research` with fake providers and a fake judge."""

import asyncio
import json

from searchprobe.cases.io import save_cases
from searchprobe.cases.models import ExcludeConstraint, TestCase, TopicConstraint
from searchprobe.cases.twins import attach_twins
from searchprobe.config import Settings
from searchprobe.pipeline.research import ResearchProfile, run_research
from tests.fakes import FakeAnthropic, FakeProvider


def _judge(kwargs):
    schema = kwargs["output_config"]["format"]["schema"]
    ids = schema["properties"]["verdicts"]["items"]["properties"]["constraint_id"]["enum"]
    prompt = kwargs["messages"][0]["content"]
    bad = "state-owned" in prompt.split("<document>")[1]
    return {"relevance": 1 if bad else 3, "rationale": "",
            "verdicts": [{"constraint_id": c, "verdict": "violated" if (bad and c == "c2")
                          else "satisfied", "quote": ""} for c in ids]}


def test_research_pipeline_writes_reports(tmp_path, monkeypatch):
    cases = attach_twins([
        TestCase(text=f"{t} that are NOT state-owned", category="negation",
                 operators=["negation"],
                 constraints=[TopicConstraint(id="c1", subject=t),
                              ExcludeConstraint(id="c2", concept="state-owned")])
        for t in ["Banks", "Airlines", "Railways"]
    ])
    save_cases(cases, tmp_path / "cases.jsonl")

    def engine(query):
        topic = query.split()[0].lower()
        return [f"https://{topic}.example/state-owned", f"https://{topic}.example/private"]

    def content(query, url):
        return "A state-owned operator" if "state-owned" in url else "A private company"

    fake = FakeProvider(engine, cost=0.001, name="exa", content=content)
    from searchprobe.providers import registry

    monkeypatch.setattr(registry.ProviderRegistry, "get_provider",
                        classmethod(lambda cls, name, settings=None: fake))
    import searchprobe.llm as llm

    monkeypatch.setattr(llm, "get_async_client", lambda settings=None: FakeAnthropic(_judge))

    settings = Settings(_env_file=None, anthropic_api_key="x",
                        database_path=str(tmp_path / "db.sqlite"))
    profile = ResearchProfile(name="t", cases=str(tmp_path / "cases.jsonl"), systems=["exa:neural"],
                              fetch=False, out_dir=str(tmp_path / "out"))
    summary = asyncio.run(run_research(profile, settings, log=lambda *a: None))

    out = tmp_path / "out"
    assert (out / "scorecard.md").exists() and (out / "metamorphic.md").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["profile"]["name"] == "t" and manifest["run_id"] == summary["run_id"]
    assert summary["violation_at_k"]["exa:neural"] == 0.5
    assert "Failure exhibits" in (out / "scorecard.md").read_text()


def test_profile_rejects_unknown_keys(tmp_path):
    import pytest
    from pydantic import ValidationError

    (tmp_path / "p.toml").write_text('name = "x"\nsystems = ["brave"]\nbudjet = 3\n')
    with pytest.raises(ValidationError):
        ResearchProfile.from_toml(tmp_path / "p.toml")


def test_each_exa_backed_system_runs_only_its_own_mode(tmp_path):
    from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
    from searchprobe.storage import Database

    profile = ResearchProfile(name="t", systems=["exa:neural", "aris+exa:auto", "brave"])
    providers, modes = profile.providers_and_modes()
    assert providers == ["exa", "aris+exa", "brave"]
    runner = BenchmarkRunner(
        BenchmarkConfig(providers=providers, modes_by_provider=modes),
        settings=Settings(_env_file=None), db=Database(str(tmp_path / "db.sqlite")),
    )
    tasks = runner._build_tasks([{"id": "q", "text": "x", "category": "negation"}])
    assert {(t.provider_name, t.search_mode) for t in tasks} == {
        ("exa", "neural"), ("aris+exa", "auto"), ("brave", None)}
