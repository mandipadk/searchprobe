"""One-command research runs: cases -> search -> judge -> scorecard -> metamorphic.

A profile (TOML) names the cases, the systems and the budgets; the run writes
a self-contained report directory with a manifest, so results can be cited and
reproduced (``searchprobe scorecard --run-id ...`` recomputes from cache).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResearchProfile(BaseModel):
    """Configuration of a research run (extra keys are rejected to catch typos)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: str = "data/cases/v1"
    categories: list[str] = Field(default_factory=list)
    limit: int | None = None
    systems: list[str] = Field(
        ..., min_length=1, description="e.g. exa:neural, brave, aris+exa:neural"
    )
    num_results: int = 10
    k: int = 10
    budget_search: float = 5.0
    budget_judge: float = 10.0
    judge: bool = True
    fetch: bool = True
    metamorphic: bool = True
    metamorphic_repeats: int = 1
    out_dir: str | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> ResearchProfile:
        with open(path, "rb") as f:
            return cls.model_validate(tomllib.load(f))

    def providers_and_modes(self) -> tuple[list[str], list[str]]:
        """Split system keys into runner providers and the Exa mode list."""
        providers: list[str] = []
        exa_modes: list[str] = []
        for key in self.systems:
            name, _, mode = key.partition(":")
            if name not in providers:
                providers.append(name)
            if mode and (name == "exa" or name.endswith("+exa")) and mode not in exa_modes:
                exa_modes.append(mode)
        return providers, exa_modes or ["auto"]


async def run_research(profile: ResearchProfile, settings: Any, log=print) -> dict[str, Any]:
    """Execute a profile; returns paths and headline numbers."""
    from searchprobe.cases.io import load_cases
    from searchprobe.cli.judge import build_engine
    from searchprobe.manifest import build_manifest
    from searchprobe.metrics.render import scorecard_markdown
    from searchprobe.metrics.scorecard import build_scorecard, compute_list_metrics
    from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
    from searchprobe.storage import Database

    out_dir = Path(profile.out_dir or f"reports/{profile.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.database_path)

    cases = load_cases(profile.cases)
    if profile.categories:
        wanted = set(profile.categories)
        keep = {c.id for c in cases if c.category.value in wanted}
        cases = [c for c in cases if c.id in keep or c.twin_of in keep]
    if profile.limit:
        bases = [c for c in cases if not c.twin_of][: profile.limit]
        ids = {c.id for c in bases} | {t for c in bases for t in c.twins.values()}
        cases = [c for c in cases if c.id in ids]
    log(f"{len(cases)} cases ({sum(not c.twin_of for c in cases)} base)")

    qs_id = db.add_cases(cases, name=f"research:{profile.name}")
    queries = db.get_queries(query_set_id=qs_id)
    providers, exa_modes = profile.providers_and_modes()
    runner = BenchmarkRunner(
        BenchmarkConfig(
            providers=providers,
            exa_modes=exa_modes,
            num_results=profile.num_results,
            budget_limit=profile.budget_search,
        ),
        settings=settings,
        db=db,
    )
    bench = await runner.run(queries, run_name=f"research:{profile.name}")
    log(
        f"Search run {bench.run_id}: {bench.successful_queries} ok, {bench.failed_queries} errors, "
        f"{bench.skipped_for_budget} skipped for budget, ${bench.total_cost:.4f}"
    )

    summary: dict[str, Any] = {
        "run_id": bench.run_id,
        "out_dir": str(out_dir),
        "search_cost": bench.total_cost,
    }
    if profile.judge:
        engine, fetcher = build_engine(
            settings,
            allow_llm=True,
            fetch=profile.fetch,
            budget=profile.budget_judge,
            concurrency=8,
        )
        try:
            judged = await engine.judge_run(bench.run_id, systems=profile.systems, k=profile.k)
        finally:
            if fetcher is not None:
                await fetcher.close()
        card = build_scorecard(compute_list_metrics(judged, k=profile.k))
        manifest = (db.get_run(bench.run_id) or {}).get("config", {}).get("manifest")
        (out_dir / "scorecard.md").write_text(
            scorecard_markdown(
                card,
                judged,
                k=profile.k,
                judge_note=f"Judge: {engine.judge.judge_id}",
                manifest=manifest,
            )
        )
        (out_dir / "scorecard.json").write_text(json.dumps(card.to_dict(), indent=2, default=str))
        summary["judge_cost"] = judged.cost_usd
        summary["violation_at_k"] = {s: c["violation_at_k"].mean for s, c in card.by_system.items()}
        log(f"Judged {judged.judged_pairs} pairs (${judged.cost_usd:.4f}); wrote scorecard")

    if profile.metamorphic:
        from searchprobe.cli.metamorphic import build_providers
        from searchprobe.metamorphic.engine import MetamorphicEngine
        from searchprobe.metamorphic.report import render_markdown, save_run
        from searchprobe.metamorphic.variants import build_variants
        from searchprobe.pipeline.cost_tracker import CostTracker

        variants = build_variants(cases, repeats=profile.metamorphic_repeats)
        if variants:
            meta = MetamorphicEngine(
                build_providers(profile.systems, settings),
                num_results=profile.num_results,
                cost_tracker=CostTracker(budget_limit=profile.budget_search),
            )
            mrun = await meta.run(cases, variants)
            summary["metamorphic_run_id"] = save_run(settings.database_path, mrun)
            (out_dir / "metamorphic.md").write_text(render_markdown(mrun))
            summary["metamorphic_cost"] = mrun.cost_usd
            log(f"Metamorphic: {mrun.searches} searches (${mrun.cost_usd:.4f}); wrote report")

    manifest = build_manifest(settings, profile=profile.model_dump(), **summary)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return summary
