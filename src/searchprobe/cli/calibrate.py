"""Calibrate the LLM judge against human labels."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Measure judge reliability against human labels (Cohen's kappa).")


def _load(settings):
    from searchprobe.calibration.store import HumanLabelStore
    from searchprobe.cases.convert import db_query_to_case
    from searchprobe.judging.store import JudgmentStore
    from searchprobe.storage import Database

    db = Database(settings.database_path)
    judgments = JudgmentStore(settings.database_path).all()
    judgments = [j for j in judgments if j.judge_id != "checker"]
    cases = {}
    for qs in db.get_queries():
        case = db_query_to_case(qs)
        cases[case.id] = case
    return db, judgments, cases, HumanLabelStore(settings.database_path)


@app.command("label")
def label(
    n: int = typer.Option(200, "--n", help="Target number of labeled pairs"),
    labeler: str = typer.Option("human", "--labeler"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Interactively label a stratified sample of judged (case, document) pairs.

    Keys: s = satisfied, v = violated, u = unaddressed, k = skip pair, q = quit.
    The judge's verdicts are hidden while you label.
    """
    from searchprobe.calibration.store import RELEVANCE_KEY, stratified_sample
    from searchprobe.evidence.passages import select_passages
    from searchprobe.evidence.store import EvidenceStore

    settings = get_settings()
    _, judgments, cases, labels = _load(settings)
    done = labels.labeled_pairs(labeler)
    remaining = max(0, n - len(done))
    if remaining == 0:
        console.print(f"[green]{len(done)} pairs already labeled by {labeler}.[/green]")
        return
    items = stratified_sample(judgments, cases, n=remaining, seed=seed, exclude=done)
    evidence = EvidenceStore(settings.database_path)
    keymap = {"s": "satisfied", "v": "violated", "u": "unaddressed"}

    for i, item in enumerate(items, 1):
        case, j = item.case, item.judgment
        snap = evidence.get(j.content_hash)
        text = snap.text if snap and snap.ok else "(page text unavailable; judge saw the snippet)"
        excerpt = select_passages(text, case.text, max_chars=2500)
        console.print(
            Panel(
                f"[bold]Query:[/bold] {case.text}\n[bold]URL:[/bold] {j.url}\n\n{excerpt}",
                title=f"Pair {i}/{len(items)} ({len(done) + i - 1} labeled so far)",
            )
        )
        skip = False
        for c in case.constraints:
            while True:
                ans = (
                    typer.prompt(f"  {c.id} [{c.type}] {c.describe()}  (s/v/u/k/q)").strip().lower()
                )
                if ans in keymap:
                    labels.put(case.id, j.content_hash, c.id, keymap[ans], labeler)
                    break
                if ans == "k":
                    skip = True
                    break
                if ans == "q":
                    console.print("[yellow]Stopped. Progress saved.[/yellow]")
                    return
            if skip:
                break
        if skip:
            continue
        while True:
            rel = typer.prompt("  Relevance 0-3 (0 unrelated ... 3 fully answers)").strip()
            if rel in {"0", "1", "2", "3"}:
                labels.put(case.id, j.content_hash, RELEVANCE_KEY, rel, labeler)
                break
    console.print("[green]Done.[/green] Run `searchprobe calibrate report`.")


@app.command("report")
def report(labeler: str | None = typer.Option(None, "--labeler")) -> None:
    """Agreement between the judge and human labels (per constraint type)."""
    from searchprobe.calibration.store import agreement_report

    settings = get_settings()
    _, judgments, cases, labels = _load(settings)
    rep = agreement_report(labels.all(labeler), judgments, cases)
    if rep.n_pairs == 0:
        console.print("[yellow]No labeled pairs yet. Run `searchprobe calibrate label`.[/yellow]")
        raise typer.Exit(0)
    for line in rep.lines():
        console.print(line)
    if rep.overall_verdict_kappa is not None and rep.overall_verdict_kappa < 0.6:
        console.print(
            "[yellow]Verdict agreement is below 0.6: treat judge-based metrics as provisional "
            "and consider a stronger judge model or higher effort.[/yellow]"
        )
