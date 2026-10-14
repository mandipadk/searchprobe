"""CLI smoke tests."""

from typer.testing import CliRunner

from searchprobe.cli.app import app


def test_report_list_runs_does_not_generate_a_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCHPROBE_DATABASE_PATH", str(tmp_path / "t.db"))
    from searchprobe.config import get_settings

    get_settings.cache_clear()
    result = CliRunner().invoke(app, ["report", "list-runs"])
    get_settings.cache_clear()
    assert "No runs found to report on" not in result.output
