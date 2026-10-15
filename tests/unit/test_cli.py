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


def test_evaluate_summary_survives_runs_where_everything_errored(capsys):
    from searchprobe.cli.evaluate import _display_summary

    _display_summary([{"provider": "brave", "status": "search_error", "weighted_score": 0.0},
                      {"provider": "brave", "status": "judge_error", "weighted_score": 0.0}],
                     show_comparisons=True)
