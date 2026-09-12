import json
import pathlib
import re

import pandas as pd
import pytest

from analysis import predict_next, run_backtest, run_panel
from src import llm_panel

ROOT = pathlib.Path(__file__).parent.parent
STUDY = {
    "xi": 0.003,
    "weight": 0.2,
    "weights": {"opening": 0.2, "closing": 0.0},
    "edges": {"opening": 0.05, "closing": 0.0},
    "books": {"opening": {"demonstrable_edge": True}},
}


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_next, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_settings_are_conservative_before_any_study_has_run(results):
    chosen = predict_next.settings()
    assert chosen["weight"] == 1.0
    assert chosen["demonstrable_edge"] is False


def test_settings_follow_the_studies_once_they_exist(results):
    (results / "backtest.json").write_text(json.dumps({"xi": 0.003}))
    (results / "betting.json").write_text(json.dumps(STUDY))
    chosen = predict_next.settings()
    assert chosen == {"xi": 0.003, "weight": 0.2, "min_edge": 0.05, "demonstrable_edge": True}


def test_a_study_without_a_demonstrable_edge_keeps_stakes_off(results):
    losing = {**STUDY, "books": {"opening": {"demonstrable_edge": False}}}
    (results / "betting.json").write_text(json.dumps(losing))
    assert predict_next.settings()["demonstrable_edge"] is False


def test_matches_already_under_way_are_not_forecast():
    fixtures = pd.DataFrame({"Datetime": [pd.Timestamp("2026-09-12 15:00"), pd.Timestamp("2026-09-13 16:30")]})
    ahead = predict_next.upcoming(fixtures, pd.Timestamp("2026-09-12 18:00"))
    assert len(ahead) == 1
    assert ahead["Datetime"].iloc[0] == pd.Timestamp("2026-09-13 16:30")


def test_the_weekly_routine_calls_only_modules_that_exist():
    script = (ROOT / "analysis" / "weekly.sh").read_text()
    modules = re.findall(r"-m ([a-z_]+\.[a-z_]+)", script)
    assert modules
    for module in modules:
        assert (ROOT / f"{module.replace('.', '/')}.py").exists(), module


def test_the_weekly_routine_stops_on_the_first_failure():
    assert "set -euo pipefail" in (ROOT / "analysis" / "weekly.sh").read_text()


UPCOMING = pd.DataFrame({
    "Datetime": [pd.Timestamp("2026-09-19 14:00"), pd.Timestamp("2026-09-19 16:30")],
    "HomeTeam": ["Arsenal", "Hull"],
    "AwayTeam": ["Chelsea", "Man United"],
})
RECORDED = pd.DataFrame({
    "model": ["a:free"],
    "asked_at": [pd.Timestamp("2026-09-18 09:00")],
    "Datetime": [pd.Timestamp("2026-09-19 14:00")],
    "HomeTeam": ["Arsenal"],
    "AwayTeam": ["Chelsea"],
})


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setattr(run_panel, "load_fixtures", lambda: UPCOMING)
    monkeypatch.setattr(run_panel, "free_models", lambda limit=6: ["a:free"])
    return run_panel


def test_the_panel_is_not_asked_when_every_fixture_is_already_on_file(panel, monkeypatch):
    monkeypatch.setattr(panel, "load_records", lambda: RECORDED)
    monkeypatch.setattr(panel, "free_models", _refuse("the roster was fetched with nothing to ask"))
    monkeypatch.setattr(panel, "run_panel", _refuse("a model was asked about a recorded fixture"))
    monkeypatch.setattr(panel, "load_fixtures", lambda: UPCOMING.head(1))
    assert panel.ask(pd.Timestamp("2026-09-18 10:00")) == {}


def test_only_the_fixtures_with_no_forecast_on_file_are_asked(panel, monkeypatch):
    seen = {}

    def capture(fixtures, models):
        seen["asked"] = fixtures
        return {}

    monkeypatch.setattr(panel, "load_records", lambda: RECORDED)
    monkeypatch.setattr(panel, "run_panel", capture)
    panel.ask(pd.Timestamp("2026-09-18 10:00"))
    assert list(seen["asked"]["HomeTeam"]) == ["Hull"]


def _refuse(message):
    def refuse(*args, **kwargs):
        raise AssertionError(message)
    return refuse


def test_the_report_snapshot_is_published_from_the_archive(panel, monkeypatch, tmp_path):
    answers = llm_panel.parse_response(json.dumps([
        {"home": "Arsenal", "away": "Chelsea", "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25},
        {"home": "Hull", "away": "Man United", "p_home": 0.2, "p_draw": 0.3, "p_away": 0.5},
    ]), UPCOMING)
    archive = tmp_path / "llm_panel"
    llm_panel.record({"a:free": answers}, UPCOMING, asked_at=pd.Timestamp("2026-09-18 09:00"), directory=archive)
    monkeypatch.setattr(panel, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(panel, "load_records", lambda: llm_panel.load_records(archive))
    written = json.loads(panel.publish(pd.Timestamp("2026-09-18 10:00")).read_text())
    assert written["models"] == ["a:free"]
    assert len(written["fixtures"]) == 2
    assert written["asked_at"].startswith("2026-09-18T09:00")


def test_nothing_is_published_when_no_recorded_fixture_is_ahead(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(panel, "load_records", lambda: llm_panel.load_records(tmp_path))
    assert panel.publish(pd.Timestamp("2026-09-18 10:00")) is None


WORKFLOW = ROOT.parent / ".github" / "workflows" / "forecast.yml"


def test_the_scheduled_workflow_calls_only_modules_that_exist():
    modules = re.findall(r"-m ([a-z_]+\.[a-z_]+)", WORKFLOW.read_text())
    assert modules
    for module in modules:
        assert (ROOT / f"{module.replace('.', '/')}.py").exists(), module


def test_the_scheduled_workflow_skips_the_panel_without_a_key():
    assert "OPENROUTER_API_KEY" in WORKFLOW.read_text()


def prediction_frame(probabilities):
    return pd.DataFrame({
        "Datetime": [pd.Timestamp("2026-09-19 14:00"), pd.Timestamp("2026-09-20 16:30")],
        "FTHG": [2, 0], "FTAG": [0, 1],
        "p_home": [p[0] for p in probabilities],
        "p_draw": [p[1] for p in probabilities],
        "p_away": [p[2] for p in probabilities],
    })


def test_significance_names_each_pair_it_was_asked_for():
    predictions = {
        "market_closing": prediction_frame([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]),
        "model": prediction_frame([[0.34, 0.33, 0.33], [0.33, 0.33, 0.34]]),
    }
    common = predictions["model"].index
    found = run_backtest.significance(predictions, common, [("market_closing", "model")], draws=100)
    assert list(found) == ["market_closing_vs_model"]
    assert found["market_closing_vs_model"]["mean"] > 0


def test_significance_reports_an_interval_that_can_cross_zero():
    even = prediction_frame([[0.34, 0.33, 0.33], [0.33, 0.33, 0.34]])
    predictions = {"market_closing": even, "model": even.copy()}
    found = run_backtest.significance(predictions, even.index, [("market_closing", "model")], draws=100)
    assert found["market_closing_vs_model"]["low"] == 0.0


def test_the_workflow_restores_only_the_panel_archive():
    # The studies are versioned on main and main is the authority. Restoring the
    # whole results directory from the branch overwrites them with an older copy,
    # and the report is then built from numbers no longer in the repository.
    restored = re.findall(r'git checkout "origin/\$RESULTS_BRANCH" -- "([^"]+)"', WORKFLOW.read_text())
    assert restored
    for path in restored:
        assert path.endswith("results/llm_panel"), path
