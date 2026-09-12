import json
import pathlib
import re

import pandas as pd
import pytest

from analysis import predict_next

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
