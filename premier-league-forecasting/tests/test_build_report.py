import json

import pytest

from analysis import build_report

BASELINE = {
    "matches": 2280,
    "closing_margin_mean": 0.043,
    "calibration": [
        {"lower": 0.0, "upper": 0.1, "n": 258, "predicted": 0.074, "observed": 0.054},
        {"lower": 0.5, "upper": 0.6, "n": 557, "predicted": 0.551, "observed": 0.560},
    ],
}
BACKTEST = {
    "xi": 0.0018,
    "test_seasons": ["2023-24"],
    "matches": 380,
    "results": {
        "market_closing": {"n": 380, "rps": 0.19, "log_loss": 0.96, "brier": 0.57, "ignorance": 1.38, "accuracy": 0.55},
        "dixon_coles": {"n": 380, "rps": 0.21, "log_loss": 1.01, "brier": 0.60, "ignorance": 1.45, "accuracy": 0.52},
    },
}
MATCHWEEK = {
    "generated_at": "2026-09-12T03:30:00",
    "settings": {"xi": 0.0018, "weight": 0.4},
    "fixtures": [{
        "Season": "2026-27", "Datetime": "2026-09-13T16:30:00",
        "HomeTeam": "Nott'm Forest", "AwayTeam": "Man United",
        "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3,
        "p_home_market": 0.38, "p_draw_market": 0.28, "p_away_market": 0.34,
    }],
}


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(build_report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(build_report, "REPORT_PATH", tmp_path / "report.html")
    return tmp_path


def write(directory, name, payload):
    (directory / name).write_text(json.dumps(payload))


def test_the_page_renders_with_nothing_to_report(results):
    page = build_report.build()
    assert "Not yet tested" in page
    assert "no results yet" in page


def test_the_page_carries_no_document_wrapper(results):
    page = build_report.build()
    for tag in ("<!doctype", "<html", "<head>", "<body"):
        assert tag not in page.lower()
    assert page.startswith("<title>")


def test_team_names_with_apostrophes_are_escaped(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    page = build_report.build()
    assert "Nott&#x27;m Forest" in page


def test_the_matchweek_table_shows_each_fixture_once(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    page = build_report.build()
    assert page.count("Man United") == 1
    assert "40%" in page


def test_the_market_ghost_bar_is_drawn_when_market_probabilities_exist(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    assert 'class="ghost"' in build_report.build()


def test_standings_rank_by_rps_and_flag_the_market(results):
    write(results, "backtest.json", BACKTEST)
    page = build_report.build()
    assert page.index("market closing") < page.index("dixon coles")
    assert "+0.02000" in page


def test_calibration_points_stay_inside_the_drawing(results):
    write(results, "baseline.json", BASELINE)
    page = build_report.build()
    assert "<svg" in page
    assert "Stated probability" in page


def test_a_losing_study_reports_no_demonstrable_edge(results):
    write(results, "betting.json", {
        "weight": 0.4, "accuracy": {"market": {"rps": 0.19}, "model": {"rps": 0.21}, "blend": {"rps": 0.195}},
        "books": {"opening": {"roi": -0.03, "bets": 120, "ci_low": -0.12, "ci_high": 0.05, "demonstrable_edge": False}},
    })
    page = build_report.build()
    assert "No demonstrable edge" in page
    assert 'class="verdict is-positive"' not in page


def test_a_winning_study_flips_the_verdict(results):
    write(results, "betting.json", {
        "weight": 0.4, "accuracy": {"market": {"rps": 0.19}, "blend": {"rps": 0.188}},
        "books": {"opening": {"roi": 0.06, "bets": 200, "ci_low": 0.01, "ci_high": 0.11, "demonstrable_edge": True}},
    })
    page = build_report.build()
    assert "Edge demonstrable" in page
    assert 'class="verdict is-positive"' in page


def test_every_colour_token_is_defined_in_the_bare_root(results):
    page = build_report.build()
    bare = page.split(":root:not(")[0]
    for token in ("--ground", "--surface", "--ink", "--accent", "--home", "--draw", "--away", "--good", "--critical"):
        assert f"{token}:" in bare


def test_the_page_is_written_to_disk(results):
    path = build_report.main()
    assert path.exists()
    assert path.read_text().startswith("<title>")
