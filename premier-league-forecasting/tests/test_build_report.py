import json
import re

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


PANEL = {
    "asked_at": "2026-09-12T18:51:46",
    "models": ["a:free", "b:free"],
    "fixtures": [{
        "kickoff": "2026-09-13T16:30:00", "home": "Nott'm Forest", "away": "Man United",
        "forecasts": {"a:free": [0.4, 0.3, 0.3], "b:free": [0.5, 0.25, 0.25]},
    }],
}
PANEL_SCORES = {
    "a:free": {"n": 20, "rps": 0.21, "log_loss": 1.0, "brier": 0.6, "ignorance": 1.4, "accuracy": 0.5},
    "market_opening": {"n": 20, "rps": 0.195, "log_loss": 0.96, "brier": 0.57, "ignorance": 1.38, "accuracy": 0.55},
}
LOSING_STUDY = {
    "weight": 0.0,
    "accuracy": {"market": {"rps": 0.19}, "model": {"rps": 0.20}},
    "books": {
        "opening": {"bets": 0, "demonstrable_edge": False},
        "model_only": {
            "bets": 1106, "roi": -0.095, "ci_low": -0.21, "ci_high": 0.025,
            "final_bankroll": 198.24, "max_drawdown": 0.88, "positive_share": 0.0601,
        },
    },
}


def test_the_panel_section_is_skipped_when_no_model_has_been_asked(results):
    assert "language models" not in build_report.build()


def test_the_panel_section_shows_every_model_forecast(results):
    write(results, "panel_latest.json", PANEL)
    page = build_report.build()
    assert "a:free" in page and "b:free" in page
    assert "Nott&#x27;m Forest" in page


def test_the_panel_section_says_so_when_nothing_has_been_scored_yet(results):
    write(results, "panel_latest.json", PANEL)
    assert "not been played" in build_report.build()


def test_the_panel_is_ranked_against_the_market_once_scores_exist(results):
    write(results, "panel_latest.json", PANEL)
    write(results, "llm_panel_scores.json", PANEL_SCORES)
    scored = build_report.build().split("Panel scored so far")[1]
    assert scored.index("market opening") < scored.index("a:free")


def test_the_page_shows_what_betting_the_model_anyway_would_have_cost(results):
    write(results, "betting.json", LOSING_STUDY)
    page = build_report.build()
    assert "Resamples in profit" in page
    assert "6.0%" in page
    assert "Bankroll, model only" in page
    assert "started at 1000" in page


def stamp(page):
    found = re.search(r"<!-- fingerprint ([0-9a-f]{12}) -->", page)
    return found.group(1) if found else None


def test_the_page_carries_a_fingerprint_of_its_contents(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    assert stamp(build_report.build()) is not None


def test_the_same_results_fingerprint_the_same(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    assert stamp(build_report.build()) == stamp(build_report.build())


def test_a_changed_number_changes_the_fingerprint(results):
    write(results, "matchweek_20260912.json", MATCHWEEK)
    before = stamp(build_report.build())
    moved = {**MATCHWEEK, "fixtures": [{**MATCHWEEK["fixtures"][0], "p_home": 0.6, "p_away": 0.1}]}
    write(results, "matchweek_20260912.json", moved)
    assert stamp(build_report.build()) != before


SIGNIFICANT = {
    **BACKTEST,
    "significance": {
        "market_closing_vs_dixon_coles": {
            "mean": 0.00761, "low": 0.00392, "high": 0.01137, "better_share": 1.0,
        },
        "dixon_coles_xg_vs_dixon_coles": {
            "mean": 0.00026, "low": -0.00347, "high": 0.00393, "better_share": 0.55,
        },
    },
}


def test_nothing_is_said_about_significance_before_it_is_measured(results):
    write(results, "backtest.json", BACKTEST)
    assert "crosses zero" not in build_report.build()


def test_a_gap_that_survives_resampling_is_reported_as_holding(results):
    write(results, "backtest.json", SIGNIFICANT)
    page = build_report.build()
    assert "market closing over dixon coles" in page
    assert "+0.00392" in page


def test_a_gap_that_does_not_survive_is_called_out_as_crossing_zero(results):
    write(results, "backtest.json", SIGNIFICANT)
    page = build_report.build()
    assert "crosses zero" in page
    assert "-0.00347" in page
