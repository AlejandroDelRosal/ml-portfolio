import numpy as np
import pandas as pd
import pytest

from src.betting import (
    blend,
    bootstrap_roi,
    has_demonstrable_edge,
    kelly_fraction,
    selection_frame,
    simulate,
    summarise,
    value_bets,
)

MODEL = np.array([[0.5, 0.3, 0.2]])
MARKET = np.array([[0.4, 0.3, 0.3]])


def predictions_frame(probabilities, odds, results):
    rows = []
    for index, (probs, price, (home_goals, away_goals)) in enumerate(zip(probabilities, odds, results)):
        rows.append({
            "Datetime": pd.Timestamp("2025-01-04") + pd.Timedelta(days=index),
            "Season": "2024-25",
            "HomeTeam": f"H{index}",
            "AwayTeam": f"A{index}",
            "FTHG": home_goals,
            "FTAG": away_goals,
            "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
            "AvgCH": price[0], "AvgCD": price[1], "AvgCA": price[2],
        })
    return pd.DataFrame(rows)


def test_blending_with_weight_one_returns_the_model():
    assert np.allclose(blend(MODEL, MARKET, 1.0), MODEL)


def test_blending_with_weight_zero_returns_the_market():
    assert np.allclose(blend(MODEL, MARKET, 0.0), MARKET)


def test_a_blend_stays_a_distribution_and_sits_between_its_parents():
    pooled = blend(MODEL, MARKET, 0.5)
    assert pooled.sum() == pytest.approx(1.0)
    assert MARKET[0, 0] < pooled[0, 0] < MODEL[0, 0]


def test_kelly_is_zero_at_a_fair_price_and_positive_with_an_edge():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert kelly_fraction(0.6, 2.0) > 0
    assert kelly_fraction(0.4, 2.0) < 0


def test_selection_frame_marks_the_winning_selection():
    frame = selection_frame(predictions_frame([[0.5, 0.3, 0.2]], [[2.0, 3.5, 4.0]], [(2, 0)]))
    assert len(frame) == 3
    assert frame.loc[frame["selection"] == "home", "won"].item()
    assert not frame.loc[frame["selection"] == "away", "won"].item()


def test_edges_follow_price_times_probability():
    frame = selection_frame(predictions_frame([[0.5, 0.3, 0.2]], [[2.5, 3.5, 4.0]], [(1, 1)]))
    assert frame.loc[frame["selection"] == "home", "edge"].item() == pytest.approx(0.25)


def test_value_bets_keep_only_selections_above_the_threshold():
    frame = selection_frame(predictions_frame([[0.5, 0.3, 0.2]], [[2.5, 3.0, 4.0]], [(1, 1)]))
    assert len(value_bets(frame, min_edge=0.2)) == 1
    assert len(value_bets(frame, min_edge=1.0)) == 0


def test_a_winning_bet_grows_the_bankroll_and_a_loser_shrinks_it():
    winner = selection_frame(predictions_frame([[0.9, 0.05, 0.05]], [[2.0, 20.0, 20.0]], [(3, 0)]))
    settled = simulate(value_bets(winner, min_edge=0.1))
    assert settled["profit"].sum() > 0
    assert settled["bankroll"].iloc[-1] > 1000.0

    loser = selection_frame(predictions_frame([[0.9, 0.05, 0.05]], [[2.0, 20.0, 20.0]], [(0, 3)]))
    settled = simulate(value_bets(loser, min_edge=0.1))
    assert settled["profit"].sum() < 0


def test_stakes_respect_the_cap():
    frame = selection_frame(predictions_frame([[0.99, 0.005, 0.005]], [[5.0, 20.0, 20.0]], [(1, 0)]))
    settled = simulate(value_bets(frame, min_edge=0.1), cap=0.02)
    assert settled["stake"].max() <= 1000.0 * 0.02 + 1e-9


def test_summary_reports_roi_over_the_amount_staked():
    frame = selection_frame(predictions_frame([[0.9, 0.05, 0.05]], [[2.0, 20.0, 20.0]], [(2, 0)]))
    settled = simulate(value_bets(frame, min_edge=0.1))
    summary = summarise(settled)
    assert summary["bets"] == 1
    assert summary["roi"] == pytest.approx(settled["profit"].sum() / settled["stake"].sum())


def test_an_empty_book_is_summarised_without_crashing():
    empty = simulate(pd.DataFrame(columns=["Datetime", "odds", "kelly", "won"]))
    assert summarise(empty)["bets"] == 0
    assert np.isnan(bootstrap_roi(empty)["low"])


def test_a_coin_flip_book_does_not_show_a_demonstrable_edge():
    rng = np.random.default_rng(7)
    odds, results = [], []
    for _ in range(300):
        odds.append([2.0, 4.0, 4.0])
        results.append((1, 0) if rng.random() < 0.5 else (0, 1))
    frame = selection_frame(predictions_frame([[0.52, 0.24, 0.24]] * 300, odds, results))
    settled = simulate(value_bets(frame, min_edge=0.0))
    interval = bootstrap_roi(settled, draws=2000)
    assert interval["low"] < 0 < interval["high"]
    assert not has_demonstrable_edge(interval)
