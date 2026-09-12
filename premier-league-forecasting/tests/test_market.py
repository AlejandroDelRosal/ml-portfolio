import pathlib

import numpy as np
import pytest

from src.loader import load_matches
from src.market import (
    CLOSING_ODDS,
    IMPLIED_METHODS,
    OPENING_ODDS,
    bookmaker_margin,
    implied_probabilities,
    market_frame,
    outcomes,
    usable,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def matches():
    return load_matches(raw_dir=FIXTURES)


def test_implied_probabilities_sum_to_one(matches):
    probs = implied_probabilities(matches)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", IMPLIED_METHODS)
def test_every_method_returns_a_valid_distribution(matches, method):
    probs = implied_probabilities(matches, method=method)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_removing_the_margin_lowers_the_raw_favourite_probability(matches):
    probs = implied_probabilities(matches, columns=CLOSING_ODDS)
    raw = 1.0 / matches[CLOSING_ODDS].to_numpy(dtype=float)
    assert (probs <= raw + 1e-9).all()


def test_margin_is_positive_and_small(matches):
    margin = bookmaker_margin(matches)
    assert (margin > 0).all()
    assert (margin < 0.20).all()


def test_missing_odds_produce_missing_probabilities(matches):
    incomplete = matches.copy()
    incomplete.loc[incomplete.index[0], "AvgCH"] = np.nan
    probs = implied_probabilities(incomplete)
    assert np.isnan(probs[0]).all()
    assert np.isfinite(probs[1:]).all()


def test_outcomes_are_encoded_as_home_draw_away(matches):
    codes = outcomes(matches)
    assert set(np.unique(codes)) <= {0.0, 1.0, 2.0}
    home_wins = matches["FTHG"] > matches["FTAG"]
    assert (codes[home_wins.to_numpy()] == 0).all()


def test_market_frame_drops_rows_without_a_result(matches):
    unplayed = matches.copy()
    unplayed.loc[unplayed.index[0], "FTR"] = np.nan
    played, probs, codes = market_frame(unplayed)
    assert len(played) == len(matches) - 1
    assert len(probs) == len(codes) == len(played)


def test_opening_and_closing_odds_are_both_usable(matches):
    assert usable(matches, CLOSING_ODDS).all()
    assert usable(matches, OPENING_ODDS).all()


def test_the_market_predictor_needs_no_training(matches):
    from src.market import MarketPredictor

    predictor = MarketPredictor()
    assert predictor.fit(matches.head(0)) is predictor


def test_the_market_predictor_prices_results_and_goals(matches):
    from src.market import MarketPredictor
    from src.models import OUTCOME_COLUMNS, PROBABILITY_COLUMNS

    predictions = MarketPredictor().predict(matches)
    assert list(predictions.columns) == PROBABILITY_COLUMNS
    assert np.allclose(predictions[OUTCOME_COLUMNS].sum(axis=1), 1.0)
    assert predictions["p_over25"].between(0, 1).all()


def test_the_opening_and_closing_snapshots_disagree(matches):
    from src.market import MarketPredictor
    from src.models import OUTCOME_COLUMNS

    closing = MarketPredictor(columns=CLOSING_ODDS).predict(matches)
    opening = MarketPredictor(columns=OPENING_ODDS, goals_columns=["Avg>2.5", "Avg<2.5"]).predict(matches)
    assert not np.allclose(closing[OUTCOME_COLUMNS], opening[OUTCOME_COLUMNS])


def test_goals_stay_unpriced_when_the_totals_market_is_absent(matches):
    from src.market import MarketPredictor

    without_totals = matches.drop(columns=["AvgC>2.5", "AvgC<2.5"])
    assert MarketPredictor().predict(without_totals)["p_over25"].isna().all()
