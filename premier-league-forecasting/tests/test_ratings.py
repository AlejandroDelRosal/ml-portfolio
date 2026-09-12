import numpy as np
import pandas as pd
import pytest

from src.models import OUTCOME_COLUMNS
from src.ratings import AWAY_WIN, DRAW, HOME_WIN, EloPredictor, PiRatingsPredictor, result_codes
from tests.test_models import synthetic_matches


def test_result_codes_follow_the_market_encoding():
    matches = pd.DataFrame({"FTHG": [2, 1, 0], "FTAG": [0, 1, 2]})
    assert list(result_codes(matches)) == [HOME_WIN, DRAW, AWAY_WIN]


@pytest.mark.parametrize("predictor", [EloPredictor, PiRatingsPredictor])
def test_predictions_are_valid_distributions(predictor):
    matches = synthetic_matches()
    model = predictor().fit(matches)
    predictions = model.predict(matches.tail(10))
    assert np.allclose(predictions[OUTCOME_COLUMNS].sum(axis=1), 1.0, atol=1e-6)
    assert (predictions[OUTCOME_COLUMNS] > 0).all(axis=None)


@pytest.mark.parametrize("predictor", [EloPredictor, PiRatingsPredictor])
def test_goal_totals_are_left_unpriced(predictor):
    matches = synthetic_matches()
    model = predictor().fit(matches)
    assert model.predict(matches.tail(3))["p_over25"].isna().all()


@pytest.mark.parametrize("predictor", [EloPredictor, PiRatingsPredictor])
def test_predicting_before_fitting_is_rejected(predictor):
    with pytest.raises(RuntimeError):
        predictor().predict(synthetic_matches().head(1))


def test_elo_rewards_the_winner_and_punishes_the_loser():
    matches = pd.DataFrame({
        "HomeTeam": ["A"] * 5,
        "AwayTeam": ["B"] * 5,
        "FTHG": [3] * 5,
        "FTAG": [0] * 5,
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="7D"),
    })
    model = EloPredictor().fit(matches)
    assert model.rating("A") > 1500 > model.rating("B")


def test_elo_treats_an_unseen_team_as_average():
    model = EloPredictor().fit(synthetic_matches())
    assert model.rating("Newcomer") == pytest.approx(1500.0)


def test_pi_ratings_learn_from_the_goal_margin():
    heavy = pd.DataFrame({
        "HomeTeam": ["A"] * 5, "AwayTeam": ["B"] * 5,
        "FTHG": [4] * 5, "FTAG": [0] * 5,
        "Datetime": pd.date_range("2024-01-01", periods=5, freq="7D"),
    })
    narrow = heavy.assign(FTHG=1, FTAG=0)
    assert PiRatingsPredictor().fit(heavy).expected_margin("A", "B") > PiRatingsPredictor().fit(narrow).expected_margin("A", "B")
