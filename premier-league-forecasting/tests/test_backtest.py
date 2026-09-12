import numpy as np
import pandas as pd
import pytest

from src.backtest import LeakageError, priced, walk_forward
from src.models import PROBABILITY_COLUMNS
from tests.test_models import synthetic_matches


class RecordingPredictor:
    """Predicts a fixed distribution and remembers the training data it saw."""

    calls: list = []

    def fit(self, train):
        RecordingPredictor.calls.append(train["Datetime"].max())
        return self

    def predict(self, fixtures):
        rows = np.tile([0.45, 0.27, 0.28, 0.5], (len(fixtures), 1))
        return pd.DataFrame(rows, columns=PROBABILITY_COLUMNS, index=fixtures.index)


class LeakingPredictor(RecordingPredictor):
    pass


@pytest.fixture(autouse=True)
def reset_calls():
    RecordingPredictor.calls = []


def test_training_data_never_reaches_the_predicted_matchday():
    matches = synthetic_matches(seasons=3)
    target = ["2022-23"]
    predictions = walk_forward(matches, RecordingPredictor, seasons=target, min_train=50)
    first_target_kickoff = matches[matches["Season"].isin(target)]["Datetime"].min()
    assert all(seen < first_target_kickoff or seen < predictions["Datetime"].max() for seen in RecordingPredictor.calls)
    for seen, day in zip(RecordingPredictor.calls, sorted(predictions["Datetime"].dt.date.unique())):
        assert seen.date() < day


def test_every_target_match_with_enough_history_is_predicted():
    matches = synthetic_matches(seasons=3)
    predictions = walk_forward(matches, RecordingPredictor, seasons=["2022-23"], min_train=50)
    expected = (matches["Season"] == "2022-23").sum()
    assert len(predictions) == expected
    assert predictions[PROBABILITY_COLUMNS].notna().all(axis=None)


def test_predictions_come_back_in_chronological_order():
    matches = synthetic_matches(seasons=3)
    predictions = walk_forward(matches, RecordingPredictor, seasons=["2022-23"], min_train=50)
    assert predictions["Datetime"].is_monotonic_increasing


def test_matchdays_without_enough_history_are_skipped():
    matches = synthetic_matches(seasons=3)
    target = ["2021-22", "2022-23"]
    everything = walk_forward(matches, RecordingPredictor, seasons=target, min_train=10)
    demanding = walk_forward(matches, RecordingPredictor, seasons=target, min_train=len(matches) // 2)
    assert len(demanding) < len(everything)


def test_an_impossible_training_window_is_reported():
    matches = synthetic_matches(seasons=1)
    with pytest.raises(ValueError):
        walk_forward(matches, RecordingPredictor, seasons=["2020-21"], min_train=10_000)


def test_priced_drops_rows_the_model_could_not_price():
    matches = synthetic_matches(seasons=3)
    predictions = walk_forward(matches, RecordingPredictor, seasons=["2022-23"], min_train=50)
    predictions.loc[predictions.index[0], "p_home"] = np.nan
    assert len(priced(predictions)) == len(predictions) - 1


def test_leakage_error_is_available_for_guarding():
    assert issubclass(LeakageError, AssertionError)
