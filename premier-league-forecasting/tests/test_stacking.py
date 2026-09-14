import numpy as np
import pytest

from src import stacking

SURE_HOME = np.array([[0.90, 0.05, 0.05], [0.80, 0.10, 0.10], [0.70, 0.20, 0.10], [0.85, 0.10, 0.05]])
SURE_AWAY = np.array([[0.05, 0.05, 0.90], [0.10, 0.10, 0.80], [0.10, 0.20, 0.70], [0.05, 0.10, 0.85]])
EVEN = np.tile([1 / 3, 1 / 3, 1 / 3], (4, 1))
HOME_WON = np.array([0, 0, 0, 0])


def test_pooling_one_forecaster_at_weight_one_returns_it_unchanged():
    pooled = stacking.log_pool([SURE_HOME], np.array([1.0]), np.zeros(3))
    assert pooled == pytest.approx(SURE_HOME, abs=1e-9)


def test_every_pooled_row_is_a_probability_distribution():
    pooled = stacking.log_pool([SURE_HOME, SURE_AWAY], np.array([0.6, 0.4]), np.array([0.1, 0.0, -0.1]))
    assert pooled.sum(axis=1) == pytest.approx(np.ones(len(pooled)))
    assert (pooled > 0).all()


def test_pooling_two_identical_forecasters_sharpens_nothing_at_half_weight_each():
    pooled = stacking.log_pool([SURE_HOME, SURE_HOME], np.array([0.5, 0.5]), np.zeros(3))
    assert pooled == pytest.approx(SURE_HOME, abs=1e-9)


def test_a_useless_forecaster_is_given_no_weight():
    # One forecaster names the result every time, the other is uninformative.
    weights, _ = stacking.fit_weights([SURE_HOME, EVEN], HOME_WON)
    assert weights[0] > weights[1]
    assert weights[1] == pytest.approx(0.0, abs=1e-6)


def test_a_forecaster_pointing_the_wrong_way_is_given_no_weight():
    weights, _ = stacking.fit_weights([SURE_HOME, SURE_AWAY], HOME_WON)
    assert weights[1] == pytest.approx(0.0, abs=1e-6)


def test_weights_are_never_negative():
    weights, _ = stacking.fit_weights([SURE_AWAY, EVEN], HOME_WON)
    assert (weights >= 0).all()


def test_the_stack_never_trains_on_the_day_it_predicts():
    days = np.array(["a", "a", "b", "b"])
    seen = []

    def spy(frames, outcomes):
        seen.append(len(outcomes))
        return np.array([1.0] * len(frames)), np.zeros(3)

    stacking.walk_forward_stack([SURE_HOME, EVEN], HOME_WON, days, min_train=2, fit=spy)
    assert seen == [2]


def test_days_without_enough_history_are_left_unpriced():
    days = np.array(["a", "a", "b", "b"])
    pooled, priced = stacking.walk_forward_stack([SURE_HOME, EVEN], HOME_WON, days, min_train=2)
    assert priced.tolist() == [False, False, True, True]
    assert np.isnan(pooled[0]).all()
