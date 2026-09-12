import numpy as np
import pytest

from src.metrics import accuracy, calibration_table, compare, evaluate, log_loss

PERFECT = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
UNIFORM = [[1 / 3, 1 / 3, 1 / 3]]


def test_perfect_forecast_scores_zero():
    scores = evaluate(PERFECT, [0, 2])
    assert scores["rps"] == pytest.approx(0.0)
    assert scores["log_loss"] == pytest.approx(0.0)
    assert scores["brier"] == pytest.approx(0.0)
    assert scores["accuracy"] == pytest.approx(1.0)


def test_uniform_forecast_matches_hand_computed_values():
    scores = evaluate(UNIFORM, [1])
    assert scores["rps"] == pytest.approx(1 / 9)
    assert scores["log_loss"] == pytest.approx(np.log(3))


def test_log_loss_is_finite_for_a_zero_probability_outcome():
    assert np.isfinite(log_loss([[1.0, 0.0, 0.0]], [2]))


def test_a_sharper_correct_forecast_scores_better():
    sharp = evaluate([[0.8, 0.1, 0.1]], [0])
    blunt = evaluate([[0.4, 0.3, 0.3]], [0])
    assert sharp["rps"] < blunt["rps"]
    assert sharp["log_loss"] < blunt["log_loss"]


def test_accuracy_picks_the_most_likely_outcome():
    assert accuracy([[0.2, 0.5, 0.3]], [1]) == pytest.approx(1.0)
    assert accuracy([[0.2, 0.5, 0.3]], [0]) == pytest.approx(0.0)


def test_calibration_table_accounts_for_every_forecast_point():
    probs = np.repeat(UNIFORM, 50, axis=0)
    table = calibration_table(probs, [1] * 50)
    assert table["n"].sum() == probs.size
    assert ((table["observed"] >= 0) & (table["observed"] <= 1)).all()


def test_a_calibrated_forecast_has_observed_close_to_predicted():
    rng = np.random.default_rng(0)
    home = rng.uniform(0.2, 0.7, 4000)
    probs = np.column_stack([home, (1 - home) / 2, (1 - home) / 2])
    draws = np.array([rng.choice(3, p=row) for row in probs])
    table = calibration_table(probs, draws, bins=5)
    populated = table[table["n"] > 100]
    assert np.allclose(populated["observed"], populated["predicted"], atol=0.05)


def test_compare_sorts_by_rps():
    table = compare({"good": evaluate([[0.9, 0.05, 0.05]], [0]), "bad": evaluate(UNIFORM, [0])})
    assert list(table.index) == ["good", "bad"]
