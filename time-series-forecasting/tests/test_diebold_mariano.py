import numpy as np
import pytest

from src.diebold_mariano import diebold_mariano_test


def test_equally_accurate_models_are_not_significantly_different():
    rng = np.random.default_rng(0)
    errors_a = rng.normal(scale=1.0, size=400)
    errors_b = rng.normal(scale=1.0, size=400)
    _stat, p_value = diebold_mariano_test(errors_a, errors_b)
    assert p_value > 0.05


def test_clearly_better_model_is_detected_as_significant():
    rng = np.random.default_rng(1)
    good_errors = rng.normal(scale=0.1, size=300)
    bad_errors = rng.normal(scale=5.0, size=300)
    stat, p_value = diebold_mariano_test(good_errors, bad_errors)
    assert p_value < 0.01
    assert stat < 0
