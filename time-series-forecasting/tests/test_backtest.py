import pandas as pd
import pytest

from src.models import naive_forecast
from src.backtest import walk_forward_errors, mean_absolute_error, root_mean_squared_error


def test_walk_forward_errors_matches_hand_computation():
    series = pd.Series([1.0, 2.0, 4.0, 4.0, 8.0])
    errors = walk_forward_errors(series, naive_forecast, min_train_size=2)
    assert list(errors) == pytest.approx([2.0, 0.0, 4.0])


def test_mean_absolute_error_and_rmse():
    errors = [1.0, -2.0, 3.0]
    assert mean_absolute_error(errors) == pytest.approx(2.0)
    assert root_mean_squared_error(errors) == pytest.approx((14 / 3) ** 0.5)
