import numpy as np
import pandas as pd
import pytest

from src.models import naive_forecast, seasonal_naive_forecast, holt_winters_forecast, arima_forecast


def test_naive_repeats_last_value():
    train = pd.Series([1.0, 2.0, 3.0])
    forecast = naive_forecast(train, horizon=3)
    assert list(forecast) == [3.0, 3.0, 3.0]


def test_seasonal_naive_repeats_same_month_last_cycle():
    train = pd.Series(range(24), dtype=float)
    forecast = seasonal_naive_forecast(train, horizon=3, period=12)
    assert list(forecast) == [12.0, 13.0, 14.0]


def test_holt_winters_produces_finite_forecast():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=48, freq="MS")
    month = np.arange(48) % 12
    values = pd.Series(10 + 0.1 * np.arange(48) + 2 * np.sin(2 * np.pi * month / 12) + rng.normal(scale=0.1, size=48), index=dates)
    values.index.freq = "MS"
    forecast = holt_winters_forecast(values, horizon=3)
    assert all(pd.notna(forecast))


def test_arima_produces_finite_forecast():
    dates = pd.date_range("2000-01-01", periods=48, freq="MS")
    values = pd.Series(range(48), index=dates, dtype=float)
    forecast = arima_forecast(values, horizon=3, order=(1, 1, 0))
    assert all(pd.notna(forecast))
