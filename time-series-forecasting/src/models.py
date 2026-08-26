import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(train, horizon: int):
    return np.full(horizon, train.iloc[-1])


def seasonal_naive_forecast(train, horizon: int, period: int = 12):
    return np.array([train.iloc[-period + (i % period)] for i in range(horizon)])


def holt_winters_forecast(train, horizon: int, seasonal_periods: int = 12):
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=seasonal_periods
    ).fit()
    return model.forecast(horizon).to_numpy()


def arima_forecast(train, horizon: int, order=(2, 1, 2)):
    model = ARIMA(train, order=order).fit()
    return model.forecast(horizon).to_numpy()


MODELS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "holt_winters": holt_winters_forecast,
    "arima": arima_forecast,
}
