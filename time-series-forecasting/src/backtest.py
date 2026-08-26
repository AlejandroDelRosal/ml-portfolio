import numpy as np


def walk_forward_errors(series, model_fn, min_train_size: int, **kwargs):
    """One-step-ahead expanding-window backtest: refit at every origin,
    forecast the next point, record the signed error."""
    errors = []
    for origin in range(min_train_size, len(series)):
        train = series.iloc[:origin]
        actual = series.iloc[origin]
        forecast = model_fn(train, horizon=1, **kwargs)[0]
        errors.append(actual - forecast)
    return np.array(errors)


def mean_absolute_error(errors) -> float:
    return float(np.mean(np.abs(np.asarray(errors))))


def root_mean_squared_error(errors) -> float:
    errors = np.asarray(errors)
    return float(np.sqrt(np.mean(errors**2)))
