import pathlib

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_unemployment_rate
from src.models import naive_forecast, seasonal_naive_forecast, holt_winters_forecast, arima_forecast
from src.backtest import walk_forward_errors, mean_absolute_error, root_mean_squared_error
from src.diebold_mariano import diebold_mariano_test

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
TEST_WINDOW = 120

MODELS = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "arima": arima_forecast,
    "holt_winters": holt_winters_forecast,
}


def plot_series(series):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series.index, series, color="#2b6cb0")
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_title("US unemployment rate, 1948-2026 (FRED, series UNRATE)")
    fig.savefig(RESULTS_DIR / "unemployment_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_errors(test_dates, errors_by_model):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, errors in errors_by_model.items():
        ax.plot(test_dates, errors, label=name, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Forecast error (percentage points)")
    ax.set_title("One-step-ahead forecast errors, walk-forward backtest")
    ax.legend()
    fig.savefig(RESULTS_DIR / "forecast_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(mae_by_model, rmse_by_model):
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(mae_by_model.keys())
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [mae_by_model[n] for n in names], width, label="MAE", color="#2b6cb0")
    ax.bar(x + width / 2, [rmse_by_model[n] for n in names], width, label="RMSE", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Percentage points")
    ax.set_title(f"Out-of-sample forecast accuracy, last {TEST_WINDOW} months")
    ax.legend()
    fig.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    series = load_unemployment_rate()
    plot_series(series)

    min_train_size = len(series) - TEST_WINDOW
    test_dates = series.index[min_train_size:]

    errors_by_model = {}
    for name, model_fn in MODELS.items():
        errors_by_model[name] = walk_forward_errors(series, model_fn, min_train_size)
        print(f"{name}: MAE = {mean_absolute_error(errors_by_model[name]):.4f}, "
              f"RMSE = {root_mean_squared_error(errors_by_model[name]):.4f}")

    plot_forecast_errors(test_dates, errors_by_model)
    mae_by_model = {n: mean_absolute_error(e) for n, e in errors_by_model.items()}
    rmse_by_model = {n: root_mean_squared_error(e) for n, e in errors_by_model.items()}
    plot_accuracy_comparison(mae_by_model, rmse_by_model)

    print("\nDiebold-Mariano test vs naive (H0: equal forecast accuracy):")
    for name in ["seasonal_naive", "arima", "holt_winters"]:
        stat, p_value = diebold_mariano_test(errors_by_model["naive"], errors_by_model[name])
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"  naive vs {name}: DM = {stat:.3f}, p = {p_value:.4f} ({significance} at 5%)")


if __name__ == "__main__":
    main()
