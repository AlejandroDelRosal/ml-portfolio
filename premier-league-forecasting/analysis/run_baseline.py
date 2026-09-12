"""Measure the bar: how well bookmaker odds forecast the Premier League.

Every later model is judged against these numbers.
"""

import json
import pathlib

import pandas as pd

from src.loader import load_matches
from src.market import CLOSING_ODDS, IMPLIED_METHODS, OPENING_ODDS, bookmaker_margin, market_frame
from src.metrics import calibration_table, compare, evaluate

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
TEST_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def run(seasons=TEST_SEASONS) -> dict:
    matches = load_matches(seasons=seasons)
    results = {}
    for label, columns in [("closing", CLOSING_ODDS), ("opening", OPENING_ODDS)]:
        for method in IMPLIED_METHODS:
            played, probs, codes = market_frame(matches, columns=columns, method=method)
            results[f"{label}_{method}"] = evaluate(probs, codes)

    table = compare(results)
    played, probs, codes = market_frame(matches, columns=CLOSING_ODDS)
    calibration = calibration_table(probs, codes)
    margins = pd.Series(bookmaker_margin(played), name="margin")

    print("Seasons:", ", ".join(seasons))
    print("\n=== Market baseline, sorted by RPS ===")
    print(table.to_string(float_format=lambda value: f"{value:.5f}"))
    print("\n=== Calibration of the closing line ===")
    print(calibration.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nBookmaker margin on the closing line: mean {margins.mean():.4f}, median {margins.median():.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "seasons": seasons,
        "matches": int(len(played)),
        "methods": {name: scores for name, scores in results.items()},
        "closing_margin_mean": float(margins.mean()),
        "calibration": calibration.to_dict(orient="records"),
    }
    (RESULTS_DIR / "baseline.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
