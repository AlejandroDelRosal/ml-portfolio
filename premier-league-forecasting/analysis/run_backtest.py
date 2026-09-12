"""Walk-forward comparison of every forecaster against the closing line.

The time decay is tuned on the validation seasons and then left alone, so the
test seasons are never used to make a choice.
"""

import json
import pathlib

import pandas as pd

from src.backtest import priced, walk_forward
from src.loader import load_matches
from src.market import MarketPredictor
from src.metrics import compare, evaluate
from src.models import OUTCOME_COLUMNS, GoalModel
from src.ratings import EloPredictor, PiRatingsPredictor, result_codes

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VALIDATION_SEASONS = ["2020-21", "2021-22", "2022-23"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
XI_GRID = [0.0, 0.0005, 0.001, 0.0018, 0.003, 0.005]


def score(predictions: pd.DataFrame) -> dict:
    usable = priced(predictions)
    return evaluate(usable[OUTCOME_COLUMNS].to_numpy(), result_codes(usable))


def tune_decay(matches: pd.DataFrame) -> float:
    print("Tuning the time decay on", ", ".join(VALIDATION_SEASONS))
    scores = {}
    for xi in XI_GRID:
        predictions = walk_forward(matches, lambda xi=xi: GoalModel(xi=xi), seasons=VALIDATION_SEASONS)
        scores[xi] = score(predictions)
        print(f"  xi={xi:<7} rps={scores[xi]['rps']:.5f} log_loss={scores[xi]['log_loss']:.5f}")
    best = min(scores, key=lambda xi: scores[xi]["log_loss"])
    print(f"  chosen xi={best}")
    return best


def run() -> dict:
    matches = load_matches()
    xi = tune_decay(matches)

    competitors = {
        "market_closing": MarketPredictor,
        "elo": EloPredictor,
        "pi_ratings": PiRatingsPredictor,
        "poisson": lambda: GoalModel("poisson", xi=xi),
        "dixon_coles": lambda: GoalModel("dixon_coles", xi=xi),
        "bivariate_poisson": lambda: GoalModel("bivariate_poisson", xi=xi),
    }

    print("\nBacktesting on", ", ".join(TEST_SEASONS))
    predictions = {}
    for name, factory in competitors.items():
        predictions[name] = walk_forward(matches, factory, seasons=TEST_SEASONS)
        print(f"  {name}: {len(priced(predictions[name]))} matches priced")

    # Every forecaster is judged on exactly the same matches.
    common = None
    for frame in predictions.values():
        index = priced(frame).index
        common = index if common is None else common.intersection(index)

    results = {name: score(frame.loc[common]) for name, frame in predictions.items()}
    table = compare(results)
    print(f"\n=== Walk-forward results on {len(common)} common matches ===")
    print(table.to_string(float_format=lambda value: f"{value:.5f}"))

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "xi": xi,
        "validation_seasons": VALIDATION_SEASONS,
        "test_seasons": TEST_SEASONS,
        "matches": int(len(common)),
        "results": results,
    }
    (RESULTS_DIR / "backtest.json").write_text(json.dumps(payload, indent=2))
    for name, frame in predictions.items():
        frame.loc[common].to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)
    return payload


if __name__ == "__main__":
    run()
