"""Does the model find value the market missed?

Each book only ever sees information available when the bet would be struck: the
opening book blends the model with the opening price and stakes into the opening
price, and the closing book does the same at the close. Mixing them, using the
closing line to bet into the opening price, would be forecasting with the answer
in hand.

The blend weight and the edge threshold are chosen on the validation seasons and
applied unchanged to the test seasons.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from src.backtest import priced, walk_forward
from src.betting import blend, bootstrap_roi, has_demonstrable_edge, selection_frame, simulate, summarise, value_bets
from src.loader import load_matches
from src.market import CLOSING_GOALS_ODDS, CLOSING_ODDS, OPENING_GOALS_ODDS, OPENING_ODDS, MarketPredictor
from src.metrics import evaluate
from src.models import DEFAULT_XI, OUTCOME_COLUMNS, GoalModel
from src.ratings import result_codes

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
VALIDATION_SEASONS = ["2020-21", "2021-22", "2022-23"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
WEIGHT_GRID = [round(value, 2) for value in np.arange(0.0, 1.01, 0.1)]
EDGE_GRID = [0.0, 0.02, 0.05, 0.08, 0.12]
MIN_BETS = 50
BOOKS = {
    "opening": (OPENING_ODDS, OPENING_GOALS_ODDS),
    "closing": (CLOSING_ODDS, CLOSING_GOALS_ODDS),
}


def chosen_xi() -> float:
    path = RESULTS_DIR / "backtest.json"
    return json.loads(path.read_text())["xi"] if path.exists() else DEFAULT_XI


def forecasts(matches: pd.DataFrame, seasons: list[str], xi: float):
    """The model and both market snapshots over exactly the same matches."""
    frames = {"model": priced(walk_forward(matches, lambda: GoalModel(xi=xi), seasons=seasons))}
    for label, (odds, goals) in BOOKS.items():
        frames[label] = priced(walk_forward(matches, lambda odds=odds, goals=goals: MarketPredictor(columns=odds, goals_columns=goals), seasons=seasons))
    common = None
    for frame in frames.values():
        common = frame.index if common is None else common.intersection(frame.index)
    return {label: frame.loc[common] for label, frame in frames.items()}


def blended(model: pd.DataFrame, market: pd.DataFrame, weight: float) -> pd.DataFrame:
    pooled = blend(model[OUTCOME_COLUMNS].to_numpy(), market[OUTCOME_COLUMNS].to_numpy(), weight)
    frame = model.copy()
    frame[OUTCOME_COLUMNS] = pooled
    return frame


def tune_weight(model: pd.DataFrame, market: pd.DataFrame, label: str) -> float:
    codes = result_codes(model)
    scores = {}
    print(f"\nBlend weight against the {label} line (1.0 is the model alone, 0.0 the market alone)")
    for weight in WEIGHT_GRID:
        pooled = blend(model[OUTCOME_COLUMNS].to_numpy(), market[OUTCOME_COLUMNS].to_numpy(), weight)
        scores[weight] = evaluate(pooled, codes)
        print(f"  weight={weight:<5} rps={scores[weight]['rps']:.5f} log_loss={scores[weight]['log_loss']:.5f}")
    best = min(scores, key=lambda weight: scores[weight]["log_loss"])
    print(f"  chosen weight={best}")
    return best


def tune_edge(frame: pd.DataFrame, odds_columns, label: str) -> float:
    selections = selection_frame(frame, odds_columns=odds_columns)
    best, best_roi = EDGE_GRID[0], -np.inf
    print(f"\nEdge threshold against the {label} line")
    for edge in EDGE_GRID:
        summary = summarise(simulate(value_bets(selections, min_edge=edge)))
        roi = summary["roi"] if summary["bets"] else float("nan")
        print(f"  edge>={edge:<5} bets={summary['bets']:<5} roi={roi:.4f}")
        if summary["bets"] >= MIN_BETS and roi > best_roi:
            best, best_roi = edge, roi
    print(f"  chosen edge>={best}")
    return best


def evaluate_book(frame: pd.DataFrame, odds_columns, edge: float, label: str) -> dict:
    selections = selection_frame(frame, odds_columns=odds_columns)
    settled = simulate(value_bets(selections, min_edge=edge))
    summary = summarise(settled)
    interval = bootstrap_roi(settled)
    summary.update({
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "positive_share": interval["positive_share"],
        "demonstrable_edge": has_demonstrable_edge(interval),
        "min_edge": edge,
    })
    print(f"\n=== Betting into the {label} line, edge>={edge} ===")
    if not summary["bets"]:
        print("  no selection cleared the threshold, which is what a market with no exploitable gap looks like")
        return summary
    print(f"  bets {summary['bets']} | staked {summary['staked']:.0f} | profit {summary['profit']:.0f}")
    print(f"  ROI {summary['roi']:.4f} | 95% interval [{interval['low']:.4f}, {interval['high']:.4f}] | max drawdown {summary['max_drawdown']:.3f}")
    print("  VERDICT:", "edge demonstrable" if summary["demonstrable_edge"] else "no demonstrable edge, do not stake this")
    return summary


def run() -> dict:
    matches = load_matches()
    xi = chosen_xi()
    print(f"Using time decay xi={xi}")

    print("\nValidation:", ", ".join(VALIDATION_SEASONS))
    validation = forecasts(matches, VALIDATION_SEASONS, xi)
    weights, edges = {}, {}
    for label, (odds, _) in BOOKS.items():
        weights[label] = tune_weight(validation["model"], validation[label], label)
        edges[label] = tune_edge(blended(validation["model"], validation[label], weights[label]), odds, label)

    print("\nTest:", ", ".join(TEST_SEASONS))
    test = forecasts(matches, TEST_SEASONS, xi)
    codes = result_codes(test["model"])
    accuracy = {
        "market_closing": evaluate(test["closing"][OUTCOME_COLUMNS].to_numpy(), codes),
        "market_opening": evaluate(test["opening"][OUTCOME_COLUMNS].to_numpy(), codes),
        "model": evaluate(test["model"][OUTCOME_COLUMNS].to_numpy(), codes),
    }
    for label in BOOKS:
        pooled = blended(test["model"], test[label], weights[label])
        accuracy[f"blend_{label}"] = evaluate(pooled[OUTCOME_COLUMNS].to_numpy(), codes)

    print(f"\n=== Forecast quality on {len(codes)} test matches ===")
    for name, scores in sorted(accuracy.items(), key=lambda item: item[1]["rps"]):
        print(f"  {name:16} rps={scores['rps']:.5f} log_loss={scores['log_loss']:.5f}")

    books = {}
    for label, (odds, _) in BOOKS.items():
        pooled = blended(test["model"], test[label], weights[label])
        books[label] = evaluate_book(pooled, odds, edges[label], label)

    # Fixed in advance, not selected after the fact: what betting the raw model
    # into the opening price would have done, whatever the tuning preferred.
    model_edge = tune_edge(blended(validation["model"], validation["opening"], 1.0), OPENING_ODDS, "opening, model alone")
    books["model_only"] = evaluate_book(blended(test["model"], test["opening"], 1.0), OPENING_ODDS, model_edge, "opening, model alone")
    edges["model_only"] = model_edge

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "xi": xi,
        "weights": weights,
        "weight": weights["opening"],
        "edges": edges,
        "accuracy": accuracy,
        "books": books,
        "matches": int(len(codes)),
    }
    (RESULTS_DIR / "betting.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
