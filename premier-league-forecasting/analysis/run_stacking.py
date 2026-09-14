"""Ask whether anything beats the closing line once the closing line is a feature.

Two questions hide in that sentence, and they are not the same. A forecaster that
sees the closing price and beats it would still have to bet at that price. A
forecaster that sees only the opening price and beats the closing line has found
something the market had not finished pricing. The second is the one worth money.

The base forecasts were produced walk-forward by `run_backtest.py`, so they carry
no knowledge of their own match. The weights on top are fitted the same way:
on earlier matchdays only.
"""

import json
import pathlib

import numpy as np
import pandas as pd

from src.market import CLOSING_ODDS, OPENING_ODDS, MarketPredictor
from src.metrics import evaluate, paired_rps_interval
from src.models import OUTCOME_COLUMNS
from src.ratings import result_codes
from src.stacking import walk_forward_stack

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
MODELS = ["dixon_coles", "dixon_coles_xg", "elo", "pi_ratings"]
MIN_TRAIN = 380


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"predictions_{name}.csv", parse_dates=["Datetime"])


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[OUTCOME_COLUMNS].to_numpy(dtype=float)


def precision_recall(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    picked = probs.argmax(axis=1)
    report = {}
    for index, label in enumerate("HDA"):
        called = picked == index
        actual = outcomes == index
        hit = int((called & actual).sum())
        report[label] = {
            "called": int(called.sum()),
            "precision": float(hit / called.sum()) if called.sum() else float("nan"),
            "recall": float(hit / actual.sum()) if actual.sum() else float("nan"),
        }
    return report


def run() -> dict:
    base = load(MODELS[0])
    outcomes = result_codes(base)
    days = base["Datetime"].dt.date.to_numpy()

    market = {
        "cierre": MarketPredictor(columns=CLOSING_ODDS).predict(base)[OUTCOME_COLUMNS].to_numpy(dtype=float),
        "apertura": MarketPredictor(columns=OPENING_ODDS).predict(base)[OUTCOME_COLUMNS].to_numpy(dtype=float),
    }
    models = {name: probabilities(load(name)) for name in MODELS}

    recipes = {
        "mercado cierre, solo": ["cierre"],
        "mercado apertura, solo": ["apertura"],
        "apertura + modelos": ["apertura"] + MODELS,
        "cierre + modelos": ["cierre"] + MODELS,
        "solo modelos": MODELS,
    }
    pool = {**market, **models}

    stacked, weights = {}, {}
    priced = None
    for label, parts in recipes.items():
        frames = [pool[part] for part in parts]
        pooled, covered = walk_forward_stack(frames, outcomes, days, min_train=MIN_TRAIN)
        stacked[label] = pooled
        priced = covered if priced is None else (priced & covered)
        final, _ = __import__("src.stacking", fromlist=["fit_weights"]).fit_weights(frames, outcomes)
        weights[label] = dict(zip(parts, np.round(final, 4).tolist()))

    contenders = {"linea de cierre (referencia)": market["cierre"], **stacked}
    scores, reports = {}, {}
    for label, probs in contenders.items():
        scores[label] = evaluate(probs[priced], outcomes[priced])
        reports[label] = precision_recall(probs[priced], outcomes[priced])

    reference = market["cierre"][priced]
    intervals = {
        label: paired_rps_interval(probs[priced], reference, outcomes[priced], days[priced])
        for label, probs in stacked.items()
    }

    print(f"Evaluado sobre {int(priced.sum())} partidos de {len(outcomes)} "
          f"(los primeros {MIN_TRAIN} entrenan y no se puntuan)\n")
    table = pd.DataFrame(scores).T[["rps", "log_loss", "brier", "accuracy"]].sort_values("rps")
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))

    print("\nContra la linea de cierre, remuestreando jornadas completas:")
    for label, interval in sorted(intervals.items(), key=lambda kv: -kv[1]["mean"]):
        if interval["low"] > 0:
            verdict = "le gana al cierre"
        elif interval["high"] < 0:
            verdict = "PEOR que el cierre, y se demuestra"
        else:
            verdict = "indistinguible del cierre"
        print(f"  {label:26} {interval['mean']:+.5f}  95% [{interval['low']:+.5f}, {interval['high']:+.5f}]  {verdict}")

    print("\nPesos aprendidos sobre todo el periodo:")
    for label, found in weights.items():
        print(f"  {label:26} " + "  ".join(f"{k}={v:g}" for k, v in found.items()))

    print("\nPrecision y recall por clase:")
    for label in table.index:
        parts = "  ".join(
            f"{c}: prec {reports[label][c]['precision']:.3f} rec {reports[label][c]['recall']:.3f} "
            f"(dijo {reports[label][c]['called']})"
            for c in "HDA"
        )
        print(f"  {label:26} {parts}")

    payload = {"matches": int(priced.sum()), "scores": scores, "significance": intervals,
               "weights": weights, "precision_recall": reports}
    (RESULTS_DIR / "stacking.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
