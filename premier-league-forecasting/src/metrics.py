import numpy as np
import pandas as pd
import penaltyblog as pb

EPSILON = 1e-15


def log_loss(probs, outcomes) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), EPSILON, 1.0)
    outcomes = np.asarray(outcomes, dtype=int)
    return float(-np.log(probs[np.arange(len(outcomes)), outcomes]).mean())


def accuracy(probs, outcomes) -> float:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    return float((probs.argmax(axis=1) == outcomes).mean())


def evaluate(probs, outcomes) -> dict:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    return {
        "n": len(outcomes),
        "rps": pb.metrics.rps_average(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "brier": pb.metrics.multiclass_brier_score(probs, outcomes),
        "ignorance": pb.metrics.ignorance_score(probs, outcomes),
        "accuracy": accuracy(probs, outcomes),
    }


def calibration_table(probs, outcomes, bins: int = 10) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    hits = np.zeros_like(probs)
    hits[np.arange(len(outcomes)), outcomes] = 1.0
    # Every forecast contributes one point per outcome, which is the standard
    # reliability diagram for a multiclass forecast.
    flat_probs = probs.ravel()
    flat_hits = hits.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat_probs, edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bin": index, "predicted": flat_probs, "observed": flat_hits})
    table = frame.groupby("bin").agg(n=("predicted", "size"), predicted=("predicted", "mean"), observed=("observed", "mean"))
    table.insert(0, "lower", edges[table.index])
    table.insert(1, "upper", edges[table.index + 1])
    return table.reset_index(drop=True)


def compare(results: dict[str, dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results).T
    frame["n"] = frame["n"].astype(int)
    return frame.sort_values("rps")
