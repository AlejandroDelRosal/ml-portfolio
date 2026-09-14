"""Combine several forecasters by logarithmic pooling, with the weights learned.

The blend study in `betting.py` pools two forecasters at a weight chosen from a
grid. This generalises that to any number of them and fits the weights directly,
which is the honest way to ask how much each source is worth once the others are
present: a forecaster that only repeats what the market already says earns
nothing, however good it looks alone.

Weights are held at or above zero. A negative weight would mean betting against a
forecaster, which is a claim this data is far too small to support.
"""

import numpy as np
from scipy.optimize import minimize

EPSILON = 1e-12
DEFAULT_MIN_TRAIN = 380
# Keeps a weight from running away on a forecaster that happens to be right in a
# short training window; small enough not to move a weight the data supports.
RIDGE = 1e-3


def log_pool(frames: list, weights: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    """Pool probabilities log-linearly: p ∝ prod_k p_k ** w_k, tilted by class."""
    logs = np.stack([np.log(np.clip(np.asarray(frame, dtype=float), EPSILON, 1.0)) for frame in frames])
    scores = np.tensordot(np.asarray(weights, dtype=float), logs, axes=(0, 0)) + np.asarray(intercepts, dtype=float)
    scores -= scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(scores)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _negative_log_likelihood(parameters, logs, outcomes, count):
    weights, intercepts = parameters[:count], np.concatenate([[0.0], parameters[count:]])
    scores = np.tensordot(weights, logs, axes=(0, 0)) + intercepts
    scores -= scores.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(scores).sum(axis=1))
    chosen = scores[np.arange(len(outcomes)), outcomes]
    return float(-(chosen - log_norm).mean() + RIDGE * (weights**2).sum())


def fit_weights(frames: list, outcomes: np.ndarray) -> tuple:
    """The weights and class tilts that score the training matches best."""
    logs = np.stack([np.log(np.clip(np.asarray(frame, dtype=float), EPSILON, 1.0)) for frame in frames])
    outcomes = np.asarray(outcomes, dtype=int)
    count = len(frames)
    start = np.concatenate([np.full(count, 1.0 / count), np.zeros(2)])
    bounds = [(0.0, None)] * count + [(None, None)] * 2
    found = minimize(_negative_log_likelihood, start, args=(logs, outcomes, count), method="L-BFGS-B", bounds=bounds)
    weights = found.x[:count]
    return weights, np.concatenate([[0.0], found.x[count:]])


def walk_forward_stack(frames: list, outcomes: np.ndarray, days, min_train: int = DEFAULT_MIN_TRAIN, fit=fit_weights):
    """Fit the weights on earlier matches only, then price the coming matchday."""
    outcomes = np.asarray(outcomes, dtype=int)
    days = np.asarray(days)
    order = np.arange(len(outcomes))
    pooled = np.full((len(outcomes), 3), np.nan)
    priced = np.zeros(len(outcomes), dtype=bool)
    for day in sorted(set(days.tolist())):
        ahead = days == day
        history = order[days < day] if days.dtype.kind not in "OUS" else order[np.isin(days, sorted(d for d in set(days.tolist()) if d < day))]
        if len(history) < min_train:
            continue
        weights, intercepts = fit([frame[history] for frame in frames], outcomes[history])
        pooled[ahead] = log_pool([frame[ahead] for frame in frames], weights, intercepts)
        priced[ahead] = True
    return pooled, priced
