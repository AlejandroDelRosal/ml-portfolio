import numpy as np
import pandas as pd
import penaltyblog as pb

from src.models import PROBABILITY_COLUMNS

CLOSING_ODDS = ["AvgCH", "AvgCD", "AvgCA"]
OPENING_ODDS = ["AvgH", "AvgD", "AvgA"]
BEST_CLOSING_ODDS = ["MaxCH", "MaxCD", "MaxCA"]
CLOSING_GOALS_ODDS = ["AvgC>2.5", "AvgC<2.5"]
OPENING_GOALS_ODDS = ["Avg>2.5", "Avg<2.5"]
OUTCOME_CODES = {"H": 0, "D": 1, "A": 2}
IMPLIED_METHODS = ["multiplicative", "additive", "power", "shin", "odds_ratio", "logarithmic"]
# Best log loss over six seasons in analysis/run_baseline.py, by a hair.
DEFAULT_METHOD = "shin"


def implied_probabilities(matches: pd.DataFrame, columns=CLOSING_ODDS, method: str = DEFAULT_METHOD) -> np.ndarray:
    odds = matches[columns].to_numpy(dtype=float)
    probs = np.full(odds.shape, np.nan)
    complete = np.isfinite(odds).all(axis=1)
    for position in np.flatnonzero(complete):
        result = pb.implied.calculate_implied(list(odds[position]), method=method)
        probs[position] = np.asarray(result.probabilities, dtype=float)
    return probs


def bookmaker_margin(matches: pd.DataFrame, columns=CLOSING_ODDS) -> np.ndarray:
    odds = matches[columns].to_numpy(dtype=float)
    return (1.0 / odds).sum(axis=1) - 1.0


def outcomes(matches: pd.DataFrame) -> np.ndarray:
    return matches["FTR"].map(OUTCOME_CODES).to_numpy(dtype=float)


def usable(matches: pd.DataFrame, columns=CLOSING_ODDS) -> pd.Series:
    odds_present = matches[columns].notna().all(axis=1)
    return odds_present & matches["FTR"].isin(OUTCOME_CODES)


def market_frame(matches: pd.DataFrame, columns=CLOSING_ODDS, method: str = DEFAULT_METHOD):
    """Rows with both a result and complete odds, with their implied probabilities."""
    played = matches[usable(matches, columns)].copy()
    probs = implied_probabilities(played, columns=columns, method=method)
    return played, probs, outcomes(played).astype(int)


class MarketPredictor:
    """The bookmakers as a forecaster, so the baseline runs through the same harness."""

    def __init__(self, columns=CLOSING_ODDS, goals_columns=CLOSING_GOALS_ODDS, method: str = DEFAULT_METHOD):
        self.columns = columns
        self.goals_columns = goals_columns
        self.method = method

    def fit(self, matches: pd.DataFrame) -> "MarketPredictor":
        # The market needs no training; it is already priced.
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        outcome = implied_probabilities(fixtures, columns=self.columns, method=self.method)
        over = np.full(len(fixtures), np.nan)
        if all(column in fixtures.columns for column in self.goals_columns):
            goals = implied_probabilities(fixtures, columns=self.goals_columns, method=self.method)
            over = goals[:, 0]
        rows = np.column_stack([outcome, over])
        return pd.DataFrame(rows, columns=PROBABILITY_COLUMNS, index=fixtures.index)
