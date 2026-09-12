import numpy as np
import pandas as pd

from src.market import CLOSING_ODDS
from src.models import OUTCOME_COLUMNS

SELECTIONS = ["home", "draw", "away"]
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MIN_EDGE = 0.05
DEFAULT_STAKE_CAP = 0.02
STARTING_BANKROLL = 1000.0


def blend(model_probs: np.ndarray, market_probs: np.ndarray, weight: float) -> np.ndarray:
    """Logarithmic pooling of two forecasts; weight 1 keeps the model alone."""
    model_probs = np.clip(np.asarray(model_probs, dtype=float), 1e-12, 1.0)
    market_probs = np.clip(np.asarray(market_probs, dtype=float), 1e-12, 1.0)
    pooled = model_probs**weight * market_probs ** (1.0 - weight)
    return pooled / pooled.sum(axis=1, keepdims=True)


def kelly_fraction(probability, odds):
    """Fraction of bankroll for a simple win bet; negative means no bet."""
    probability = np.asarray(probability, dtype=float)
    odds = np.asarray(odds, dtype=float)
    return (probability * odds - 1.0) / (odds - 1.0)


def selection_frame(predictions: pd.DataFrame, odds_columns=CLOSING_ODDS, probability_columns=OUTCOME_COLUMNS) -> pd.DataFrame:
    """One row per selection, with its price, its probability and whether it won."""
    rows = []
    winners = np.where(
        predictions["FTHG"] > predictions["FTAG"], 0,
        np.where(predictions["FTHG"] == predictions["FTAG"], 1, 2),
    )
    for position, (_, match) in enumerate(predictions.iterrows()):
        for index, selection in enumerate(SELECTIONS):
            rows.append({
                "Datetime": match["Datetime"],
                "Season": match["Season"],
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "selection": selection,
                "probability": match[probability_columns[index]],
                "odds": match[odds_columns[index]],
                "won": winners[position] == index,
            })
    frame = pd.DataFrame(rows).dropna(subset=["probability", "odds"])
    frame["edge"] = frame["probability"] * frame["odds"] - 1.0
    frame["kelly"] = kelly_fraction(frame["probability"], frame["odds"])
    return frame


def value_bets(selections: pd.DataFrame, min_edge: float = DEFAULT_MIN_EDGE) -> pd.DataFrame:
    return selections[selections["edge"] >= min_edge].copy()


def simulate(
    bets: pd.DataFrame,
    bankroll: float = STARTING_BANKROLL,
    fraction: float = DEFAULT_KELLY_FRACTION,
    cap: float = DEFAULT_STAKE_CAP,
) -> pd.DataFrame:
    """Settle bets matchday by matchday, staking off the bankroll of that morning."""
    if bets.empty:
        return pd.DataFrame(columns=["Datetime", "stake", "profit", "bankroll"])
    settled = []
    for day, group in bets.sort_values("Datetime").groupby(bets["Datetime"].dt.date, sort=True):
        opening = bankroll
        for _, bet in group.iterrows():
            stake = opening * float(np.clip(bet["kelly"] * fraction, 0.0, cap))
            profit = stake * (bet["odds"] - 1.0) if bet["won"] else -stake
            bankroll += profit
            settled.append({
                "Datetime": bet["Datetime"],
                "day": day,
                "selection": bet["selection"],
                "odds": bet["odds"],
                "probability": bet["probability"],
                "edge": bet["edge"],
                "won": bet["won"],
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            })
    return pd.DataFrame(settled)


def summarise(settled: pd.DataFrame, bankroll: float = STARTING_BANKROLL) -> dict:
    if settled.empty:
        return {"bets": 0, "staked": 0.0, "profit": 0.0, "roi": float("nan"), "final_bankroll": bankroll, "max_drawdown": 0.0}
    equity = settled["bankroll"]
    drawdown = (equity.cummax() - equity) / equity.cummax()
    staked = settled["stake"].sum()
    return {
        "bets": int(len(settled)),
        "staked": float(staked),
        "profit": float(settled["profit"].sum()),
        "roi": float(settled["profit"].sum() / staked) if staked else float("nan"),
        "final_bankroll": float(equity.iloc[-1]),
        "max_drawdown": float(drawdown.max()),
    }


def bootstrap_roi(settled: pd.DataFrame, draws: int = 10_000, seed: int = 0) -> dict:
    """Resample whole matchdays, since bets on the same day settle together."""
    if settled.empty:
        return {"low": float("nan"), "high": float("nan"), "positive_share": float("nan")}
    rng = np.random.default_rng(seed)
    days = list(settled.groupby("day"))
    profits = np.array([group["profit"].sum() for _, group in days])
    stakes = np.array([group["stake"].sum() for _, group in days])
    picks = rng.integers(0, len(days), size=(draws, len(days)))
    sampled_roi = profits[picks].sum(axis=1) / stakes[picks].sum(axis=1)
    return {
        "low": float(np.quantile(sampled_roi, 0.025)),
        "high": float(np.quantile(sampled_roi, 0.975)),
        "positive_share": float((sampled_roi > 0).mean()),
    }


def has_demonstrable_edge(interval: dict) -> bool:
    return bool(interval["low"] > 0)
