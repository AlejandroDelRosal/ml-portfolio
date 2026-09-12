"""Forecast the upcoming matchweek and flag anything that looks like value.

Upcoming matches only have opening prices, so the edge is measured against those.
If the backtest did not demonstrate an edge, the table is printed as information
and every stake is held at zero.
"""

import json
import pathlib

import pandas as pd

from src.betting import DEFAULT_KELLY_FRACTION, DEFAULT_MIN_EDGE, DEFAULT_STAKE_CAP, blend, selection_frame, value_bets
from src.loader import load_fixtures, load_matches
from src.market import OPENING_ODDS, MarketPredictor
from src.models import DEFAULT_XI, OUTCOME_COLUMNS, GoalModel

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
BANKROLL = 1000.0


def settings() -> dict:
    """Whatever the backtest and the betting study last decided."""
    chosen = {"xi": DEFAULT_XI, "weight": 1.0, "min_edge": DEFAULT_MIN_EDGE, "demonstrable_edge": False}
    backtest = RESULTS_DIR / "backtest.json"
    betting = RESULTS_DIR / "betting.json"
    if backtest.exists():
        chosen["xi"] = json.loads(backtest.read_text())["xi"]
    if betting.exists():
        study = json.loads(betting.read_text())
        chosen["weight"] = study["weight"]
        chosen["min_edge"] = study["edges"]["opening"]
        chosen["demonstrable_edge"] = study["books"]["opening"]["demonstrable_edge"]
    return chosen


def upcoming(fixtures: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    return fixtures[fixtures["Datetime"] >= now]


def run(now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now()
    chosen = settings()
    history = load_matches()
    fixtures = upcoming(load_fixtures(), now)
    if fixtures.empty:
        print("No Premier League fixture ahead in the published file.")
        return {"fixtures": 0}

    model = GoalModel(xi=chosen["xi"]).fit(history)
    model_probs = model.predict(fixtures)
    market_probs = MarketPredictor(columns=OPENING_ODDS).predict(fixtures)
    pooled = blend(model_probs[OUTCOME_COLUMNS].to_numpy(), market_probs[OUTCOME_COLUMNS].to_numpy(), chosen["weight"])

    table = fixtures[["Datetime", "HomeTeam", "AwayTeam"] + OPENING_ODDS].copy()
    table[OUTCOME_COLUMNS] = pooled
    for name, source in [("model", model_probs), ("market", market_probs)]:
        for column in OUTCOME_COLUMNS:
            table[f"{column}_{name}"] = source[column].to_numpy()

    priced = table.dropna(subset=OUTCOME_COLUMNS).copy()
    priced["FTHG"] = pd.NA
    priced["FTAG"] = pd.NA
    priced["Season"] = fixtures.loc[priced.index, "Season"]
    selections = selection_frame(priced, odds_columns=OPENING_ODDS)
    picks = value_bets(selections, min_edge=chosen["min_edge"]).sort_values("edge", ascending=False)
    picks["stake"] = 0.0 if not chosen["demonstrable_edge"] else (
        BANKROLL * (picks["kelly"] * DEFAULT_KELLY_FRACTION).clip(0.0, DEFAULT_STAKE_CAP)
    )

    print(f"Matchweek forecast as of {now:%Y-%m-%d %H:%M}, blend weight {chosen['weight']}, decay {chosen['xi']}\n")
    shown = table.dropna(subset=OUTCOME_COLUMNS)[["Datetime", "HomeTeam", "AwayTeam"] + OUTCOME_COLUMNS]
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print(f"\nSelections with an edge of {chosen['min_edge']:.0%} or better:")
    if picks.empty:
        print("  none")
    else:
        columns = ["Datetime", "HomeTeam", "AwayTeam", "selection", "probability", "odds", "edge", "stake"]
        print(picks[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if not chosen["demonstrable_edge"]:
        print("\nThe backtest showed no demonstrable edge, so every stake is held at zero. This table is information, not advice.")

    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": now.isoformat(),
        "settings": chosen,
        "fixtures": json.loads(priced.drop(columns=["FTHG", "FTAG"]).to_json(orient="records", date_format="iso")),
        "picks": json.loads(picks.to_json(orient="records", date_format="iso")),
    }
    path = RESULTS_DIR / f"matchweek_{now:%Y%m%d}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {path}")
    return payload


if __name__ == "__main__":
    run()
