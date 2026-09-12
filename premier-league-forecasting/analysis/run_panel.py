"""Ask the free model panel about the coming matchweek, and score what it said before.

The panel is only ever asked about matches that have not kicked off, because a
language model that already knows the result is not forecasting.
"""

import json
import pathlib

import pandas as pd

from src.llm_panel import PANEL_SIZE, free_models, load_records, record, run_panel
from src.loader import league_now, load_fixtures, load_matches
from src.market import MarketPredictor, OPENING_ODDS
from src.metrics import compare, evaluate
from src.models import OUTCOME_COLUMNS
from src.ratings import result_codes

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


def ask(now: pd.Timestamp) -> dict:
    fixtures = load_fixtures()
    fixtures = fixtures[fixtures["Datetime"] > now]
    if fixtures.empty:
        print("No fixture ahead; the panel has nothing to forecast.")
        return {}
    models = free_models(limit=PANEL_SIZE)
    print(f"Asking {len(models)} free models about {len(fixtures)} fixtures:")
    for model in models:
        print(f"  {model}")
    panel = run_panel(fixtures, models=models)
    if not panel:
        print("No model answered usably.")
        return {}
    path = record(panel, fixtures, asked_at=now)
    print(f"\nRecorded {len(panel)} model answers in {path}")
    for model, predictions in panel.items():
        answered = predictions[OUTCOME_COLUMNS].notna().all(axis=1).sum()
        print(f"  {model}: {answered}/{len(fixtures)} fixtures priced")
    return panel


def score() -> dict:
    """Rank the panel against the market on the matches it was asked about."""
    records = load_records()
    if records.empty:
        print("\nNothing recorded yet.")
        return {}
    played = load_matches()
    played = played[played["FTR"].notna()]
    key = ["HomeTeam", "AwayTeam"]
    played["day"] = played["Datetime"].dt.date
    records["day"] = records["Datetime"].dt.date
    joined = records.merge(played[key + ["day", "FTHG", "FTAG", "FTR"] + OPENING_ODDS], on=key + ["day"], how="inner")
    joined = joined.dropna(subset=OUTCOME_COLUMNS)
    if joined.empty:
        print("\nThe recorded fixtures have not been played yet.")
        return {}

    codes = result_codes(joined)
    results = {}
    for model, group in joined.groupby("model"):
        results[model] = evaluate(group[OUTCOME_COLUMNS].to_numpy(), result_codes(group))
    market = MarketPredictor(columns=OPENING_ODDS).predict(joined)
    results["market_opening"] = evaluate(market[OUTCOME_COLUMNS].to_numpy(), codes)

    print(f"\n=== Panel scored on {joined['day'].nunique()} matchdays, {len(joined)} model forecasts ===")
    print(compare(results).to_string(float_format=lambda value: f"{value:.5f}"))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "llm_panel_scores.json").write_text(json.dumps(results, indent=2))
    return results


def run(now: pd.Timestamp | None = None) -> dict:
    now = now or league_now()
    panel = ask(now)
    return {"asked": len(panel), "scores": score()}


if __name__ == "__main__":
    run()
