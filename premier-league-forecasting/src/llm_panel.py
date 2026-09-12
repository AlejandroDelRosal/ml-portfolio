"""Ask free language models for match probabilities and record them prospectively.

Language models already know how past seasons ended, so scoring them on history
measures memory, not forecasting. Every prediction here is stamped and stored
before kick-off, and the recorder refuses anything else.
"""

import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
import requests

from src.models import OUTCOME_COLUMNS, PROBABILITY_COLUMNS

OPENROUTER_URL = "https://openrouter.ai/api/v1"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
PANEL_SIZE = 6
REQUEST_TIMEOUT = 180
PANEL_DIR = pathlib.Path(__file__).parent.parent / "results" / "llm_panel"
# A fixture is the pairing at a kick-off: the same teams rescheduled is a new one.
FIXTURE_KEY = ["HomeTeam", "AwayTeam", "Datetime"]

PROMPT = """You are forecasting English Premier League matches.

For each fixture below give calibrated probabilities for home win, draw and away win.
The three probabilities must sum to 1. Favour realism over boldness: draws happen in
about a quarter of matches and heavy favourites still lose.

Reply with nothing but a JSON array, one object per fixture, in this exact shape:
[{{"home": "Team", "away": "Team", "p_home": 0.0, "p_draw": 0.0, "p_away": 0.0}}]

Fixtures:
{fixtures}
"""


class PanelError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise PanelError(f"set {API_KEY_VARIABLE} to query the panel")
    return key


def free_models(limit: int = PANEL_SIZE, session=requests) -> list[str]:
    """The free roster rotates, so it is read live rather than hard coded."""
    response = session.get(f"{OPENROUTER_URL}/models", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    free = [model for model in response.json()["data"] if model["id"].endswith(":free")]
    free.sort(key=lambda model: model.get("context_length") or 0, reverse=True)
    return [model["id"] for model in free[:limit]]


def build_prompt(fixtures: pd.DataFrame) -> str:
    lines = [f"{home} vs {away}" for home, away in zip(fixtures["HomeTeam"], fixtures["AwayTeam"])]
    return PROMPT.format(fixtures="\n".join(lines))


def extract_json(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise PanelError("no JSON array in the reply")
    return json.loads(match.group(0))


CLUB_SUFFIXES = ("fc", "afc", "cf")


def _normalise(name: str) -> str:
    letters = re.sub(r"[^a-z]", "", str(name).lower())
    for suffix in CLUB_SUFFIXES:
        if letters.endswith(suffix) and len(letters) > len(suffix) + 2:
            return letters[: -len(suffix)]
    return letters


def parse_response(text: str, fixtures: pd.DataFrame) -> pd.DataFrame:
    entries = extract_json(text)
    by_pair = {(_normalise(entry.get("home", "")), _normalise(entry.get("away", ""))): entry for entry in entries}
    rows = []
    for position, (home, away) in enumerate(zip(fixtures["HomeTeam"], fixtures["AwayTeam"])):
        entry = by_pair.get((_normalise(home), _normalise(away)))
        # Models write "Manchester United" where the data says "Man United", so a
        # complete reply in the order asked is matched by position instead.
        if entry is None and len(entries) == len(fixtures):
            entry = entries[position]
        rows.append(_row(entry))
    return pd.DataFrame(rows, columns=PROBABILITY_COLUMNS, index=fixtures.index)


def _row(entry) -> list[float]:
    missing = [np.nan] * len(PROBABILITY_COLUMNS)
    if not entry:
        return missing
    try:
        probabilities = np.array([float(entry["p_home"]), float(entry["p_draw"]), float(entry["p_away"])])
    except (KeyError, TypeError, ValueError):
        return missing
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0 or (probabilities < 0).any():
        return missing
    # Models rarely sum to exactly one, which is a rounding slip rather than a refusal.
    return list(probabilities / total) + [np.nan]


def ask_model(model: str, fixtures: pd.DataFrame, key: str, session=requests) -> pd.DataFrame:
    """One request per model per matchday keeps the panel inside the free tier."""
    response = session.post(
        f"{OPENROUTER_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": build_prompt(fixtures)}], "temperature": 0},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_response(content, fixtures)


def run_panel(fixtures: pd.DataFrame, models: list[str] | None = None, key: str | None = None, session=requests) -> dict:
    key = key or api_key()
    models = models or free_models(session=session)
    panel = {}
    for model in models:
        try:
            panel[model] = ask_model(model, fixtures, key, session=session)
        except Exception as exc:
            print(f"  {model}: {type(exc).__name__}: {str(exc)[:80]}")
    return panel


def record(panel: dict, fixtures: pd.DataFrame, asked_at: pd.Timestamp, directory: pathlib.Path = PANEL_DIR) -> pathlib.Path:
    """Store the panel, refusing anything that is not a genuine forecast."""
    kickoff = fixtures["Datetime"].min()
    if asked_at >= kickoff:
        raise PanelError(f"asked at {asked_at} but the first kick-off was {kickoff}; that is not a forecast")
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for model, predictions in panel.items():
        frame = fixtures[["Datetime", "HomeTeam", "AwayTeam"]].join(predictions)
        frame.insert(0, "model", model)
        frame.insert(1, "asked_at", asked_at)
        frames.append(frame)
    path = directory / f"panel_{asked_at:%Y%m%d_%H%M}.csv"
    pd.concat(frames).to_csv(path, index=False)
    return path


def snapshot(panel: dict, fixtures: pd.DataFrame, asked_at: pd.Timestamp) -> dict:
    """A JSON-shaped view of the panel, so the report stays free of pandas."""
    rows = []
    for position in range(len(fixtures)):
        fixture = fixtures.iloc[position]
        forecasts = {}
        for model, predictions in panel.items():
            values = predictions.iloc[position][OUTCOME_COLUMNS]
            # A model that skipped a fixture is absent rather than present and empty.
            if values.notna().all():
                forecasts[model] = [float(value) for value in values]
        rows.append({
            "kickoff": fixture["Datetime"].isoformat(),
            "home": str(fixture["HomeTeam"]),
            "away": str(fixture["AwayTeam"]),
            "forecasts": forecasts,
        })
    return {"asked_at": asked_at.isoformat(), "models": list(panel), "fixtures": rows}


def unasked(fixtures: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Fixtures with no forecast on file, so running twice never asks twice."""
    if records.empty:
        return fixtures
    already = pd.MultiIndex.from_frame(records[FIXTURE_KEY])
    current = pd.MultiIndex.from_frame(fixtures[FIXTURE_KEY])
    return fixtures[~current.isin(already)]


def load_records(directory: pathlib.Path = PANEL_DIR) -> pd.DataFrame:
    paths = sorted(directory.glob("panel_*.csv"))
    if not paths:
        return pd.DataFrame(columns=["model", "asked_at", "Datetime", "HomeTeam", "AwayTeam"] + PROBABILITY_COLUMNS)
    frames = [pd.read_csv(path, parse_dates=["asked_at", "Datetime"]) for path in paths]
    records = pd.concat(frames, ignore_index=True).sort_values("asked_at", kind="mergesort")
    # A fixture asked more than once counts once, at the earliest asking.
    return records.drop_duplicates(subset=["model"] + FIXTURE_KEY, keep="first").reset_index(drop=True)
