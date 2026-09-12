import pathlib

import pandas as pd

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"

MATCH_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
]
STAT_COLUMNS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
# Only published from 2026-27 onwards; earlier seasons carry them as missing.
XG_COLUMNS = ["HxG", "AxG"]
OPENING_1X2 = ["B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"]
CLOSING_1X2 = ["B365CH", "B365CD", "B365CA", "AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA"]
GOALS_ODDS = [
    "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5",
    "B365C>2.5", "B365C<2.5", "AvgC>2.5", "AvgC<2.5",
]
HANDICAP_ODDS = ["AHh", "AvgAHH", "AvgAHA", "AHCh", "AvgCAHH", "AvgCAHA"]

ODDS_COLUMNS = OPENING_1X2 + CLOSING_1X2 + GOALS_ODDS + HANDICAP_ODDS
NUMERIC_COLUMNS = ["FTHG", "FTAG", "HTHG", "HTAG"] + STAT_COLUMNS + XG_COLUMNS + ODDS_COLUMNS
CANONICAL_COLUMNS = MATCH_COLUMNS + STAT_COLUMNS + XG_COLUMNS + ODDS_COLUMNS


FIXTURES_PATH = RAW_DIR / "fixtures.csv"
COMPETITION = "E0"
# Kick-off times are published in England's local time, so anything comparing a
# clock against them has to be in that clock, not the machine's.
LEAGUE_TIMEZONE = "Europe/London"
# Seasons run August to May, so a January match belongs to the year before it.
SEASON_START_MONTH = 7


def season_label(path: pathlib.Path) -> str:
    code = path.stem.split("_")[1]
    return f"20{code[:2]}-{code[2:]}"


def league_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=LEAGUE_TIMEZONE).tz_localize(None)


def season_from_date(moment: pd.Timestamp) -> str:
    start = moment.year if moment.month > SEASON_START_MONTH else moment.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def _parse_datetime(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")
    two_digit = pd.to_datetime(frame["Date"], format="%d/%m/%y", errors="coerce")
    date = date.fillna(two_digit)
    time = frame["Time"].fillna("00:00").replace("", "00:00")
    stamp = date.dt.strftime("%Y-%m-%d") + " " + time.astype(str)
    return pd.to_datetime(stamp, format="%Y-%m-%d %H:%M", errors="coerce")


def attach_xg(season: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Fill missing expected goals from a sidecar, keyed on the pairing.

    A pairing occurs once per season, so it identifies the match without leaning
    on a kick-off time that may have moved since either file was written.
    """
    key = ["HomeTeam", "AwayTeam"]
    filled = season.copy()
    for column in XG_COLUMNS:
        if column not in filled.columns:
            filled[column] = pd.NA
    merged = filled.merge(sidecar[key + XG_COLUMNS], on=key, how="left", suffixes=("", "_sidecar"))
    for column in XG_COLUMNS:
        # The sidecar wins. football-data publishes expected goals only from
        # 2026-27 and from a different provider, so deferring to it would fit the
        # model on two definitions of a chance with a break between them. What
        # the season file publishes is kept only where the sidecar has nothing.
        published = pd.to_numeric(merged[column], errors="coerce")
        incoming = pd.to_numeric(merged[f"{column}_sidecar"], errors="coerce")
        filled[column] = incoming.fillna(published).values
    return filled


def load_season(path: pathlib.Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[frame["HomeTeam"].notna()].copy()
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[CANONICAL_COLUMNS].copy()
    frame["Season"] = season_label(path)
    frame["Datetime"] = _parse_datetime(frame)
    sidecar = path.parent / f"xg_{path.stem}.csv"
    if sidecar.exists():
        frame = attach_xg(frame, pd.read_csv(sidecar, encoding="utf-8-sig"))
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_fixtures(path: pathlib.Path = FIXTURES_PATH, competition: str = COMPETITION) -> pd.DataFrame:
    """Upcoming matches, shaped like played ones so forecasters need no special case."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[(frame["Div"] == competition) & frame["HomeTeam"].notna()].copy()
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[CANONICAL_COLUMNS].copy()
    frame["Datetime"] = _parse_datetime(frame)
    frame["Season"] = frame["Datetime"].map(season_from_date)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[CANONICAL_COLUMNS + ["Season", "Datetime"]]
    return frame.sort_values("Datetime", kind="mergesort").reset_index(drop=True)


def load_matches(raw_dir: pathlib.Path = RAW_DIR, seasons: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("E0_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no season files in {raw_dir}; run data/fetch_football_data.py first")
    frames = [load_season(path) for path in paths]
    matches = pd.concat(frames, ignore_index=True)
    if seasons is not None:
        matches = matches[matches["Season"].isin(seasons)]
    # mergesort keeps same-kickoff matches in file order, so the sort is reproducible.
    matches = matches.sort_values("Datetime", kind="mergesort").reset_index(drop=True)
    return matches
