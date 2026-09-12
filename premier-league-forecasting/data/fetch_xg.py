"""Backfill expected goals for the seasons football-data.co.uk does not publish them for.

The season files carry HxG and AxG only from 2026-27, so the seasons the backtest
walks over have none at all. Understat has them per match. They are written to a
sidecar rather than back into the season file: what was downloaded stays exactly
as it was downloaded, and the loader fills the gap at read time.
"""

import pathlib

import pandas as pd

RAW_DIR = pathlib.Path(__file__).parent / "raw"
LEAGUE = "ENG-Premier League"
# The four digit code is the one Understat reads unambiguously. Its single year
# codes overlap: "2021" returns 2020-21, the same season as "2020".
SEASON_STEMS = ["E0_1920", "E0_2021", "E0_2122", "E0_2223", "E0_2324", "E0_2425", "E0_2526", "E0_2627"]
SEASON_START_MONTH = 7
# 2019-20 was suspended in March and finished on 26 July 2020, so the window has
# to reach past midsummer. It still separates neighbouring seasons, which sit a
# full year apart.
SEASON_END_MONTH = 8

# Understat spells clubs out where football-data.co.uk abbreviates. Verified
# against every season the backtest walks over; anything else keeps its own name.
TEAM_NAMES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
}
COLUMNS = ["HomeTeam", "AwayTeam", "HxG", "AxG"]


def to_football_data(schedule: pd.DataFrame) -> pd.DataFrame:
    """Understat's schedule in the shape and vocabulary of the season files."""
    frame = pd.DataFrame({
        "HomeTeam": schedule["home_team"].map(lambda name: TEAM_NAMES.get(name, name)),
        "AwayTeam": schedule["away_team"].map(lambda name: TEAM_NAMES.get(name, name)),
        "HxG": pd.to_numeric(schedule["home_xg"], errors="coerce"),
        "AxG": pd.to_numeric(schedule["away_xg"], errors="coerce"),
    })
    # A fixture not yet played carries no xG, and a blank row would read as a
    # match where nobody created a chance rather than as an absence of data.
    return frame.dropna(subset=["HxG", "AxG"]).reset_index(drop=True)[COLUMNS]


def season_window(stem: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The calendar a season may fall in. Seasons run August to May."""
    start_year = 2000 + int(stem.split("_")[1][:2])
    return (
        pd.Timestamp(year=start_year, month=SEASON_START_MONTH, day=1),
        pd.Timestamp(year=start_year + 1, month=SEASON_END_MONTH, day=31),
    )


def check_season(schedule: pd.DataFrame, stem: str) -> None:
    """Refuse a schedule that is not the season asked for.

    Merging on the pairing cannot catch this: two neighbouring seasons share most
    of their fixtures, so the wrong season's expected goals attach quietly to the
    right season's matches. The dates are the only thing that gives it away.
    """
    start, end = season_window(stem)
    dates = pd.to_datetime(schedule["date"])
    if dates.min() < start or dates.max() > end:
        raise ValueError(
            f"{stem}: Understat returned {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}, "
            f"outside {start:%Y-%m-%d}..{end:%Y-%m-%d}"
        )


def write_sidecar(frame: pd.DataFrame, stem: str, directory: pathlib.Path = RAW_DIR) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"xg_{stem}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def fetch(stem: str) -> pd.DataFrame:
    """Imported here so the sidecars can be read without the scraper installed."""
    import soccerdata

    schedule = soccerdata.Understat(leagues=LEAGUE, seasons=stem.split("_")[1]).read_schedule()
    check_season(schedule, stem)
    return schedule


def main(stems: list[str] | None = None) -> None:
    for stem in stems or SEASON_STEMS:
        frame = to_football_data(fetch(stem))
        path = write_sidecar(frame, stem)
        print(f"{stem}: expected goals for {len(frame)} matches in {path.name}")


if __name__ == "__main__":
    main()
