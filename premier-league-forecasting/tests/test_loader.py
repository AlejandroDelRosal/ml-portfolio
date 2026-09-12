import pathlib

import pandas as pd
import pytest

from src.loader import (
    CANONICAL_COLUMNS,
    load_fixtures,
    load_matches,
    load_season,
    season_from_date,
    season_label,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CURRENT = FIXTURES / "E0_2627_sample.csv"
EARLIER = FIXTURES / "E0_2324_sample.csv"
UPCOMING = FIXTURES / "fixtures_sample.csv"


def test_season_label_expands_the_file_code():
    assert season_label(CURRENT) == "2026-27"
    assert season_label(EARLIER) == "2023-24"


def test_load_season_keeps_only_canonical_columns():
    frame = load_season(CURRENT)
    assert list(frame.columns) == CANONICAL_COLUMNS + ["Season", "Datetime"]


def test_blank_trailing_rows_are_dropped():
    raw = pd.read_csv(CURRENT, encoding="utf-8-sig")
    frame = load_season(CURRENT)
    assert len(raw) == 7
    assert len(frame) == 6


def test_kickoff_is_parsed_as_a_timestamp():
    frame = load_season(CURRENT)
    assert frame["Datetime"].notna().all()
    assert frame["Datetime"].dt.year.unique().tolist() == [2026]


def test_missing_expected_goals_are_present_but_empty():
    earlier = load_season(EARLIER)
    current = load_season(CURRENT)
    assert earlier["HxG"].isna().all()
    assert current["HxG"].notna().all()


def test_goals_and_odds_are_numeric():
    frame = load_season(CURRENT)
    for column in ["FTHG", "FTAG", "AvgCH", "AvgCD", "AvgCA", "AHh"]:
        assert pd.api.types.is_numeric_dtype(frame[column])


def test_load_matches_is_sorted_and_spans_both_seasons():
    matches = load_matches(raw_dir=FIXTURES)
    assert matches["Datetime"].is_monotonic_increasing
    assert set(matches["Season"]) == {"2023-24", "2026-27"}
    assert len(matches) == 12


def test_load_matches_can_filter_seasons():
    matches = load_matches(raw_dir=FIXTURES, seasons=["2026-27"])
    assert set(matches["Season"]) == {"2026-27"}


def test_missing_directory_is_reported():
    with pytest.raises(FileNotFoundError):
        load_matches(raw_dir=FIXTURES / "does-not-exist")


@pytest.mark.parametrize("date,expected", [
    ("2026-08-21", "2026-27"),
    ("2026-12-26", "2026-27"),
    ("2027-05-30", "2026-27"),
    ("2026-07-15", "2025-26"),
])
def test_season_boundary_falls_in_the_summer(date, expected):
    assert season_from_date(pd.Timestamp(date)) == expected


def test_fixtures_load_with_the_same_shape_as_played_matches():
    upcoming = load_fixtures(UPCOMING)
    played = load_season(CURRENT)
    assert list(upcoming.columns) == list(played.columns)
    assert len(upcoming) == 10


def test_fixtures_have_prices_but_no_result():
    upcoming = load_fixtures(UPCOMING)
    assert upcoming["FTR"].isna().all()
    assert upcoming[["AvgH", "AvgD", "AvgA"]].notna().all(axis=None)


def test_fixtures_keep_only_the_requested_competition():
    assert set(load_fixtures(UPCOMING)["Div"]) == {"E0"}


def test_fixtures_are_chronological_and_dated():
    upcoming = load_fixtures(UPCOMING)
    assert upcoming["Datetime"].is_monotonic_increasing
    assert upcoming["Datetime"].notna().all()
    assert set(upcoming["Season"]) == {"2026-27"}
