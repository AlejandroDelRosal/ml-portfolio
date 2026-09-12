import pandas as pd
import pytest

from data import fetch_xg

SCHEDULE = pd.DataFrame({
    "home_team": ["Manchester City", "Arsenal"],
    "away_team": ["Nottingham Forest", "Wolverhampton Wanderers"],
    "home_xg": [2.31, 1.04],
    "away_xg": [0.42, 1.55],
})


def test_understat_club_names_become_football_data_names():
    converted = fetch_xg.to_football_data(SCHEDULE)
    assert list(converted["HomeTeam"]) == ["Man City", "Arsenal"]
    assert list(converted["AwayTeam"]) == ["Nott'm Forest", "Wolves"]


def test_only_the_pairing_and_its_expected_goals_are_kept():
    assert list(fetch_xg.to_football_data(SCHEDULE).columns) == ["HomeTeam", "AwayTeam", "HxG", "AxG"]


def test_an_unmapped_club_keeps_its_own_name():
    other = pd.DataFrame({"home_team": ["Everton"], "away_team": ["Fulham"], "home_xg": [1.0], "away_xg": [0.5]})
    assert fetch_xg.to_football_data(other)["HomeTeam"].item() == "Everton"


def test_a_sidecar_is_written_next_to_the_season_it_describes(tmp_path):
    path = fetch_xg.write_sidecar(fetch_xg.to_football_data(SCHEDULE), "E0_2324", directory=tmp_path)
    assert path.name == "xg_E0_2324.csv"
    assert list(pd.read_csv(path)["HomeTeam"]) == ["Man City", "Arsenal"]


def test_a_fixture_understat_never_played_is_dropped_rather_than_written(tmp_path):
    unplayed = pd.DataFrame({
        "home_team": ["Everton"], "away_team": ["Fulham"],
        "home_xg": [float("nan")], "away_xg": [float("nan")],
    })
    assert fetch_xg.to_football_data(unplayed).empty


def schedule_dated(first, last):
    return pd.DataFrame({
        "home_team": ["Arsenal", "Everton"], "away_team": ["Everton", "Arsenal"],
        "home_xg": [1.0, 0.8], "away_xg": [0.5, 1.2],
        "date": [pd.Timestamp(first), pd.Timestamp(last)],
    })


def test_the_season_window_brackets_august_to_may():
    start, end = fetch_xg.season_window("E0_2122")
    assert start.year == 2021 and end.year == 2022


def test_a_schedule_inside_its_season_window_is_accepted():
    fetch_xg.check_season(schedule_dated("2021-08-13", "2022-05-22"), "E0_2122")


def test_a_schedule_from_the_wrong_season_is_refused():
    # Understat's single-year season codes are ambiguous and have silently
    # returned the previous season, which pairing-based merging cannot detect.
    with pytest.raises(ValueError, match="outside"):
        fetch_xg.check_season(schedule_dated("2020-09-12", "2021-05-23"), "E0_2122")


def test_west_bromwich_albion_is_mapped():
    frame = pd.DataFrame({
        "home_team": ["West Bromwich Albion"], "away_team": ["Arsenal"],
        "home_xg": [1.1], "away_xg": [0.9],
    })
    assert fetch_xg.to_football_data(frame)["HomeTeam"].item() == "West Brom"


def test_the_season_suspended_by_the_pandemic_is_still_accepted():
    # 2019-20 was halted in March and finished on 26 July 2020.
    fetch_xg.check_season(schedule_dated("2019-08-09", "2020-07-26"), "E0_1920")


def test_the_window_still_separates_neighbouring_seasons():
    with pytest.raises(ValueError, match="outside"):
        fetch_xg.check_season(schedule_dated("2020-09-12", "2021-05-23"), "E0_2122")
