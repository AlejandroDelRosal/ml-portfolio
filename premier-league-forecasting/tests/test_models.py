import numpy as np
import pandas as pd
import pytest

from src.models import PROBABILITY_COLUMNS, PROMOTED, GoalModel, appearance_labels, writable


def synthetic_matches(seasons: int = 3, teams: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [f"Team{index}" for index in range(teams)]
    rows = []
    start = pd.Timestamp("2020-08-01")
    for season in range(seasons):
        for round_index in range(teams * 2):
            for position in range(teams // 2):
                home = names[(round_index + position) % teams]
                away = names[(round_index + position + 1 + teams // 2) % teams]
                if home == away:
                    continue
                rows.append({
                    "Season": f"20{20 + season}-{21 + season}",
                    "Datetime": start + pd.Timedelta(days=season * 300 + round_index * 7),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": rng.poisson(1.6),
                    "FTAG": rng.poisson(1.2),
                })
    return pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)


def test_writable_returns_a_mutable_copy():
    series = pd.Series([1, 2, 3])
    array = writable(series, np.int64)
    assert array.flags.writeable
    array[0] = 99
    assert series.iloc[0] == 1


def test_first_appearances_are_labelled_as_promoted():
    matches = synthetic_matches(seasons=1)
    home, away, appearances = appearance_labels(matches, threshold=4)
    assert home[0] == PROMOTED
    assert away[0] == PROMOTED
    assert home[-1] != PROMOTED
    assert all(count > 0 for count in appearances.values())


def test_appearance_counts_never_look_ahead():
    matches = synthetic_matches(seasons=1)
    _, _, appearances = appearance_labels(matches, threshold=4)
    assert sum(appearances.values()) == 2 * len(matches)


def test_model_fits_and_returns_valid_distributions():
    matches = synthetic_matches()
    model = GoalModel().fit(matches)
    predictions = model.predict(matches.tail(5))
    assert list(predictions.columns) == PROBABILITY_COLUMNS
    row_sums = predictions[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert predictions["p_over25"].between(0, 1).all()


def test_unknown_team_falls_back_to_the_promoted_rating():
    matches = synthetic_matches()
    model = GoalModel().fit(matches)
    fixture = pd.DataFrame({"HomeTeam": ["Newcomer"], "AwayTeam": ["Team1"]})
    assert model.label("Newcomer") == PROMOTED
    predictions = model.predict(fixture)
    assert predictions.notna().all(axis=None)


def test_a_side_sitting_on_the_threshold_is_still_priced():
    matches = synthetic_matches(seasons=1)
    newcomer = matches.tail(5).assign(HomeTeam="Newcomer", AwayTeam="Team1")
    model = GoalModel(threshold=len(newcomer)).fit(pd.concat([matches, newcomer]))
    assert model.appearances["Newcomer"] == len(newcomer)
    assert model.label("Newcomer") == PROMOTED
    assert model.can_price("Newcomer", "Team1")
    assert model.predict(pd.DataFrame({"HomeTeam": ["Newcomer"], "AwayTeam": ["Team1"]})).notna().all(axis=None)


def test_pricing_is_impossible_without_a_promoted_reference():
    matches = synthetic_matches()
    model = GoalModel(threshold=0).fit(matches)
    fixture = pd.DataFrame({"HomeTeam": ["Newcomer"], "AwayTeam": ["Team1"]})
    assert model.predict(fixture).isna().all(axis=None)


def test_home_advantage_makes_the_home_side_favourite_between_equals():
    matches = synthetic_matches()
    model = GoalModel().fit(matches)
    fixture = pd.DataFrame({"HomeTeam": ["Team1"], "AwayTeam": ["Team2"]})
    reverse = pd.DataFrame({"HomeTeam": ["Team2"], "AwayTeam": ["Team1"]})
    assert model.predict(fixture)["p_home"].iloc[0] > model.predict(reverse)["p_away"].iloc[0]


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError):
        GoalModel(name="not_a_model")


def test_predicting_before_fitting_is_rejected():
    with pytest.raises(RuntimeError):
        GoalModel().predict(synthetic_matches().head(1))


def with_expected_goals(matches: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return matches.assign(
        HxG=rng.gamma(shape=3.0, scale=0.5, size=len(matches)),
        AxG=rng.gamma(shape=2.5, scale=0.5, size=len(matches)),
    )


def test_the_default_target_is_goals():
    assert GoalModel().target == ("FTHG", "FTAG")


def test_a_model_can_be_fitted_on_expected_goals():
    matches = with_expected_goals(synthetic_matches())
    model = GoalModel(target=("HxG", "AxG")).fit(matches)
    priced = model.predict(matches.tail(4))
    assert priced[PROBABILITY_COLUMNS[:3]].notna().all(axis=None)
    assert priced[PROBABILITY_COLUMNS[:3]].sum(axis=1).round(6).eq(1.0).all()


def test_expected_goals_and_goals_do_not_produce_the_same_forecast():
    matches = with_expected_goals(synthetic_matches())
    on_goals = GoalModel().fit(matches).predict(matches.tail(4))
    on_xg = GoalModel(target=("HxG", "AxG")).fit(matches).predict(matches.tail(4))
    assert not np.allclose(on_goals["p_home"], on_xg["p_home"])


def test_a_match_without_expected_goals_is_left_out_of_the_fit():
    matches = with_expected_goals(synthetic_matches())
    matches.loc[matches.index[:20], ["HxG", "AxG"]] = np.nan
    model = GoalModel(target=("HxG", "AxG")).fit(matches)
    assert model.predict(matches.tail(4))["p_home"].notna().all()
