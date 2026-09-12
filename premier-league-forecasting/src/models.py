import numpy as np
import pandas as pd
import penaltyblog as pb

MODEL_CLASSES = {
    "poisson": pb.models.PoissonGoalsModel,
    "dixon_coles": pb.models.DixonColesGoalModel,
    "bivariate_poisson": pb.models.BivariatePoissonGoalModel,
    "negative_binomial": pb.models.NegativeBinomialGoalModel,
    "zero_inflated": pb.models.ZeroInflatedPoissonGoalsModel,
}
DEFAULT_MODEL = "dixon_coles"
DEFAULT_XI = 0.0018
# A side needs this many prior matches before it gets its own attack and defence
# ratings; below it, it is priced as a generic newly promoted side.
PROMOTED_MATCHES = 10
PROMOTED = "__promoted__"
OUTCOME_COLUMNS = ["p_home", "p_draw", "p_away"]
PROBABILITY_COLUMNS = OUTCOME_COLUMNS + ["p_over25"]


def writable(series: pd.Series, dtype) -> np.ndarray:
    # pandas 3 hands back read-only arrays and the Cython loss needs writable ones.
    return series.to_numpy(dtype=dtype, copy=True)


def appearance_labels(matches: pd.DataFrame, threshold: int = PROMOTED_MATCHES):
    """Label each side, replacing barely seen teams with a shared promoted side."""
    appearances: dict[str, int] = {}
    home_labels, away_labels = [], []
    for home, away in zip(matches["HomeTeam"], matches["AwayTeam"]):
        home_labels.append(home if appearances.get(home, 0) >= threshold else PROMOTED)
        away_labels.append(away if appearances.get(away, 0) >= threshold else PROMOTED)
        appearances[home] = appearances.get(home, 0) + 1
        appearances[away] = appearances.get(away, 0) + 1
    return home_labels, away_labels, appearances


GOALS = ("FTHG", "FTAG")
EXPECTED_GOALS = ("HxG", "AxG")
# Goals are counts; expected goals are not, and rounding them would throw away
# the fraction of a chance that separates a good shot from a tap-in.
TARGET_DTYPES = {GOALS: np.int64, EXPECTED_GOALS: np.float64}


class GoalModel:
    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        xi: float = DEFAULT_XI,
        threshold: int = PROMOTED_MATCHES,
        target: tuple[str, str] = GOALS,
    ):
        if name not in MODEL_CLASSES:
            raise KeyError(f"unknown model {name}; choose from {sorted(MODEL_CLASSES)}")
        self.name = name
        self.xi = xi
        self.threshold = threshold
        self.target = tuple(target)
        self.model = None
        self.appearances: dict[str, int] = {}
        self.teams: set[str] = set()

    def fit(self, matches: pd.DataFrame) -> "GoalModel":
        home_column, away_column = self.target
        played = matches.dropna(subset=[home_column, away_column])
        home_labels, away_labels, appearances = appearance_labels(played, self.threshold)
        weights = pb.models.dixon_coles_weights(played["Datetime"], xi=self.xi) if self.xi else None
        dtype = TARGET_DTYPES.get(self.target, np.float64)
        self.model = MODEL_CLASSES[self.name](
            writable(played[home_column], dtype),
            writable(played[away_column], dtype),
            np.array(home_labels, dtype=object),
            np.array(away_labels, dtype=object),
            weights=np.asarray(weights, dtype=np.float64).copy() if weights is not None else None,
        )
        self.model.fit()
        self.appearances = appearances
        self.teams = set(home_labels) | set(away_labels)
        return self

    def label(self, team: str) -> str:
        # A side sitting exactly on the threshold spent every training row as the
        # promoted side, so its own name never reached the fit.
        known = self.appearances.get(team, 0) >= self.threshold and team in self.teams
        return team if known else PROMOTED

    def can_price(self, home: str, away: str) -> bool:
        return self.label(home) in self.teams and self.label(away) in self.teams

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit the model before predicting")
        rows = []
        for home, away in zip(fixtures["HomeTeam"], fixtures["AwayTeam"]):
            if not self.can_price(home, away):
                rows.append([np.nan] * len(PROBABILITY_COLUMNS))
                continue
            grid = self.model.predict(self.label(home), self.label(away))
            home_win, draw, away_win = grid.home_draw_away
            rows.append([home_win, draw, away_win, grid.total_goals("over", 2.5)])
        return pd.DataFrame(rows, columns=PROBABILITY_COLUMNS, index=fixtures.index)
