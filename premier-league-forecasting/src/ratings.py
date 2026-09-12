import numpy as np
import pandas as pd
import penaltyblog as pb

from src.models import PROBABILITY_COLUMNS

# penaltyblog encodes results the same way the market module does.
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2


def result_codes(matches: pd.DataFrame) -> np.ndarray:
    home, away = matches["FTHG"].to_numpy(), matches["FTAG"].to_numpy()
    return np.where(home > away, HOME_WIN, np.where(home == away, DRAW, AWAY_WIN))


def _probability_frame(rows, index) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PROBABILITY_COLUMNS, index=index)


class EloPredictor:
    """Elo ratings updated match by match, read out as 1X2 probabilities."""

    def __init__(self, k: float = 20.0, home_field_advantage: float = 100.0, draw_base: float = 0.3, draw_width: float = 200.0):
        self.k = k
        self.home_field_advantage = home_field_advantage
        self.draw_base = draw_base
        self.draw_width = draw_width
        self.model = None

    def fit(self, matches: pd.DataFrame) -> "EloPredictor":
        played = matches.dropna(subset=["FTHG", "FTAG"])
        self.model = pb.ratings.Elo(k=self.k, home_field_advantage=self.home_field_advantage)
        for home, away, code in zip(played["HomeTeam"], played["AwayTeam"], result_codes(played)):
            self.model.update_ratings(home, away, int(code))
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit the model before predicting")
        rows = []
        for home, away in zip(fixtures["HomeTeam"], fixtures["AwayTeam"]):
            probs = self.model.calculate_match_probabilities(home, away, self.draw_base, self.draw_width)
            # Elo says nothing about how many goals get scored.
            rows.append([probs["home_win"], probs["draw"], probs["away_win"], np.nan])
        return _probability_frame(rows, fixtures.index)

    def rating(self, team: str) -> float:
        return self.model.get_team_rating(team)


class PiRatingsPredictor:
    """Constantinou and Fenton pi-ratings, which learn from the goal margin."""

    def __init__(self, alpha: float = 0.15, beta: float = 0.1, k: float = 0.75, sigma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.sigma = sigma
        self.model = None

    def fit(self, matches: pd.DataFrame) -> "PiRatingsPredictor":
        played = matches.dropna(subset=["FTHG", "FTAG"])
        self.model = pb.ratings.PiRatingSystem(alpha=self.alpha, beta=self.beta, k=self.k, sigma=self.sigma)
        margins = (played["FTHG"] - played["FTAG"]).to_numpy(dtype=int)
        for home, away, margin, date in zip(played["HomeTeam"], played["AwayTeam"], margins, played["Datetime"]):
            self.model.update_ratings(home, away, int(margin), date=date)
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit the model before predicting")
        rows = []
        for home, away in zip(fixtures["HomeTeam"], fixtures["AwayTeam"]):
            probs = self.model.calculate_match_probabilities(home, away)
            rows.append([probs["home_win"], probs["draw"], probs["away_win"], np.nan])
        return _probability_frame(rows, fixtures.index)

    def expected_margin(self, home: str, away: str) -> float:
        return self.model.expected_goal_difference(home, away)
