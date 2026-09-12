import pandas as pd

from src.models import OUTCOME_COLUMNS

MIN_TRAIN_MATCHES = 760


class LeakageError(AssertionError):
    pass


def walk_forward(
    matches: pd.DataFrame,
    predictor_factory,
    seasons: list[str],
    min_train: int = MIN_TRAIN_MATCHES,
) -> pd.DataFrame:
    """Predict each matchday using only matches that kicked off before it.

    predictor_factory builds a fresh predictor per matchday; the predictor takes
    a training frame in fit and returns one probability row per fixture.
    """
    matches = matches.sort_values("Datetime", kind="mergesort")
    target = matches[matches["Season"].isin(seasons)]
    predictions = []
    for day, fixtures in target.groupby(target["Datetime"].dt.date, sort=True):
        train = matches[matches["Datetime"] < fixtures["Datetime"].min()]
        if len(train) < min_train:
            continue
        if not train["Datetime"].max() < fixtures["Datetime"].min():
            raise LeakageError(f"training data reaches into {day}")
        predictor = predictor_factory()
        predictor.fit(train)
        probabilities = predictor.predict(fixtures)
        predictions.append(fixtures.join(probabilities))
    if not predictions:
        raise ValueError("no matchday had enough training history")
    return pd.concat(predictions).sort_values("Datetime", kind="mergesort")


def priced(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rows the forecaster could actually price; not every model prices goals."""
    return predictions.dropna(subset=OUTCOME_COLUMNS)
