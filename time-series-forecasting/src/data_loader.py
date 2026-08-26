import pathlib

import pandas as pd

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "unrate_raw.csv"


def load_unemployment_rate(path: pathlib.Path = DATA_PATH) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.set_index("observation_date")
    series = df["UNRATE"]
    series.index.freq = "MS"
    # October 2025 is missing (delayed BLS release during the government
    # shutdown); linearly interpolate the single-month gap.
    return series.interpolate()
