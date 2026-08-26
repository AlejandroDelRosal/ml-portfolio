import pathlib

import requests

SERIES_ID = "UNRATE"
URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}"
OUTPUT_PATH = pathlib.Path(__file__).parent / "unrate_raw.csv"


def fetch() -> None:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    OUTPUT_PATH.write_bytes(response.content)


if __name__ == "__main__":
    fetch()
