import argparse
import pathlib

import requests

# The www host answers 302 and only the bare host serves the files directly.
BASE_URL = "https://football-data.co.uk"
COMPETITION = "E0"
SEASONS = ("1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627")
RAW_DIR = pathlib.Path(__file__).parent / "raw"
HEADERS = {"User-Agent": "premier-league-forecasting"}


def season_url(season: str) -> str:
    return f"{BASE_URL}/mmz4281/{season}/{COMPETITION}.csv"


def download(url: str, path: pathlib.Path) -> int:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return len(response.content)


def fetch(seasons: tuple[str, ...] = SEASONS, force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = seasons[-1]
    for season in seasons:
        path = RAW_DIR / f"{COMPETITION}_{season}.csv"
        # Finished seasons never change; the current one grows every matchweek.
        if path.exists() and not force and season != current:
            continue
        size = download(season_url(season), path)
        print(f"{path.name}: {size} bytes")

    fixtures = RAW_DIR / "fixtures.csv"
    size = download(f"{BASE_URL}/fixtures.csv", fixtures)
    print(f"{fixtures.name}: {size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Premier League results and odds")
    parser.add_argument("--force", action="store_true", help="re-download finished seasons")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
