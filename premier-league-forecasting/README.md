# Premier League Forecasting: Beaten by the Closing Line

Poisson family goal models, Elo and pi-ratings forecasting Premier League matches, scored against the price the bookmakers settle on just before kick-off. The headline result is negative, and that is the finding: over three held-out seasons no model here beats the closing line, the best blend of model and market puts zero weight on the model, and betting the model would have lost money.

## Data

Results and odds come from [football-data.co.uk](https://football-data.co.uk/englandm.php), competition E0, seasons 2019-20 through 2026-27: 2,690 matches. Each row carries the score, half-time score, shots, shots on target, fouls, corners and cards, plus opening and closing prices from several bookmakers for the match result, over/under 2.5 goals and the Asian handicap. Expected goals appear in the feed only from 2026-27, so they are loaded but play no part in the models below.

Two practical notes for anyone re-running this. The bare host serves the CSV files directly while the `www` host answers a 302, so the fetch script uses the bare host. And pandas 3 returns read-only arrays from `to_numpy()`, which the Cython loss functions inside penaltyblog reject, so every array handed to a model is an explicit writable copy.

Three sources that older tutorials still recommend were checked and are not usable as of September 2026: the Club Elo API returns 502 and its fixtures endpoint is switched off, FBref returns 403, and Understat no longer embeds season data in its league pages. Elo and pi-ratings are therefore computed here from the match results themselves.

## Method

Every forecast is produced walk-forward: for each matchday, the model is refit on matches that kicked off strictly before it, then asked about that day's fixtures. A test in `tests/test_backtest.py` fails if any training window reaches into the day being predicted.

Newly promoted sides have no history in this division. A team with fewer than ten prior matches is priced as a shared generic promoted side, which also covers the case of a team sitting exactly on that boundary, whose own name never reaches the fit.

The time decay on the likelihood is tuned on the validation seasons only, by log loss:

| xi | RPS | Log loss |
|---|---|---|
| 0 | 0.20674 | 0.99314 |
| 0.001 | 0.20500 | 0.98882 |
| 0.0018 | 0.20406 | 0.98668 |
| **0.003** | **0.20349** | **0.98605** |
| 0.005 | 0.20402 | 0.98910 |

Seasons 2020-21 to 2022-23 are the validation set, 2023-24 to 2025-26 the test set. Nothing is chosen on the test set.

## The bar: what the market already knows

Bookmaker odds carry a margin, so they are not probabilities until it is removed. Six removal methods were compared over 2,280 matches; Shin's method won on log loss by a hair, and the choice barely matters.

| Line | RPS | Log loss | Brier | Top pick correct |
|---|---|---|---|---|
| Closing | 0.19644 | 0.96219 | 0.57111 | 55.3% |
| Opening | 0.19798 | 0.96686 | 0.57446 | 54.7% |

The closing line beats the opening line on every metric, which is the expected result: prices absorb information as money arrives. The mean margin on the closing line is 4.30%.

The closing line is also close to perfectly calibrated. Grouped into ten buckets, what it says happens about as often as it says:

| Stated | Observed | Forecasts |
|---|---|---|
| 0.07 | 0.06 | 297 |
| 0.16 | 0.17 | 1,150 |
| 0.25 | 0.25 | 2,514 |
| 0.34 | 0.35 | 912 |
| 0.45 | 0.43 | 702 |
| 0.55 | 0.56 | 541 |
| 0.65 | 0.63 | 368 |
| 0.74 | 0.74 | 241 |
| 0.84 | 0.86 | 110 |

The one visible wobble is the 0.45 bucket, where favourites priced just under even money won a little less often than the price implied, and the sparse top bucket, where the biggest favourites slightly overperformed. Neither gap is large enough to bet into once the 4.30% margin is paid.

## Walk-forward results

Six forecasters, 1,140 test matches, every one judged on exactly the same fixtures.

| Forecaster | RPS | vs market | Log loss | Brier | Top pick correct |
|---|---|---|---|---|---|
| **Market, closing line** | **0.19375** | | 0.95938 | 0.56988 | 55.0% |
| Dixon-Coles | 0.20162 | +0.00787 | 0.98437 | 0.58698 | 51.8% |
| Poisson | 0.20170 | +0.00794 | 0.98523 | 0.58751 | 51.8% |
| Bivariate Poisson | 0.20171 | +0.00796 | 0.98493 | 0.58743 | 51.9% |
| Elo | 0.20504 | +0.01129 | 1.01088 | 0.59918 | 52.1% |
| Pi-ratings | 0.20974 | +0.01599 | 1.04057 | 0.60899 | 48.9% |

The model results sit where the literature says they should. A gap of roughly 0.008 RPS behind the closing line is what a competently fitted goal model looks like; the three Poisson variants are indistinguishable from one another, and the rating systems, which only see results or goal margins rather than a scoring process, trail them.

## Does the model find value the market missed?

The two forecasts were combined by logarithmic pooling, with the weight chosen on the validation seasons. Against both the opening and the closing line, the chosen weight is **zero**: every gram of model makes the combined forecast worse, monotonically.

| Weight on the model | RPS | Log loss |
|---|---|---|
| **0.0** | **0.19310** | **0.94894** |
| 0.3 | 0.19494 | 0.95502 |
| 0.6 | 0.19796 | 0.96559 |
| 1.0 | 0.20349 | 0.98605 |

That answers the question directly. The model carries no information about these matches that the price does not already contain. With the weight at zero the best available forecast is the market itself, and a market's own probabilities can never clear a positive edge against its own prices, so no selection qualifies as a bet against either line.

To make the negative result concrete rather than vacuous, one configuration was fixed in advance: bet the raw model, with no market input, into the opening price, at the edge threshold the validation seasons preferred.

| | |
|---|---|
| Bets | 1,106 |
| Staked | 8,474 units |
| Profit | -802 units |
| ROI | **-9.46%** |
| 95% interval on ROI | [-21.31%, +2.54%] |
| Maximum drawdown | 88% |

Quarter Kelly, stakes capped at 2% of bankroll, settled matchday by matchday, resampled by matchday for the interval. The point estimate is a heavy loss; the interval still touches zero, which is the honest way to say that even 1,106 bets over three seasons is not many. Either way the verdict is the same, and `analysis/predict_next.py` enforces it: until a study shows a positive lower bound, every suggested stake is held at zero.

### A trap worth documenting

An earlier version of this study tuned the blend, landed on weight zero, and then bet those probabilities into the opening price. It reported a 15.8% ROI with a 95% interval of [2.7%, 29.2%] and declared a demonstrable edge. That number is an artifact. With the weight at zero the forecast is the closing line, so the strategy amounted to betting the closing price into the opening price: a bet on line movement, placed with the movement already known. Each book now blends and settles against the same snapshot, so no book can see a price that would not exist yet when the bet is struck.

## What this means

The Premier League match-result market is efficient enough that a well-specified goal model, fitted carefully and validated without leakage, adds nothing to it. That is a real answer to a real question, and it is worth more than a tuned number that would not survive contact with a bookmaker.

Where the same machinery could still earn its keep: markets with less money and less attention, such as lower divisions, and secondary markets like totals and the Asian handicap, which this codebase already prices through the same probability grid but which are not studied here.

## The language model panel

Language models already know how past seasons ended, so scoring them on history measures memory rather than forecasting. The panel in `src/llm_panel.py` is therefore only ever asked about fixtures that have not kicked off, and `record()` refuses to store anything else. It asks each free model on OpenRouter for all of a matchday's fixtures in a single request, which keeps the whole panel inside the free tier, reads the roster live because it rotates, and tolerates the usual failures: prose wrapped around the JSON, probabilities that do not quite sum to one, a club named "Manchester United" where the data says "Man United".

It needs a key:

```bash
export OPENROUTER_API_KEY=...    # free, no card required, at openrouter.ai/keys
```

Without one, the weekly routine skips the panel and carries on.

## Contents

- `data/fetch_football_data.py`: downloads every season plus the current fixture list, re-fetching only what can still change
- `src/loader.py`: one frame for played matches and upcoming fixtures alike, so forecasters need no special case
- `src/market.py`: implied probabilities with the margin removed, and the bookmakers wrapped as a forecaster
- `src/models.py`: the penaltyblog goal models behind one interface, with promoted sides handled
- `src/ratings.py`: Elo and pi-ratings as forecasters of the same shape
- `src/backtest.py`: the walk-forward harness, which refuses to leak
- `src/betting.py`: pooling, edge, Kelly staking, bankroll simulation and a matchday bootstrap
- `src/metrics.py`: RPS, log loss, Brier, ignorance and calibration
- `src/llm_panel.py`: the OpenRouter panel, recorded before kick-off
- `analysis/`: the studies above, the weekly forecast, and the report
- `tests/`: 117 tests, none of which touch the network

## Running it

```bash
uv venv && uv pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m data.fetch_football_data
.venv/bin/python -m analysis.run_baseline     # what the market scores
.venv/bin/python -m analysis.run_backtest     # every forecaster, walk-forward
.venv/bin/python -m analysis.run_betting      # is there any value to take
.venv/bin/python -m analysis.predict_next     # the coming matchweek
.venv/bin/python -m analysis.build_report     # the page in results/report.html
```

`analysis/weekly.sh` chains the last steps for a cron entry. The backtest takes about twelve minutes; everything else runs in seconds.
