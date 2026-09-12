"""Render the weekly report as one self-contained page.

Published pages cannot fetch anything at runtime, so every number is baked in
at build time.
"""

import html
import json
import pathlib
from string import Template

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
REPORT_PATH = RESULTS_DIR / "report.html"

PAGE = Template("""<title>Closing Line Report</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600&display=swap">
<style>
  :root {
    --ground: #f6f7f4;
    --surface: #ffffff;
    --edge: #d8dcd4;
    --ink: #14170f;
    --muted: #5d6356;
    --accent: #0e6b63;
    --accent-soft: #d7e8e5;
    --good: #2f8f5b;
    --warn: #b7791f;
    --critical: #b3402f;
    --home: #0e6b63;
    --draw: #9aa392;
    --away: #b3402f;
  }
  :root:not([data-theme="light"]) {
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #101310;
      --surface: #181c17;
      --edge: #2b3129;
      --ink: #edefea;
      --muted: #9aa392;
      --accent: #4fbfae;
      --accent-soft: #1d3330;
      --good: #5cc189;
      --warn: #d9a441;
      --critical: #e0705c;
      --home: #4fbfae;
      --draw: #7d8878;
      --away: #e0705c;
    }
  }
  :root[data-theme="dark"] {
    --ground: #101310;
    --surface: #181c17;
    --edge: #2b3129;
    --ink: #edefea;
    --muted: #9aa392;
    --accent: #4fbfae;
    --accent-soft: #1d3330;
    --good: #5cc189;
    --warn: #d9a441;
    --critical: #e0705c;
    --home: #4fbfae;
    --draw: #7d8878;
    --away: #e0705c;
  }

  body {
    background: var(--ground);
    color: var(--ink);
    font-family: "Public Sans", ui-sans-serif, system-ui, sans-serif;
    line-height: 1.5;
  }
  .page {
    max-width: 1080px;
    margin: 0 auto;
    padding-block: 40px 72px;
    padding-left: 20px;
    padding-right: 20px;
    display: flex;
    flex-direction: column;
    gap: 40px;
  }
  h1, h2 {
    font-family: "Bricolage Grotesque", ui-sans-serif, system-ui, sans-serif;
    text-wrap: balance;
    margin: 0;
  }
  h1 { font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 800; letter-spacing: -0.02em; }
  h2 { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }
  p { margin: 0; max-width: 66ch; }
  .lede { color: var(--muted); }
  .eyebrow {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.74rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }
  section { display: flex; flex-direction: column; gap: 16px; }

  .verdict {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px 16px;
    padding: 20px 24px;
    border-left: 4px solid var(--critical);
    background: var(--surface);
  }
  .verdict.is-positive { border-left-color: var(--good); }
  .verdict strong { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.35rem; }
  .verdict .detail { color: var(--muted); font-size: 0.95rem; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px; background: var(--edge); border: 1px solid var(--edge); }
  .tile { background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
  .tile .value { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 500; }
  .tile .note { font-size: 0.8rem; color: var(--muted); }
  .delta-good { color: var(--good); }
  .delta-bad { color: var(--critical); }

  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--edge); white-space: nowrap; }
  th { font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
  td.num { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; text-align: right; }
  tbody tr:last-child td { border-bottom: none; }
  .leader td:first-child { font-weight: 600; }
  .market-row { background: var(--accent-soft); }

  .split { width: 190px; }
  .bar { display: flex; height: 9px; overflow: hidden; }
  .bar span { display: block; height: 100%; }
  .bar .h { background: var(--home); }
  .bar .d { background: var(--draw); }
  .bar .a { background: var(--away); }
  .ghost { display: flex; height: 3px; margin-top: 2px; opacity: 0.45; }
  .ghost span { display: block; height: 100%; }
  .ghost .h { background: var(--home); }
  .ghost .d { background: var(--draw); }
  .ghost .a { background: var(--away); }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.8rem; color: var(--muted); }
  .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: middle; }

  .chart { background: var(--surface); border: 1px solid var(--edge); padding: 16px; }
  svg { display: block; width: 100%; height: auto; max-width: 100%; }
  .axis { stroke: var(--edge); stroke-width: 1; fill: none; }
  .ideal { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; fill: none; }
  .tick { fill: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 10px; }
  .point { fill: var(--accent); }
  .trace { stroke: var(--accent); stroke-width: 2; fill: none; }

  footer { color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--edge); padding-top: 16px; }
  code { font-family: "IBM Plex Mono", monospace; font-size: 0.85em; }
  @media (max-width: 560px) { .split { width: 130px; } }
</style>

<div class="page">
  <header>
    <p class="eyebrow">Premier League $season &middot; built $generated</p>
    <h1>Closing Line Report</h1>
    <p class="lede">Every forecast here is scored against the bookmakers' closing price, the hardest benchmark in football. $subtitle</p>
  </header>

  <div class="verdict $verdict_class">
    <strong>$verdict</strong>
    <span class="detail">$verdict_detail</span>
  </div>

  <section>
    <p class="eyebrow">Where the system stands</p>
    <div class="tiles">$tiles</div>
  </section>

  <section>
    <h2>This matchweek</h2>
    <p class="lede">The blended forecast, with the market's implied split as the thin line underneath each bar.</p>
    <div class="legend">
      <span><i class="swatch" style="background: var(--home)"></i>Home</span>
      <span><i class="swatch" style="background: var(--draw)"></i>Draw</span>
      <span><i class="swatch" style="background: var(--away)"></i>Away</span>
    </div>
    <div class="scroll">$matchweek</div>
  </section>

  <section>
    <h2>Walk-forward standings</h2>
    <p class="lede">$standings_note</p>
    <div class="scroll">$standings</div>
  </section>

  <section>
    <h2>Is the closing line calibrated?</h2>
    <p class="lede">Each point is a bucket of forecasts. On the dashed line, the market's stated probability matched how often those results actually happened.</p>
    <div class="chart">$calibration</div>
  </section>

  <footer>
    <p>Sources: football-data.co.uk results and odds, $matches matches. Model: $model with time decay xi=$xi, blended with the market at weight $weight. Stakes are quarter Kelly, capped at 2% of bankroll, and held at zero until a backtest shows a positive lower bound on ROI.</p>
  </footer>
</div>
""")


def read_json(name: str) -> dict:
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def latest_matchweek() -> dict:
    paths = sorted(RESULTS_DIR.glob("matchweek_*.json"))
    return json.loads(paths[-1].read_text()) if paths else {}


def tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
    return f'<div class="tile"><span class="eyebrow">{html.escape(label)}</span><span class="value">{value}</span>{note_html}</div>'


def bar(probabilities, klass: str = "bar") -> str:
    home, draw, away = (max(float(value), 0.0) for value in probabilities)
    total = home + draw + away or 1.0
    widths = [100 * home / total, 100 * draw / total, 100 * away / total]
    segments = "".join(f'<span class="{key}" style="width: {width:.2f}%"></span>' for key, width in zip("hda", widths))
    return f'<div class="{klass}">{segments}</div>'


def matchweek_table(matchweek: dict) -> str:
    fixtures = matchweek.get("fixtures", [])
    if not fixtures:
        return '<p class="lede">No fixture published yet for the coming matchweek.</p>'
    rows = []
    for fixture in fixtures:
        probabilities = [fixture["p_home"], fixture["p_draw"], fixture["p_away"]]
        market = [fixture.get(f"{key}_market") for key in ("p_home", "p_draw", "p_away")]
        ghost = bar(market, klass="ghost") if all(value is not None for value in market) else ""
        kickoff = str(fixture.get("Datetime", ""))[:16].replace("T", " ")
        rows.append(
            f'<tr><td>{html.escape(kickoff)}</td>'
            f'<td>{html.escape(str(fixture["HomeTeam"]))}</td>'
            f'<td>{html.escape(str(fixture["AwayTeam"]))}</td>'
            f'<td class="split">{bar(probabilities)}{ghost}</td>'
            + "".join(f'<td class="num">{value:.0%}</td>' for value in probabilities)
            + "</tr>"
        )
    header = "<tr><th>Kick-off</th><th>Home</th><th>Away</th><th>Forecast</th><th>H</th><th>D</th><th>A</th></tr>"
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def standings_table(backtest: dict) -> str:
    results = backtest.get("results", {})
    if not results:
        return '<p class="lede">Run the walk-forward backtest to populate this table.</p>'
    ordered = sorted(results.items(), key=lambda item: item[1]["rps"])
    market = results.get("market_closing", {}).get("rps")
    rows = []
    for name, scores in ordered:
        gap = "" if market is None else f"{scores['rps'] - market:+.5f}"
        klass = ' class="market-row"' if name == "market_closing" else ""
        rows.append(
            f"<tr{klass}><td>{html.escape(name.replace('_', ' '))}</td>"
            f'<td class="num">{scores["rps"]:.5f}</td>'
            f'<td class="num">{gap}</td>'
            f'<td class="num">{scores["log_loss"]:.5f}</td>'
            f'<td class="num">{scores["brier"]:.5f}</td>'
            f'<td class="num">{scores["accuracy"]:.1%}</td></tr>'
        )
    header = "<tr><th>Forecaster</th><th>RPS</th><th>vs market</th><th>Log loss</th><th>Brier</th><th>Top pick</th></tr>"
    return f'<table class="leader"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def calibration_chart(baseline: dict) -> str:
    table = baseline.get("calibration", [])
    if not table:
        return '<p class="lede">Run the baseline to populate this chart.</p>'
    left, top, size = 46, 16, 300
    def place(value):
        return left + value * size, top + (1 - value) * size
    ticks = []
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        x, y = place(value)
        ticks.append(f'<text class="tick" x="{left - 8:.0f}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}</text>')
        ticks.append(f'<text class="tick" x="{x:.1f}" y="{top + size + 20:.0f}" text-anchor="middle">{value:.2f}</text>')
    points, path = [], []
    for bucket in table:
        x, y = place(bucket["predicted"])
        _, observed_y = place(bucket["observed"])
        radius = 3 + 5 * (bucket["n"] / max(row["n"] for row in table)) ** 0.5
        points.append(f'<circle class="point" cx="{x:.1f}" cy="{observed_y:.1f}" r="{radius:.1f}"></circle>')
        path.append(f"{x:.1f},{observed_y:.1f}")
    start_x, start_y = place(0.0)
    end_x, end_y = place(1.0)
    return (
        f'<svg viewBox="0 0 {left + size + 30} {top + size + 52}" role="img" aria-label="Calibration of the closing line">'
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" class="axis"></rect>'
        f'<line class="ideal" x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}"></line>'
        f'<polyline class="trace" points="{" ".join(path)}"></polyline>'
        f'{"".join(points)}{"".join(ticks)}'
        f'<text class="tick" x="{left + size / 2:.0f}" y="{top + size + 40:.0f}" text-anchor="middle">Stated probability</text>'
        f'<text class="tick" x="14" y="{top + size / 2:.0f}" text-anchor="middle" transform="rotate(-90 14 {top + size / 2:.0f})">Observed frequency</text>'
        "</svg>"
    )


def build() -> str:
    baseline, backtest, betting = read_json("baseline.json"), read_json("backtest.json"), read_json("betting.json")
    matchweek = latest_matchweek()
    settings = matchweek.get("settings", {})

    book = betting.get("books", {}).get("opening", {})
    accuracy = betting.get("accuracy", {})
    demonstrable = bool(book.get("demonstrable_edge"))
    if not betting:
        verdict, detail = "Not yet tested", "Run the betting study to decide whether any of this is worth staking."
    elif demonstrable:
        verdict = "Edge demonstrable"
        detail = f"ROI {book['roi']:.1%} over {book['bets']} bets, 95% interval [{book['ci_low']:.1%}, {book['ci_high']:.1%}]."
    else:
        verdict = "No demonstrable edge"
        detail = "The confidence interval on ROI crosses zero, so stakes stay at zero and these forecasts are information, not advice."

    tiles = []
    market_rps = accuracy.get("market", {}).get("rps") or backtest.get("results", {}).get("market_closing", {}).get("rps")
    if market_rps:
        tiles.append(tile("Market RPS", f"{market_rps:.4f}", "the bar to beat"))
    for label, key in [("Model RPS", "model"), ("Blend RPS", "blend")]:
        scores = accuracy.get(key)
        if scores:
            gap = scores["rps"] - market_rps if market_rps else None
            note = "" if gap is None else f"{gap:+.4f} vs market"
            tiles.append(tile(label, f'<span class="{"delta-good" if gap and gap < 0 else "delta-bad"}">{scores["rps"]:.4f}</span>', note))
    if book:
        tiles.append(tile("Simulated ROI", f"{book['roi']:.1%}" if book.get("bets") else "no bets", f"{book.get('bets', 0)} bets into the opening line"))
    if baseline.get("closing_margin_mean"):
        tiles.append(tile("Bookmaker margin", f"{baseline['closing_margin_mean']:.2%}", "closing line overround"))
    if not tiles:
        tiles.append(tile("Status", "empty", "no results yet"))

    return PAGE.substitute(
        season=matchweek.get("fixtures", [{}])[0].get("Season", "2026-27") if matchweek.get("fixtures") else "2026-27",
        generated=str(matchweek.get("generated_at", ""))[:10] or "pending",
        subtitle="A forecast that cannot beat that price is not a signal, and the page says so.",
        verdict=html.escape(verdict),
        verdict_detail=html.escape(detail),
        verdict_class="is-positive" if demonstrable else "",
        tiles="".join(tiles),
        matchweek=matchweek_table(matchweek),
        standings=standings_table(backtest),
        standings_note=(
            f"Walk-forward over {', '.join(backtest.get('test_seasons', []))}, {backtest.get('matches', 0)} matches, "
            "every forecaster judged on exactly the same fixtures."
        ) if backtest else "Not run yet.",
        calibration=calibration_chart(baseline),
        matches=backtest.get("matches", baseline.get("matches", 0)),
        model=html.escape(str(settings.get("model", "Dixon-Coles"))),
        xi=settings.get("xi", backtest.get("xi", "unset")),
        weight=settings.get("weight", betting.get("weight", "unset")),
    )


def main() -> pathlib.Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build())
    print(f"Wrote {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    main()
