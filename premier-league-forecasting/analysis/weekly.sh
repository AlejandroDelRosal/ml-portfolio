#!/usr/bin/env bash
# Weekly routine: refresh results and prices, forecast the coming matchweek,
# ask the model panel if a key is configured, and rebuild the report.
#
# crontab -e
#   30 9 * * THU cd ~/repos/ml-portfolio/premier-league-forecasting && ./analysis/weekly.sh >> results/weekly.log 2>&1
#
# The backtest is not part of this routine: it takes minutes and its answers
# only move when a season's worth of new matches has landed.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python

"$PYTHON" -m data.fetch_football_data
"$PYTHON" -m analysis.predict_next

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    "$PYTHON" -m analysis.run_panel
else
    echo "OPENROUTER_API_KEY not set, skipping the model panel"
fi

"$PYTHON" -m analysis.build_report
