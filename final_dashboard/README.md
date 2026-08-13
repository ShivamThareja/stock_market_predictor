# Final phase: Streamlit dashboard

## What's in this folder

| File | Purpose |
|---|---|
| `dashboard.py` | The app — 4 tabs tying every phase together |
| `run_dashboard.sh` | Launches it with the venv + macOS OpenMP workaround already set up |

## What changed from Phase 3

Every phase so far ends with "run this script, read the terminal output."
That's fine for building each piece, but there's no single place to see
today's news, what it's routed to, and what the model thinks happens next
— all at once. This dashboard is that place. It's a **viewer**, not a new
pipeline stage: it reads whatever the last run of each script already
wrote to disk (the SQLite DB, the routed-events CSV, the price CSV, Phase
3's metrics/predictions CSVs) and renders them. The one thing it computes
itself is the **live predictions** panel (below) — everything else is a
read-only view of files other phases produced.

## How to run

```bash
./run_dashboard.sh
```

or manually:

```bash
export DYLD_LIBRARY_PATH="$(pwd)/../venv/lib/python3.9/site-packages/torch/lib"   # see Common errors
source ../venv/bin/activate
pip install streamlit plotly   # if not already installed
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Nothing here re-fetches news or
re-trains the model on its own — use the sidebar's **Refresh data** button
after re-running any earlier phase's script (it just clears Streamlit's
cache and re-reads the files).

## The four tabs

- **Overview** — headline counts, tickers tracked, current model accuracy,
  a NIFTY 50 price chart, and a one-line diagram of how the phases connect.
- **News & Sentiment** — filterable headline table, sentiment distribution,
  sentiment-over-time (stacked by day), filtered by topic/sentiment.
- **Routing** — net sentiment per sector (bar chart, red=bearish/green=bullish),
  most-covered stocks, and a table of the latest routed events.
- **Predictions** — Phase 3's last training run: accuracy vs the
  majority-class baseline, confusion matrix, feature importance, **and a
  live predictions panel**.

## Live predictions — the one thing built fresh here

Phase 3's training script only uses (ticker, day) rows whose next-session
close has already happened — that's what makes it a fair backtest. But
that also means the most recent 1-2 days of news (whatever's too fresh
for the market to have reacted to yet) gets excluded from that CSV
entirely. The dashboard's **Predictions** tab picks those rows back up:
it trains a second model on *all* resolved history (not the time-split
held-out one used for the accuracy numbers above it) and scores today's
in-flight news with it — the closest thing this project has to an actual
forward prediction. Both this model and the evaluated one call the exact
same feature code (`phase3_prediction/feature_engineering.py`), so "how a
prediction gets made" never quietly diverges between the two.

## Color language (consistent across every tab)

- 🟢 green = positive sentiment / predicted UP
- 🔴 red = negative sentiment / predicted DOWN
- ⚪ gray = neutral sentiment / predicted FLAT

Same meaning everywhere it appears on this page. Topics (RBI, US_Fed, etc.)
get their own separate identity palette so they're never confused with
the sentiment/direction colors above.

## Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` / `plotly` | `pip install streamlit plotly` |
| `XGBoostError: Library not loaded: @rpath/libomp.dylib` (macOS) | Use `./run_dashboard.sh`, or manually `export DYLD_LIBRARY_PATH` as shown above — see `phase3_prediction/README.md` for why |
| Tabs show "no data yet" | Run the earlier phase's script first (each tab says which one), then click **Refresh data** |
| Live predictions panel is empty | Nothing's in-flight right now — every routed news day already has a resolved next-session outcome; check back after `week3_pipeline.py` picks up fresh headlines |

## What's next

There isn't a "Phase 4" planned beyond this — the roadmap's four phases
plus this dashboard are the whole project. The real next step is time:
let `phase0_week3/week3_pipeline.py` keep running so more real trading
days accumulate, periodically re-run Phase 2 routing + Phase 3 training,
and watch the accuracy-vs-baseline number on the Predictions tab move
from "roughly baseline" toward something more trustworthy as the dataset
grows past its current ~28 days.
