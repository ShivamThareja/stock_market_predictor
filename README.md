# Stock Market Predictor

A global financial news to Indian stock market predictor.

Reads financial news from around the world, uses AI to understand
sentiment, and predicts how Indian stocks (NIFTY50) will react -
both from same-timezone Indian news (RBI, NSE announcements) and
cross-timezone overnight news (US Fed, Nikkei, global markets) -
before NSE opens each morning.

## Project status
All phases complete: Phase 0, Phase 1, Week 3 (automated pipeline),
Week 4 (FinBERT sentiment labeling), Phase 2 (news-to-stock routing),
Phase 3 (XGBoost prediction model), and the final Streamlit dashboard
tying it all together.

## Roadmap
- **Phase 0 (done)** - learn the fundamentals: stock data (OHLCV,
  returns, correlations) and financial news fetching/labeling
- **Phase 1 (done)** - expand from 6 tickers to all 56 NIFTY50
  constituents + 5 sector indices, with a ticker-to-sector map
  that later phases use to auto-route news to affected stocks
- **Week 3 (done)** - automated news fetching into an hourly
  pipeline using APScheduler + SQLite instead of manual runs
- **Week 4 (done)** - FinBERT for automated sentiment scoring,
  validated against hand-labeled data and iteratively corrected;
  fine-tuning attempted and found to need more data to help
- **Phase 2 (done)** - route labeled headlines to affected
  stocks/sectors using the sector map, via direct company-name
  matching plus topic-based macro routing (RBI -> Banking sector,
  US Fed -> whole index, etc.)
- **Phase 3 (done)** - XGBoost model predicting next-session
  direction (UP/DOWN/FLAT) per stock from aggregated daily
  sentiment + price momentum + market-wide sentiment. Time-based
  train/test split. Sentiment-only features scored no better than
  a majority-class baseline (33.3%); adding price momentum and
  same-day index sentiment lifted it to 42.2% (+8.9 pts) on a
  small (28-day) backfilled dataset. Documented as a data-size
  ceiling, not a modeling problem - see `phase3_prediction/README.md`
- **Final (done)** - Streamlit dashboard tying data collection,
  sentiment, routing, and predictions into one view, plus a live
  in-flight predictions panel for news too recent to have a
  resolved outcome yet - see `final_dashboard/README.md`

## Tech stack
Python, yfinance, pandas, matplotlib, seaborn, NewsAPI, FinBERT,
APScheduler, SQLite, XGBoost, Streamlit, Plotly

## Structure
- `phase0_week1/` - Stock data exploration, OHLCV, correlations
- `phase0_week2/` - Financial news fetching and labeling
- `phase0_week3/` - Automated hourly news pipeline (SQLite + APScheduler)
- `phase0_week4/` - FinBERT sentiment labeling, validation, fine-tuning
- `phase1_expansion/` - All 56 NIFTY50 tickers + sector indices,
  sector-level correlation, ticker-to-sector lookup for later phases
- `phase2_routing/` - Routes labeled headlines to affected stocks/sectors
- `phase3_prediction/` - XGBoost model predicting next-session stock
  direction from daily sentiment + price momentum features
- `final_dashboard/` - Streamlit dashboard tying every phase together,
  with a live in-flight predictions panel

## Running the dashboard
```bash
cd final_dashboard
./run_dashboard.sh
```
See `final_dashboard/README.md` for what each tab shows and the
macOS OpenMP workaround the script sets up automatically.

Built incrementally, one phase at a time, by a BTech CSE student
learning finance, ML and NLP through this project.
