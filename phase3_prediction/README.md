# Phase 3: XGBoost next-session direction model

## What's in this folder

| File | Purpose |
|---|---|
| `phase3_xgboost_model.py` | Main script — builds the training set and trains/evaluates XGBoost |
| `phase3_daily_features.csv` | Generated — one row per (ticker, day): sentiment + momentum features + label |
| `phase3_test_predictions.csv` | Generated — held-out test rows with actual vs predicted label |

## What changed from Phase 2

Phase 2 produced one row per (headline, affected stock) — too granular to
predict from directly (a stock can get 20 headlines in one day, each with
its own sentiment). Phase 3:

1. **Aggregates** Phase 2's rows down to one row per (ticker, day) —
   headline count, % positive/negative/neutral, average confidence, a
   confidence-weighted net sentiment score.
2. **Labels** each row with what actually happened: the stock's real
   next-trading-session return, pulled from `phase1_close_prices_all.csv`.
   `UP` / `DOWN` / `FLAT` with a ±0.3% deadband (mirrors Phase 0/1's own
   0.5% noise-filter convention — small moves aren't a signal the news
   caused).
3. **Adds two feature families that don't depend on news volume being
   large**, because a few weeks of backfilled headlines is not much data:
   - Price momentum going into the news day (prior 1-day, prior 3-day
     return) — a standard technical signal, available for every row.
   - Same-day NIFTY-index net sentiment, attached to *every* ticker's row
     — captures the project's own "cross-timezone overnight news moves
     the whole market" idea (US Fed / Global Markets headlines route to
     `^NSEI`, not individual stocks) as a feature single-stock rows can use.
4. **Trains XGBoost** (`multi:softprob`, 3 classes) on a **time-based**
   split — train on the earliest ~80% of days, test on the most recent
   ~20%. Not a random shuffle: a random split lets rows from the same day
   leak sentiment patterns between train and test, which a real predictor
   never gets to do (it only ever has the past).

## How to run

```bash
pip install xgboost scikit-learn   # if not already installed
python3 phase3_xgboost_model.py
```

Requires `phase2_routing/phase2_routed_events.csv` (Phase 2),
`phase1_expansion/phase1_close_prices_all.csv` + `phase1_sector_map.csv`
(Phase 1) to already exist.

## What actually happened, honestly

The news database this was first built on had **6 articles from a single
weekend** — nowhere near enough to train anything. Before building Phase 3
for real, I backfilled ~29 days of real headlines (NewsAPI's free-tier
limit is ~1 month back) across the same 6 topics Phase 0/2/3 use, ran
Week 4's FinBERT labeling on all of them (343 articles total), re-ran
Phase 2 routing (973 headline→stock rows), and refreshed Phase 1's price
data through today so next-session returns exist for the new dates.

That produced **230 labeled (ticker, day) rows across 28 days** — the
real dataset this model trains on. Sentiment-only features (headline
count, % pos/neg/neutral, confidence, net sentiment) got:

- **33.3% test accuracy — dead even with the majority-class baseline**
  (always predicting `UP`). No lift at all.

Adding the price-momentum + market-sentiment features moved that to:

- **42.2% test accuracy — a +8.9 point lift over baseline.**

That lift is real (the momentum/market-sentiment features do carry
information a 3-class model can use), but it's measured on **45 test
rows across 6 days** — small enough that this number will swing a lot
run to run as more data comes in. Read it as "the pipeline works
end-to-end and the features aren't pure noise," not as a trustworthy
production accuracy figure. I stopped tuning here on purpose — squeezing
more lift out of a 6-day test slice risks fitting to that specific slice
rather than finding a real pattern.

## Why not push accuracy higher right now

Same conclusion Week 4 reached with FinBERT fine-tuning on 69 examples:
**this is a data-size ceiling, not a modeling problem.** 28 days, most of
it concentrated in a handful of heavily-covered topics (RBI/Banking, US
Fed → whole index), isn't enough to separate real signal from noise for
a 3-class problem. The lever that actually moves this number is time:
let `phase0_week3/week3_pipeline.py` keep running (hourly, as designed)
so real trading days accumulate, then re-run Phase 2 routing and this
script. The time-based split means every re-run tests on genuinely new
days — that's the real test of whether this is learning anything.

## Common errors

| Error | Fix |
|---|---|
| `XGBoostError: Library not loaded: @rpath/libomp.dylib` (macOS) | XGBoost needs the OpenMP runtime, normally installed via `brew install libomp`. Without Homebrew available, this repo's `venv` already has `torch` installed (Week 4), which bundles its own `libomp.dylib` — point XGBoost at it: `export DYLD_LIBRARY_PATH="$(pwd)/../venv/lib/python3.9/site-packages/torch/lib"` before running the script. |
| `Missing ../phase2_routing/phase2_routed_events.csv` | Run `phase2_routing/phase2_news_routing.py` first |
| `Only N rows — below MIN_ROWS_TO_TRAIN` | Not enough accumulated news days yet — let Week 3's pipeline run longer |

## Next: Final phase
Streamlit dashboard tying data collection, sentiment labeling, routing,
and this model together into one view — and a way to watch accuracy
(and the sentiment-only vs sentiment+momentum comparison above) trend
over time as more real days of data come in.
