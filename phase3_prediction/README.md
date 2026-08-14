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
3. **Adds feature families that don't depend on news volume being
   large**, because a few weeks of backfilled headlines is not much data:
   - Price momentum going into the news day (prior 1-day, prior 3-day
     return) — a standard technical signal, available for every row.
   - Same-day NIFTY-index net sentiment, attached to *every* ticker's row
     — captures the project's own "cross-timezone overnight news moves
     the whole market" idea (US Fed / Global Markets headlines route to
     `^NSEI`, not individual stocks) as a feature single-stock rows can use.
   - Market-wide macro momentum (`feature_engineering.py`'s
     `MARKET_SERIES`): NIFTY's own prior-day return, S&P 500 prior-day
     return (the literal overnight US->India signal), crude oil, USD/INR,
     and India VIX prior-day change — fetched by
     `phase1_expansion/phase1_macro_data.py`, attached to every row the
     same way market sentiment is. Plus day-of-week.
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
for real, backfilled ~29 days of real headlines (NewsAPI's free-tier
limit is ~1 month back) across the same 6 topics Phase 0/2/3 use, ran
Week 4's FinBERT labeling on all of them, ran Phase 2 routing, and
refreshed Phase 1's price data so next-session returns exist for the
new dates.

**Round 1** (sentiment + momentum features only, 230 labeled rows / 28
days): sentiment-only features scored 33.3% — dead even with the
majority-class baseline. Adding price momentum + same-day index
sentiment moved that to **42.2% (+8.9 pts over baseline)**.

**Round 2** — a later push specifically to improve accuracy added three
things: `week4_label_rules.py`'s targeted sentiment corrections (24 of
386 headlines fixed — RBI hold/hike/cut and rally/fall language FinBERT
was getting backwards), the macro/momentum features described above, and
more accumulated news days (29 by this point). Result on **244 labeled
rows across 29 days**:

- **43.5% test accuracy — a +10.9 point lift over the 32.6% baseline**,
  on 46 test rows across 6 days.

Also attempted: fine-tuning FinBERT on the larger, rule-corrected
dataset (386 examples, well past the 200+ threshold Week 4 originally
flagged as needed). Evaluated honestly against the untouched Week 2
human labels (not a self-referential check against FinBERT's own prior
output) — **it tied zero-shot exactly, 69.6% vs 69.6%.** No improvement.
Full writeup in `phase0_week4/README.md`.

**Net movement: 42.2% → 43.5%, a real but modest +1.3 point gain** from
meaningfully more engineering effort (label corrections + 5 new macro
features). Both numbers are measured on ~45 test rows across ~6 days —
close enough together that some of this difference could just be
different test-slice noise rather than a genuine improvement. Neither
number should be read as a stable, production-grade figure yet.

## Why 55%+ isn't happening yet, and what would actually get there

Same conclusion Week 4 reached with FinBERT fine-tuning, now confirmed a
second time on the fine-tuning retry above: **this is a data-size
ceiling, not a modeling or feature-selection problem.** 29 days,
concentrated in a handful of heavily-covered topics (RBI/Banking, US Fed
→ whole index), just isn't enough to separate real signal from noise for
a 3-class problem — no amount of additional features or label-cleanup
effort changes that ceiling much, which is exactly what Round 2 shows:
real, legitimate improvements (rule-corrected labels, 5 new macro
features) moved the number by scarcely more than a point. The lever that
actually moves this is time: let `phase0_week3/week3_pipeline.py` keep
running (hourly cron, see root README) so real trading days accumulate
across varied market conditions, then periodically re-run Phase 2
routing and this script. The time-based split means every re-run tests
on genuinely new days — that's the real test of whether this is learning
anything, and it's the only lever left that's actually likely to close
a meaningful chunk of the gap to 55%.

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
