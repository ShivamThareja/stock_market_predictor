# Phase 4: Self-learning engine

## What's in this folder

| File | Purpose |
|---|---|
| `schema.py` | Creates the 6 new tables in `phase0_week3/news_pipeline.db` (idempotent) |
| `signals.py` | Shared logic: signal weighting, model persistence, SHAP-style contribution/error attribution, TF-IDF similar-headline search, source credibility, retrain-trigger check |
| `prediction_logger.py` | Morning (7 AM IST): predicts NIFTY's next-session direction, logs full reasoning |
| `learning_engine.py` | Afternoon (4 PM IST): records the actual outcome, attributes errors, adjusts signal/source weights, retrains if triggered, generates tomorrow's watchlist |
| `scheduler.py` | APScheduler orchestrator — runs the whole hourly/daily cadence as one long-lived process |
| `../run_system.sh` | One command to start the scheduler detached (survives closing the terminal) |
| `model_state.pkl`, `learning_state.json` | Generated — saved model + retrain bookkeeping (the .pkl is gitignored, regenerable) |

## Correction from the original spec, made before writing anything

The spec said to add tables to "the existing news.db." This project's real,
populated database is `phase0_week3/news_pipeline.db` (386+ headlines,
everything Phase 0–3 read/write). There's a separate `phase0_week3/news.db`
sitting in that folder — empty (0 bytes), untracked, gitignored last session
specifically because nothing reads or writes it. Every script here points at
`news_pipeline.db`. Building this on `news.db` would have produced a system
with its own tables that no other part of the project — or reality — ever
touches.

## Two technical adaptations, and why

**"Signal weights" don't work the way the spec assumes, so they're
implemented differently than "reduce a signal's weight by 10%" literally
suggests.** Multiplying a feature *column* by a constant before training is a
no-op for tree models — XGBoost splits on value *order*, not magnitude, so
scaling "sentiment" by 0.5 wouldn't move a single split in the tree. Verified
this before building on it (see `signals.py`'s docstring for the test).
What actually works: XGBoost's own `feature_weights` fit parameter, which
biases which features get *sampled* as split candidates under column
sub-sampling (the model already uses `colsample_bytree=0.8`, so sampling is
active and this has a real, verified effect). That's the real mechanism
`signal_weights.current_weight` controls — via `feature_weights_array()` and
`train_weighted_model()` in `signals.py`.

**Source credibility attribution is coarse, and says so.** "Did headlines
from source X contribute to correct vs wrong predictions" implies per-source
causal attribution, which isn't really extractable from a same-day-aggregate
model (a day's NIFTY prediction blends dozens of headlines from many sources
into one number). What's implemented: every source with at least one
headline routed to NIFTY that day shares equal credit or blame for that
day's single outcome. Same "reasonable first pass, not precision attribution"
caveat Phase 2's company-name matching already carries — documented in
`signals.update_source_credibility()`, not hidden.

**Error attribution reuses the morning's own reasoning, not a re-guess.**
When a prediction turns out wrong, `main_error_signal` is read back from
that morning's `top_signals` (the SHAP-style per-signal contribution toward
the predicted class, computed via XGBoost's `pred_contribs`) — the actual
signal that drove the actual wrong call, not a fresh computation with
whatever model happens to exist at 4 PM (which could differ if a retrain
fired in between). If no historical headline resembled today's news closely
(max TF-IDF similarity < 0.15), the reason is `no_precedent` instead —
directly answering the spec's "was there news with no historical
precedent?" question with actual data rather than a coin flip.

## One scope decision: NIFTY-level, not per-stock

The spec's Step 1 says "pull the actual NIFTY close" and compares it to
"this morning's logged prediction" (singular) — so this system predicts
**one thing per day: NIFTY 50's next-session direction**, not Phase 3's
per-ticker predictions. It reuses Phase 3's exact feature-engineering
(`feature_engineering.build_dataset`, unmodified, imported read-only) filtered
to the `^NSEI` row, so the two systems share a data pipeline but serve
different purposes: Phase 3 is the offline, evaluated benchmark; this is the
live, continuously-adjusted, single-symbol production loop.

## One gap surfaced, not papered over

The forward-chain rule "crude oil up >2% → flag aviation stocks" has no
target: **this project doesn't track any aviation/airline tickers** — the
NIFTY50 universe in `phase1_sector_map.csv` has 10 sectors and none of them
is aviation (no IndiGo/SpiceJet-equivalent in this dataset). Rather than
silently drop the rule or attach it to an unrelated sector, it still logs to
`forward_signals` with `sector="Aviation"`, `stock=NULL`, and a reasoning
string that says exactly this — visible in the data instead of quietly
missing.

## How the daily cycle actually works

**Morning (`prediction_logger.py`):** first re-runs the existing hourly
pipeline once (fetch → label → rule-correct → route) so the prediction uses
the freshest possible overnight news even if the hourly schedule hasn't
fired yet this hour — then builds today's NIFTY feature row, loads (or
trains, if none saved yet) the current weighted model, predicts, computes
SHAP-style top signals, pulls the 5 most sentiment-weighted headlines behind
today's routed events, finds the 3 most similar historical headlines via
TF-IDF (and what NIFTY actually did the session after each one appeared),
and logs all of it to `prediction_log` — one row per calendar day.

**Afternoon (`learning_engine.py`):** for every unresolved `prediction_log`
row, fetches NIFTY's real close (direct yfinance pull, not the full
56-ticker `phase1_nifty50.py` refresh — that script is untouched and a
1-2 minute chunked download would be wasteful just to check one number),
compares to the prediction, and:
- Updates the signal's TRUE win/loss streak. 3 wrong in a row → halve its
  weight immediately (`prediction_errors.decay_triggered=1`); 2 correct in a
  row while decayed → restore to the pre-decay weight.
- Separately, tracks a rolling "since last adjustment" tally per signal
  (deliberately a different counter from the streak above, or the two rules
  would interfere): every 5th wrong attribution → -10%; every 5th correct →
  +10%. Both clamp to [0.2×, 2.0×].
- Updates `source_credibility` for every source behind that day's routed
  NIFTY headlines.
- Checks the retrain trigger (50+ new labeled headlines since last retrain,
  cumulative signal-weight drift >20%, or <38% accuracy over the last 10
  predictions) and retrains if any fired — logging old vs. new accuracy to
  `learning_state.json`'s `retrain_history`.
- Generates tomorrow's `forward_signals` from today's actual crude
  oil/S&P 500/gold/USD-INR moves and whether RBI news broke today.

**Every hour, independently of the above:** the existing pipeline runs
(fetch → label → rule-correct → route), followed by a lightweight
retrain-trigger check (`learning_engine.py --retrain-check-only`) — this is
the same Step 6 logic, just also invoked hourly so a 50-headline threshold
crossing doesn't have to wait until 4 PM to act.

## How to run

```bash
./run_system.sh          # from the repo root — starts the scheduler detached
tail -f phase4_learning/logs/scheduler.log
```

To stop: `kill $(cat phase4_learning/scheduler.pid)`.

To test any single step manually without waiting for its scheduled time:
```bash
export DYLD_LIBRARY_PATH="$(pwd)/../venv/lib/python3.9/site-packages/torch/lib"   # macOS libomp workaround
python3 prediction_logger.py
python3 learning_engine.py
python3 learning_engine.py --retrain-check-only
```

## What's actually been verified vs. what's still theoretical

Ran the full morning → afternoon cycle once manually against real data
before relying on the schedule: logged a real prediction (NIFTY FLAT, 70.5%
confidence), resolved it against a real fetched close the next check
(correct — actual move was -0.29%, inside the FLAT deadband), retrained
successfully, and confirmed the retrain trigger correctly stops firing
immediately after a retrain (caught and fixed a real bug here: the trigger
compares against total *headline* count, but the retrain bookkeeping was
initially storing a different, smaller row count — would have retrained on
literally every single check forever if shipped as first written). Also
confirmed the scheduler process survives detaching from its launching shell
(`nohup`/`disown`, verified via `ps` showing `PPID 1`).

**Not yet verified because it can't be within one sitting:** the weight
adjustment thresholds (5 wrong/right attributions, 3-in-a-row decay,
20%+ cumulative drift, source credibility crossing 20 predictions) — all of
that logic runs and is unit-testable, but hasn't fired for real yet because
one resolved prediction isn't enough history to reach any of those
thresholds. It'll start firing for real once the scheduler has been running
for a few weeks. Same honest-reporting posture as the rest of this project:
the mechanism is built and tested where it can be, not claimed to be proven
where it can't be yet.
