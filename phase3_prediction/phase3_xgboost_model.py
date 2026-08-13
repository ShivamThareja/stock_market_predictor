"""
PHASE 3: XGBoost prediction model
Project: Global Financial News → Indian Stock Market Predictor
Goal: Given a day's news sentiment about a stock (from Phase 2's
      routed events), predict whether that stock goes UP, DOWN, or
      stays FLAT on the next trading day.

What changed from Phase 2:
  Phase 2 produced one row per (headline, affected stock) — too
  granular to predict from directly (a stock can get 20 headlines
  in a day, each with its own sentiment). Phase 3 aggregates that
  down to one row per (ticker, day): "how positive/negative was the
  news about this stock today, in total" — and attaches the label
  the model is actually trying to predict: did the stock go up,
  down, or stay flat by the next trading session's close.

Why "next trading session", not "next calendar day":
  News published on a Saturday can't move the market until Monday.
  For every (ticker, headline date) pair we find the most recent
  close AT OR BEFORE the headline date (the baseline) and the next
  close AFTER it (the target), and measure the return between them.
  That's the real "before NSE opens" move the project's README
  describes, whether the news broke on a trading day or a weekend.
  Feature/label construction lives in feature_engineering.py — the
  final dashboard's live-prediction panel reuses the exact same code
  path so training and live inference never drift apart.

Label thresholds:
  UP    : next-session return > +FLAT_THRESHOLD
  DOWN  : next-session return < -FLAT_THRESHOLD
  FLAT  : everything in between
  FLAT_THRESHOLD (0.3%) reuses the spirit of Phase 0/1's own
  NOISE_THRESHOLD (0.5%) — small moves are market noise, not a
  signal the news caused, so they shouldn't count as UP/DOWN.

Split: time-based (train on the earliest ~80% of days, test on the
most recent ~20%), NOT a random shuffle. A random split lets rows
from the same day leak sentiment patterns between train and test;
a real predictor only ever has the past to learn from.

Run: python3 phase3_xgboost_model.py
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_engineering import FEATURE_COLS, build_dataset, fit_encoders, make_model, model_matrix

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

ROUTED_EVENTS_CSV = Path("../phase2_routing/phase2_routed_events.csv")
PRICES_CSV = Path("../phase1_expansion/phase1_close_prices_all.csv")
SECTOR_MAP_CSV = Path("../phase1_expansion/phase1_sector_map.csv")

TEST_FRACTION = 0.2           # most recent slice of days held out for testing
MIN_ROWS_TO_TRAIN = 40        # below this, a train/test split is too noisy to trust

OUTPUT_DATASET_CSV = Path("phase3_daily_features.csv")
OUTPUT_PREDICTIONS_CSV = Path("phase3_test_predictions.csv")
OUTPUT_IMPORTANCE_CSV = Path("phase3_feature_importance.csv")
OUTPUT_METRICS_CSV = Path("phase3_metrics.csv")


print("=" * 60)
print("  PHASE 3: XGBoost next-session direction model")
print("=" * 60)

for p in (ROUTED_EVENTS_CSV, PRICES_CSV, SECTOR_MAP_CSV):
    if not p.exists():
        print(f"\nMissing {p} — run the earlier phases first.")
        raise SystemExit(1)


# ─────────────────────────────────────────────
# STEP 1: Load inputs, build the daily dataset
# ─────────────────────────────────────────────

routed = pd.read_csv(ROUTED_EVENTS_CSV)
prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True).sort_index()
sector_map = pd.read_csv(SECTOR_MAP_CSV)

print(f"\nLoaded {len(routed)} routed (headline, stock) rows across "
      f"{pd.to_datetime(routed['published_at']).dt.normalize().nunique()} distinct news days.")
print(f"Loaded price history for {prices.shape[1]} tickers, "
      f"{prices.index[0].date()} -> {prices.index[-1].date()}.")

print("\nBuilding daily (ticker, date) features: sentiment aggregates, "
      "price momentum, market-wide sentiment...")
full_dataset = build_dataset(routed, prices, sector_map)

n_unresolved = full_dataset["label"].isna().sum()
dataset = full_dataset[full_dataset["label"].notna()].reset_index(drop=True)
if n_unresolved:
    print(f"  {n_unresolved} (ticker, date) rows have no next-session close yet "
          f"(too recent) — excluded from training, left for the dashboard's live predictions.")

dataset.to_csv(OUTPUT_DATASET_CSV, index=False)
print(f"  Built {len(dataset)} labeled (ticker, day) training rows "
      f"across {dataset['date'].nunique()} days and {dataset['ticker'].nunique()} tickers.")
print(f"  Saved -> {OUTPUT_DATASET_CSV}")

print("\n  Label distribution:")
print(dataset["label"].value_counts().to_string())

if len(dataset) < MIN_ROWS_TO_TRAIN:
    print(f"\nOnly {len(dataset)} rows — below MIN_ROWS_TO_TRAIN ({MIN_ROWS_TO_TRAIN}).")
    print("Not enough data yet for a train/test split to mean anything.")
    print("Let phase0_week3/week3_pipeline.py run for more days, "
          "then re-run phase2_news_routing.py and this script.")
    raise SystemExit(0)


# ─────────────────────────────────────────────
# STEP 2: Time-based train/test split
# ─────────────────────────────────────────────

unique_dates = sorted(dataset["date"].unique())
split_idx = max(1, int(len(unique_dates) * (1 - TEST_FRACTION)))
split_date = unique_dates[split_idx]

train = dataset[dataset["date"] < split_date]
test = dataset[dataset["date"] >= split_date]

print(f"\n{'─' * 60}")
print(f"Time-based split at {split_date.date()}")
print(f"  Train: {len(train)} rows ({train['date'].nunique()} days, "
      f"{train['date'].min().date()} -> {train['date'].max().date()})")
print(f"  Test : {len(test)} rows ({test['date'].nunique()} days, "
      f"{test['date'].min().date()} -> {test['date'].max().date()})")

if len(test) == 0 or train["label"].nunique() < 2:
    print("\nTest split is empty or train set has only one class — "
          "need more days of data across more varied market moves.")
    raise SystemExit(0)


# ─────────────────────────────────────────────
# STEP 3: Train XGBoost
# ─────────────────────────────────────────────

sector_encoder, label_encoder = fit_encoders(dataset)
X_train = model_matrix(train, sector_encoder)
X_test = model_matrix(test, sector_encoder)
y_train = label_encoder.transform(train["label"])

model_features = FEATURE_COLS + ["sector_enc"]

print(f"\n{'─' * 60}")
print("Training XGBoost (multi:softprob, 3 classes)...")

model = make_model()
model.fit(X_train, y_train)

pred = model.predict(X_test)
pred_labels = label_encoder.inverse_transform(pred)
true_labels = test["label"].values


# ─────────────────────────────────────────────
# STEP 4: Evaluate — against a majority-class
# baseline, not in isolation (raw accuracy alone
# is meaningless when classes are imbalanced)
# ─────────────────────────────────────────────

model_accuracy = accuracy_score(true_labels, pred_labels)
majority_label = train["label"].mode().iloc[0]
baseline_accuracy = (test["label"] == majority_label).mean()

print(f"\n{'─' * 60}")
print("RESULTS")
print(f"{'─' * 60}")
print(f"  XGBoost accuracy          : {model_accuracy:.1%}  ({(pred_labels == true_labels).sum()}/{len(test)})")
print(f"  Majority-class baseline   : {baseline_accuracy:.1%}  (always predicting '{majority_label}')")
print(f"  Lift over baseline        : {(model_accuracy - baseline_accuracy) * 100:+.1f} pts")

print("\n  Classification report:")
print(classification_report(true_labels, pred_labels, zero_division=0))

print("  Confusion matrix (rows = actual, cols = predicted), order [DOWN, FLAT, UP]:")
labels_order = ["DOWN", "FLAT", "UP"]
cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
print(cm_df.to_string())

print("\n  Feature importance:")
importances = pd.Series(model.feature_importances_, index=model_features).sort_values(ascending=False)
for feat, imp in importances.items():
    print(f"    {feat:<16} {imp:.3f}")
importances.rename("importance").rename_axis("feature").reset_index().to_csv(OUTPUT_IMPORTANCE_CSV, index=False)

test_out = test[["ticker", "date", "sector", "next_session_return", "label"]].copy()
test_out["predicted_label"] = pred_labels
test_out.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
print(f"\n  Saved test-set predictions -> {OUTPUT_PREDICTIONS_CSV}")
print(f"  Saved feature importances  -> {OUTPUT_IMPORTANCE_CSV}")

pd.DataFrame([{
    "run_at": pd.Timestamp.now().isoformat(),
    "n_labeled_rows": len(dataset),
    "n_news_days": dataset["date"].nunique(),
    "train_rows": len(train),
    "test_rows": len(test),
    "split_date": str(split_date.date()),
    "model_accuracy": model_accuracy,
    "baseline_accuracy": baseline_accuracy,
    "majority_label": majority_label,
    "lift_pts": (model_accuracy - baseline_accuracy) * 100,
}]).to_csv(OUTPUT_METRICS_CSV, index=False)
print(f"  Saved run metrics          -> {OUTPUT_METRICS_CSV}")

print("\n" + "=" * 60)
print("  PHASE 3 COMPLETE")
print("=" * 60)
print(f"""
What you just built:
  - Aggregated Phase 2's per-headline sentiment into daily
    per-ticker features (volume, sentiment mix, confidence)
  - Labeled each (ticker, day) with its actual next-trading-
    session direction (UP/DOWN/FLAT) from real price data
  - Trained XGBoost on a strict time-based split and compared
    it against a majority-class baseline, not raw accuracy alone

Caveat worth taking seriously: this dataset spans only
{dataset['date'].nunique()} news days (~1 backfilled month) and is
dominated by whatever topics/sectors got the most news coverage
in that window. Numbers here will move a lot as more real days
accumulate — treat this run as "the pipeline works end-to-end",
not "here is a production-grade accuracy number". Re-run this
script periodically as phase0_week3/week3_pipeline.py collects
more days; the time-based split means genuinely new test days
each time, which is the real test of whether this is learning
signal or noise.

Next: run the final dashboard (../final_dashboard/) to see data
collection, sentiment, routing, and this model in one view,
including live in-flight predictions for news too recent to have
a next-session close yet.
""")
