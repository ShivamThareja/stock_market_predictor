"""
PHASE 0 — WEEK 4
Project: Global Financial News → Indian Stock Market Predictor
Goal: Replace manual sentiment labeling with FinBERT — a BERT model
      fine-tuned specifically on financial text — and prove it's
      trustworthy before relying on it.

What changed from Week 2/3:
  - Week 2: you hand-labeled 69 headlines (positive/negative/neutral)
  - Week 3: the pipeline started collecting headlines automatically,
    leaving sentiment_label NULL for every new row
  - Week 4 (this file): FinBERT fills in that NULL column automatically

Two steps, in order, on purpose:
  STEP A — VALIDATION
    Run FinBERT on your Week 2 hand-labeled headlines and compare
    its answers to your human labels. This tells you how much to
    trust FinBERT BEFORE you let it label new data unsupervised.
  STEP B — PRODUCTION LABELING
    Run FinBERT on every row in Week 3's database where
    sentiment_label is still NULL, and write the results back.

Model used: ProsusAI/finbert — trained specifically on financial
news/reports, not generic text. A generic sentiment model would
call "stock falls 5%" neutral-ish; FinBERT knows that's negative
in a financial context.

Run: python3 week4_finbert.py
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd
from transformers import pipeline

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

# Paths point back into Week 2/3's folders on purpose — this pipeline
# is meant to operate on the SAME evolving dataset across phases, not
# a copy. Week 4 updates Week 3's database in place.
WEEK2_LABELED_CSV = Path("../phase0_week2/week2_labeling_template.csv")
WEEK3_DB = Path("../phase0_week3/news_pipeline.db")

MODEL_NAME = "ProsusAI/finbert"

# FinBERT's own output labels are already lowercase positive/negative/neutral
# — same vocabulary Week 2's manual labeling used. No remapping needed,
# but we validate against this set anyway rather than trusting it blindly.
VALID_LABELS = {"positive", "negative", "neutral"}

BATCH_SIZE = 16


# ─────────────────────────────────────────────
# STEP 0: Load FinBERT
# ─────────────────────────────────────────────

print("=" * 60)
print("  PHASE 0 — WEEK 4: FinBERT Sentiment Labeling")
print("=" * 60)

print(f"\nLoading {MODEL_NAME}...")
print("(First run downloads ~400MB of model weights — one-time, then cached)")

classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

print("Model loaded.\n")


def classify_headlines(headlines):
    """Run FinBERT on a list of headlines, batched. Returns (label, score) pairs."""
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i:i + BATCH_SIZE]
        # truncation=True guards against any unusually long headline/description
        # blowing past BERT's 512-token limit
        preds = classifier(batch, truncation=True)
        for p in preds:
            label = p["label"].lower()
            if label not in VALID_LABELS:
                # Defensive check — if a future model version changes its
                # label vocabulary, fail loud here instead of silently
                # writing garbage into the database.
                raise ValueError(f"Unexpected FinBERT label: {p['label']!r}")
            results.append((label, round(p["score"], 4)))
    return results


# ─────────────────────────────────────────────
# STEP A: VALIDATION — FinBERT vs your Week 2 human labels
# ─────────────────────────────────────────────

print("─" * 60)
print("STEP A: Validating FinBERT against your Week 2 hand-labels")
print("─" * 60)

if not WEEK2_LABELED_CSV.exists():
    print(f"  Skipping — {WEEK2_LABELED_CSV} not found.")
else:
    labeled_df = pd.read_csv(WEEK2_LABELED_CSV)
    labeled_df = labeled_df[labeled_df["sentiment_label"].notna()]
    labeled_df = labeled_df[labeled_df["sentiment_label"].astype(str).str.strip() != ""]

    if len(labeled_df) == 0:
        print("  Skipping — no labeled rows found in the Week 2 file.")
    else:
        print(f"  Running FinBERT on {len(labeled_df)} hand-labeled headlines...")
        predictions = classify_headlines(labeled_df["headline"].tolist())
        labeled_df = labeled_df.copy()
        labeled_df["finbert_label"] = [p[0] for p in predictions]
        labeled_df["finbert_score"] = [p[1] for p in predictions]
        labeled_df["human_label"] = labeled_df["sentiment_label"].astype(str).str.strip().str.lower()

        agree = (labeled_df["finbert_label"] == labeled_df["human_label"])
        accuracy = agree.mean() * 100

        print(f"\n  Agreement with your human labels: {accuracy:.1f}%  "
              f"({agree.sum()}/{len(labeled_df)})")

        print("\n  Breakdown by human label:")
        for label in ["positive", "negative", "neutral"]:
            subset = labeled_df[labeled_df["human_label"] == label]
            if len(subset) == 0:
                continue
            subset_agree = (subset["finbert_label"] == label).mean() * 100
            print(f"    {label:<10} {len(subset):>3} headlines  →  FinBERT matched {subset_agree:.0f}%")

        disagreements = labeled_df[~agree]
        if len(disagreements) > 0:
            print(f"\n  Sample disagreements (first 3 of {len(disagreements)}):")
            for _, row in disagreements.head(3).iterrows():
                print(f"    \"{row['headline'][:70]}...\"")
                print(f"      you said: {row['human_label']}   FinBERT said: {row['finbert_label']} (confidence {row['finbert_score']})")

        disagreements_path = Path("week4_validation_disagreements.csv")
        disagreements.to_csv(disagreements_path, index=False)
        print(f"\n  Full disagreement list saved → {disagreements_path}")

        print(f"\n  {'⚠️  Below 70% — review disagreements before trusting FinBERT on new data.' if accuracy < 70 else '✓ Looks trustworthy enough to auto-label new headlines.'}")


# ─────────────────────────────────────────────
# STEP B: PRODUCTION LABELING — fill in Week 3's database
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("STEP B: Auto-labeling unlabeled rows in Week 3's database")
print("─" * 60)

if not WEEK3_DB.exists():
    print(f"  Skipping — {WEEK3_DB} not found.")
    print("  Run phase0_week3/week3_pipeline.py at least once first.")
else:
    conn = sqlite3.connect(WEEK3_DB)

    # Add a confidence column if it doesn't exist yet — SQLite has no
    # "ADD COLUMN IF NOT EXISTS", so we try and ignore the error if it's
    # already there (e.g. this script was run before).
    try:
        conn.execute("ALTER TABLE news_articles ADD COLUMN sentiment_confidence REAL")
    except sqlite3.OperationalError:
        pass

    unlabeled = pd.read_sql(
        "SELECT id, headline FROM news_articles WHERE sentiment_label IS NULL",
        conn,
    )

    if len(unlabeled) == 0:
        print("  Nothing to label — every row already has a sentiment_label.")
    else:
        print(f"  Found {len(unlabeled)} unlabeled headlines. Running FinBERT...")
        predictions = classify_headlines(unlabeled["headline"].tolist())

        cur = conn.cursor()
        for (row_id, _), (label, score) in zip(unlabeled.itertuples(index=False), predictions):
            cur.execute(
                "UPDATE news_articles SET sentiment_label = ?, sentiment_confidence = ? WHERE id = ?",
                (label, score, row_id),
            )
        conn.commit()

        counts = pd.Series([p[0] for p in predictions]).value_counts()
        print(f"\n  Labeled {len(unlabeled)} new rows:")
        for label, count in counts.items():
            print(f"    {label:<10} {count}")

    total = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    still_null = conn.execute(
        "SELECT COUNT(*) FROM news_articles WHERE sentiment_label IS NULL"
    ).fetchone()[0]
    print(f"\n  Database totals: {total} articles, {still_null} still unlabeled")
    conn.close()


print("\n" + "=" * 60)
print("  WEEK 4 COMPLETE")
print("=" * 60)
print("""
What you just built:
  ✓ Loaded FinBERT — a finance-domain sentiment model
  ✓ Validated it against your own hand-labeled Week 2 data
  ✓ Auto-labeled every unlabeled headline in Week 3's database

Next: Phase 2 — use phase1_sector_map.csv to automatically route
each labeled headline to the stocks/sectors it affects (e.g. an
RBI headline → flag every Banking stock), building the dataset
Phase 3's XGBoost model will train on.
""")
