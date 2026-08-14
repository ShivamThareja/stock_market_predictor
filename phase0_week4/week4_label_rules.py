"""
PHASE 0 — WEEK 4 (continued): Rule-based sentiment corrections
Project: Global Financial News → Indian Stock Market Predictor
Goal: Catch a handful of domain patterns FinBERT gets wrong
      systematically — rate policy language and market-direction
      verbs — without touching headlines it already gets right.

Why "correct known errors" instead of "reclassify by keyword":
  The obvious version of this ("any headline with 'rally' in it ->
  positive") was tested against this project's own real headline set
  before being written this way, and it breaks immediately:
  "Profit-taking, debt supply stall India bond rally" contains
  "rally" but is bearish — the rally is stalling. Blanket keyword
  overrides would have flipped that to positive, making the label
  WORSE than FinBERT's own (correct) "negative" call.

  So every rule here is gated on what FinBERT ALREADY said: it only
  fires to fix a specific, predictable failure mode, and leaves
  FinBERT's call alone everywhere else — including cases where the
  keyword appears but FinBERT was already right.

Rules (all headline-text, case-insensitive):
  1. Rate HOLD  (RBI/Fed/repo/interest rate + "hold(s)")
     FinBERT said negative -> neutral
     ("hold" isn't inherently bad news; FinBERT's finance-tuning
     still leans negative on any "no change" framing)
  2. Rate HIKE  (RBI/Fed/repo/interest rate + "hike(s/d)")
     FinBERT said positive -> negative
     (a hike is tightening; treat as negative unless FinBERT already
     called it negative/neutral)
  3. Rate CUT   (RBI/Fed/repo/interest rate + "cut(s)")
     FinBERT said negative -> positive
     (a cut is stimulative; same asymmetric gating as HOLD)
  4. Market RALLY words (rallies/gains/climbs/surges/jumps/soars/
     rises), no FALL words in the same headline
     FinBERT said neutral -> positive
  5. Market FALL words (falls/slides/tanks/plunges/drops/slumps/
     sinks/tumbles/declines), no RALLY words in the same headline
     FinBERT said neutral -> negative
  6. Pure schedule/preview headlines ("weekly wrap", "what to
     expect", "things to know", "week ahead", "day ahead", "top
     stories") with no rate or rally/fall language alongside
     FinBERT said positive/negative -> neutral

Deliberately NOT covered: "cut" also means budget/jobs/forecast
cuts (negative), not just rate cuts — the RATE_CONTEXT gate handles
the common case but headlines like "RBI cuts growth forecast" will
still slip through as a false positive-rule match sometimes. Same
"reasonable first pass, not precision NLP" caveat Phase 2's company
matching already carries. Spot-check week4_label_rules_changes.csv
as the dataset grows.

Preserves FinBERT's original call in a new sentiment_label_finbert
column before overwriting sentiment_label, so nothing here is
destructive or unauditable.

Run: python3 week4_label_rules.py
"""

import re
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DB_FILE = Path("../phase0_week3/news_pipeline.db")
CHANGES_CSV = Path("week4_label_rules_changes.csv")

RATE_CONTEXT = re.compile(r"\b(RBI|Fed|Federal Reserve|repo rate|interest rate|rate)\b", re.IGNORECASE)
HOLD_RE = re.compile(r"\bhold(s|ing)?\b", re.IGNORECASE)
HIKE_RE = re.compile(r"\bhik(e|es|ed)\b", re.IGNORECASE)
CUT_RE = re.compile(r"\bcut(s)?\b", re.IGNORECASE)

RALLY_RE = re.compile(r"\b(rall(y|ies|ied)|gains?|climbs?|surges?|jumps?|soars?|rises?|rose|advance[sd]?)\b", re.IGNORECASE)
FALL_RE = re.compile(r"\b(falls?|fell|slides?|tanks?|plunges?|drops?|slumps?|sinks?|tumbles?|declines?)\b", re.IGNORECASE)

PREVIEW_RE = re.compile(r"\b(weekly wrap|what to expect|things to know|week ahead|day ahead|top stories)\b", re.IGNORECASE)


def apply_rules(headline, current_label):
    """Returns (new_label, rule_name) or (current_label, None) if no rule fires."""
    rate_ctx = RATE_CONTEXT.search(headline)

    if rate_ctx and HOLD_RE.search(headline) and current_label == "negative":
        return "neutral", "rate_hold"
    if rate_ctx and HIKE_RE.search(headline) and current_label == "positive":
        return "negative", "rate_hike"
    if rate_ctx and CUT_RE.search(headline) and current_label == "negative":
        return "positive", "rate_cut"

    has_rally = bool(RALLY_RE.search(headline))
    has_fall = bool(FALL_RE.search(headline))
    if has_rally and not has_fall and current_label == "neutral":
        return "positive", "market_rally"
    if has_fall and not has_rally and current_label == "neutral":
        return "negative", "market_fall"

    if (PREVIEW_RE.search(headline) and not rate_ctx and not has_rally and not has_fall
            and current_label in ("positive", "negative")):
        return "neutral", "preview_digest"

    return current_label, None


print("=" * 60)
print("  WEEK 4: Rule-based sentiment corrections")
print("=" * 60)

if not DB_FILE.exists():
    print(f"\nMissing {DB_FILE} — run phase0_week3/week3_pipeline.py first.")
    raise SystemExit(1)

conn = sqlite3.connect(DB_FILE)
try:
    conn.execute("ALTER TABLE news_articles ADD COLUMN sentiment_label_finbert TEXT")
except sqlite3.OperationalError:
    pass  # already added by a previous run

news = pd.read_sql(
    "SELECT id, headline, sentiment_label, sentiment_label_finbert FROM news_articles "
    "WHERE sentiment_label IS NOT NULL", conn,
)
print(f"\nLoaded {len(news)} labeled headlines.")

# Preserve FinBERT's original call the first time this row is touched —
# never overwrite it on a re-run, so repeated runs stay idempotent and
# the audit trail always points back to the true zero-shot output.
news["sentiment_label_finbert"] = news["sentiment_label_finbert"].fillna(news["sentiment_label"])

changes = []
cur = conn.cursor()
for row in news.itertuples(index=False):
    new_label, rule = apply_rules(row.headline, row.sentiment_label_finbert)
    if rule is not None and new_label != row.sentiment_label:
        changes.append({
            "id": row.id, "headline": row.headline, "rule": rule,
            "finbert_label": row.sentiment_label_finbert, "corrected_label": new_label,
        })
    cur.execute(
        "UPDATE news_articles SET sentiment_label = ?, sentiment_label_finbert = ? WHERE id = ?",
        (new_label, row.sentiment_label_finbert, row.id),
    )
conn.commit()
conn.close()

print(f"\n{'─' * 60}")
print("RESULTS")
print(f"{'─' * 60}")
print(f"  Rows changed: {len(changes)} / {len(news)}")

if changes:
    changes_df = pd.DataFrame(changes)
    print("\n  By rule:")
    print(changes_df["rule"].value_counts().to_string())
    print("\n  Sample corrections:")
    for c in changes[:8]:
        print(f"    [{c['rule']:<14}] {c['finbert_label']} -> {c['corrected_label']}  \"{c['headline'][:65]}\"")
    changes_df.to_csv(CHANGES_CSV, index=False)
    print(f"\n  Full change log -> {CHANGES_CSV}")
else:
    print("  No changes — either already applied, or no rows matched a rule's gate condition.")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
print("""
sentiment_label now reflects FinBERT + these targeted corrections;
sentiment_label_finbert keeps the original zero-shot call for audit.
Downstream (Phase 2 routing, Phase 3 training) reads sentiment_label,
so re-run phase2_news_routing.py after this to propagate the changes.
""")
