"""
PHASE 2: News → Sector → Stocks routing
Project: Global Financial News → Indian Stock Market Predictor
Goal: Take every FinBERT-labeled headline from Week 3's database and
      figure out which specific stocks it's actually relevant to,
      producing the (headline, affected stock, sentiment) dataset
      Phase 3's XGBoost model will train on.

What changed from Week 4:
  Week 4 answered "what does this headline feel like" (sentiment).
  Phase 2 answers "who does this headline actually affect" (routing).
  Neither is useful alone — a "negative" label is meaningless to a
  prediction model unless it's attached to a specific stock/sector.

Two routing mechanisms, used together, on purpose:
  1. COMPANY NAME MATCHING — if a headline mentions a company by
     name ("HDFC Bank", "TCS"), tag that exact stock. Precise, but
     only catches headlines that name a company directly.
  2. TOPIC → SECTOR MAPPING — the 6 search topics from Week 2/3
     (RBI, NIFTY, Indian_Banking, Indian_IT, US_Fed, Global_Markets)
     each map to a broad sector or the whole index. Catches macro
     headlines ("RBI hikes repo rate") that affect a whole sector
     without naming any single company — this is the "same-timezone
     Indian news" and "cross-timezone overnight news" routing the
     project's README describes.

Known limitation: some company display names are short (e.g. "SBI",
"ITC") and could theoretically match inside an unrelated word or a
different company's name. Word-boundary matching (not raw substring)
guards against the worst of this, but it's not perfect — treat this
as a reasonable first pass, not a guaranteed-precise NER system.

Run: python3 phase2_news_routing.py
"""

import re
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

SECTOR_MAP_CSV = Path("../phase1_expansion/phase1_sector_map.csv")
NEWS_DB = Path("../phase0_week3/news_pipeline.db")
OUTPUT_CSV = Path("phase2_routed_events.csv")

# Topic -> sector routing for headlines that don't name a specific
# company. "NIFTY 50 Index" is used as the target for genuinely
# market-wide/macro topics (global cues, US Fed) rather than any
# one sector, matching how the project's own README frames
# cross-timezone overnight news as a whole-market signal.
TOPIC_TO_SECTOR = {
    "RBI":            "Banking",
    "Indian_Banking": "Banking",
    "Indian_IT":      "IT",
    "NIFTY":          "Index",
    "US_Fed":         "Index",
    "Global_Markets": "Index",
}


# ─────────────────────────────────────────────
# STEP 1: Load the sector map and labeled news
# ─────────────────────────────────────────────

print("=" * 60)
print("  PHASE 2: News -> Sector -> Stocks routing")
print("=" * 60)

if not SECTOR_MAP_CSV.exists():
    print(f"\nMissing {SECTOR_MAP_CSV} — run phase1_expansion/phase1_nifty50.py first.")
    raise SystemExit(1)

if not NEWS_DB.exists():
    print(f"\nMissing {NEWS_DB} — run phase0_week3/week3_pipeline.py first.")
    raise SystemExit(1)

sector_map = pd.read_csv(SECTOR_MAP_CSV)
print(f"\nLoaded {len(sector_map)} tickers from the sector map.")

conn = sqlite3.connect(NEWS_DB)
news = pd.read_sql(
    "SELECT id, topic, headline, source, published_at, sentiment_label, sentiment_confidence "
    "FROM news_articles WHERE sentiment_label IS NOT NULL",
    conn,
)
conn.close()

print(f"Loaded {len(news)} labeled headlines from Week 3's database.")

if len(news) == 0:
    print("\nNo labeled headlines found. Run phase0_week4/week4_finbert.py first.")
    raise SystemExit(1)


# ─────────────────────────────────────────────
# STEP 2: Build company-name matchers
# ─────────────────────────────────────────────

# Word-boundary regex per company name, longest names first — this
# matters for names where one is a substring of another (e.g. "Tata
# Motors" vs "Tata Steel" vs "Tata Consumer" all start with "Tata"),
# so a longer, more specific match wins over a shorter partial one.
company_rows = sector_map[sector_map["sector"] != "Index"].copy()
company_rows["name_len"] = company_rows["name"].str.len()
company_rows = company_rows.sort_values("name_len", ascending=False)

matchers = [
    (row.ticker, row.name, row.sector, re.compile(r"\b" + re.escape(row.name) + r"\b", re.IGNORECASE))
    for row in company_rows.itertuples(index=False)
]


def match_companies(headline):
    """Return list of (ticker, name, sector) for every company named in the headline."""
    matches = []
    matched_spans = []
    for ticker, name, sector, pattern in matchers:
        m = pattern.search(headline)
        if not m:
            continue
        # Skip if this span is already covered by a longer match found earlier
        # (e.g. don't also match "Tata" generically once "Tata Motors" matched)
        if any(m.start() >= s and m.end() <= e for s, e in matched_spans):
            continue
        matches.append((ticker, name, sector))
        matched_spans.append((m.start(), m.end()))
    return matches


# ─────────────────────────────────────────────
# STEP 3: Route every headline
# ─────────────────────────────────────────────

print("\nRouting headlines to stocks/sectors...")

routed_rows = []

for row in news.itertuples(index=False):
    base = dict(
        article_id=row.id,
        headline=row.headline,
        topic=row.topic,
        source=row.source,
        published_at=row.published_at,
        sentiment_label=row.sentiment_label,
        sentiment_confidence=row.sentiment_confidence,
    )

    company_matches = match_companies(row.headline)
    for ticker, name, sector in company_matches:
        routed_rows.append({**base, "ticker": ticker, "company_name": name, "sector": sector, "matched_via": "company_name"})

    already_tagged_tickers = {t for t, _, _ in company_matches}
    macro_sector = TOPIC_TO_SECTOR.get(row.topic)
    if macro_sector:
        if macro_sector == "Index":
            index_row = sector_map[sector_map["ticker"] == "^NSEI"].iloc[0]
            if index_row["ticker"] not in already_tagged_tickers:
                routed_rows.append({
                    **base, "ticker": index_row["ticker"], "company_name": index_row["name"],
                    "sector": "Index", "matched_via": "topic_macro",
                })
        else:
            sector_tickers = sector_map[sector_map["sector"] == macro_sector]
            for _, srow in sector_tickers.iterrows():
                if srow["ticker"] in already_tagged_tickers:
                    continue
                routed_rows.append({
                    **base, "ticker": srow["ticker"], "company_name": srow["name"],
                    "sector": macro_sector, "matched_via": "topic_macro",
                })

routed = pd.DataFrame(routed_rows)


# ─────────────────────────────────────────────
# STEP 4: Summary + save
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print("ROUTING SUMMARY")
print(f"{'─' * 60}")
print(f"  Labeled headlines processed : {len(news)}")
print(f"  Total (headline, stock) rows: {len(routed)}")
print(f"  Unique stocks/index touched : {routed['ticker'].nunique()}")

unrouted = news[~news["id"].isin(routed["article_id"])]
if len(unrouted) > 0:
    print(f"  Headlines with NO routing   : {len(unrouted)}  (unmapped topic — check TOPIC_TO_SECTOR)")

print(f"\n  By matching method:")
print(routed["matched_via"].value_counts().to_string())

print(f"\n  By sector:")
print(routed["sector"].value_counts().to_string())

print(f"\n  Sentiment distribution across routed events:")
print(routed["sentiment_label"].value_counts().to_string())

if len(routed) > 0:
    print(f"\n  Sample routed events:")
    for _, r in routed.head(5).iterrows():
        print(f"    [{r['sentiment_label']:>8}] {r['company_name']:<20} ({r['matched_via']:<12}) <- {r['headline'][:60]}")

routed.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved -> {OUTPUT_CSV}")

print("\n" + "=" * 60)
print("  PHASE 2 ROUTING COMPLETE")
print("=" * 60)
print(f"""
What you just built:
  ✓ Matched company names directly mentioned in headlines to specific stocks
  ✓ Routed macro/topic headlines (RBI, US Fed, etc.) to whole sectors
  ✓ Produced one row per (headline, affected stock) pair — the shape
    Phase 3 needs: sentiment + which stock it's about + when

Next: Phase 3 — join phase2_routed_events.csv against actual next-day
price moves (from phase1_close_prices_all.csv) to build XGBoost's
training set: "given this sentiment about this stock, did the stock
go up, down, or stay flat the next trading day?"
""")
