# Phase 2: News → Sector → Stocks Routing

## What's in this folder

| File | Purpose |
|---|---|
| `phase2_news_routing.py` | Main script — routes labeled headlines to affected stocks |

## What changed from Week 4

Week 4 answered "what does this headline feel like" (sentiment). Phase 2
answers "who does this headline actually affect" — neither is useful to a
prediction model alone. A `negative` label means nothing to Phase 3's
XGBoost model unless it's attached to a specific stock.

## Two routing mechanisms, used together

1. **Company name matching** — if a headline names a company directly
   ("HDFC Bank", "TCS"), that exact stock gets tagged. Word-boundary regex
   matching, longest company names checked first (so "Tata Motors" wins
   over a generic "Tata" match).
2. **Topic → sector mapping** — the 6 search topics from Week 2/3 (RBI,
   NIFTY, Indian_Banking, Indian_IT, US_Fed, Global_Markets) each map to a
   whole sector or the NIFTY 50 index itself. This catches macro headlines
   that affect a whole sector without naming any single company — e.g.
   "RBI hikes repo rate" → every Banking stock, "Fed holds rates steady" →
   the whole index (same-timezone vs cross-timezone news, per the
   project's core idea).

Both run per headline — a headline can get tagged via company name,
topic-macro, or both (duplicates on the same ticker are collapsed).

## How to run

```bash
python3 phase2_news_routing.py
```

Requires `phase1_expansion/phase1_sector_map.csv` (from Phase 1) and
`phase0_week3/news_pipeline.db` with at least some FinBERT-labeled rows
(from Week 4) to already exist.

## What the script does

1. Loads the 56-ticker sector map and every labeled headline
2. For each headline: checks for direct company name mentions, then adds
   topic-based sector/index routing for anything not already covered
3. Produces one row per (headline, affected stock) pair
4. Prints a summary: routing method breakdown, sector breakdown,
   sentiment breakdown

## Files created after running

| File | Purpose |
|---|---|
| `phase2_routed_events.csv` | One row per (headline, stock) — sentiment, ticker, sector, how it was matched |

## Known limitation

A few company display names are short (e.g. "SBI", "ITC") and could
theoretically match inside an unrelated word or a different entity's name.
Word-boundary matching (not raw substring search) guards against the
worst of this, but it's a reasonable first pass, not a precision NER
system. Worth spot-checking `phase2_routed_events.csv` for false positives
as the dataset grows.

## Next: Phase 3
Join `phase2_routed_events.csv` against actual next-day price moves (from
`phase1_expansion/phase1_close_prices_all.csv`) to build XGBoost's
training set: given this sentiment about this stock, did it go up, down,
or stay flat the next trading day?
