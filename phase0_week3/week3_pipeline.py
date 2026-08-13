"""
PHASE 0 — WEEK 3
Project: Global Financial News → Indian Stock Market Predictor
Goal: Turn Week 2's one-off news pull into an automated pipeline
      that runs every hour and stores results in a real database.

What changed from Week 2:
  - CSV overwritten every run  →  SQLite database, append-only
  - Manual "run it when I remember"  →  APScheduler runs the fetch
    automatically on a fixed interval
  - No duplicate handling across runs  →  UNIQUE constraint on
    headline, so the same story never gets stored twice even
    though every run re-fetches an overlapping time window
  - Same 6 search topics carried forward from Week 2
  - sentiment_label column kept (NULL for now) — Week 4's FinBERT
    will fill it in automatically, same column your manual
    Week 2 labeling used

IMPORTANT: Uses the same NewsAPI key as Week 2 — paste it into
config.py the same way (see config_template.py).

Run ONE fetch cycle and exit (good for testing):
  python3 week3_pipeline.py --once

Run the real pipeline (fetches now, then every hour, forever
until you Ctrl+C):
  python3 week3_pipeline.py
"""

import sqlite3
import sys
import warnings
from datetime import datetime, timedelta

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

warnings.filterwarnings("ignore")

from config import NEWS_API_KEY


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

# Same 6 topics from Week 2 — already proven to return relevant results
SEARCH_QUERIES = {
    "RBI":            "RBI repo rate India",
    "NIFTY":          "NIFTY 50 stock market India",
    "Indian_Banking": "Indian banking sector HDFC ICICI SBI",
    "Indian_IT":      "Indian IT sector TCS Infosys Wipro",
    "US_Fed":         "Federal Reserve interest rate decision",
    "Global_Markets": "global stock market crash rally",
}

DB_FILE = "news_pipeline.db"
ARTICLES_PER_QUERY = 20

# Why 2 days back (not Week 2's 7): this now runs every hour, so
# each run only needs to cover the gap since the last one. We still
# look back further than 1 hour on purpose — if the pipeline was
# offline for a while (laptop closed, script crashed), a 2-day
# window means we don't silently lose articles. The UNIQUE
# constraint on headline means re-fetching the same 2-day window
# every run costs nothing — duplicates are just skipped.
DAYS_BACK = 2

FETCH_INTERVAL_HOURS = 1

BASE_URL = "https://newsapi.org/v2/everything"

# Free tier quota check — do this math BEFORE scheduling anything:
#   6 topics × 1 request each × 24 runs/day = 144 requests/day
#   NewsAPI free tier allows 1,000 requests/day.
#   144/1000 = comfortably under the limit, with room to add more
#   topics later if needed.


# ─────────────────────────────────────────────
# STEP 1: Database setup
# ─────────────────────────────────────────────

def init_db():
    """
    Create the news_articles table if it doesn't exist yet.
    headline has a UNIQUE constraint — this is what makes the
    pipeline safe to run every hour without piling up duplicates.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            topic           TEXT,
            headline        TEXT UNIQUE NOT NULL,
            source          TEXT,
            published_at    TEXT,
            url             TEXT,
            description     TEXT,
            sentiment_label TEXT,
            fetched_at      TEXT
        )
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# STEP 2: One fetch-and-store cycle
# ─────────────────────────────────────────────

def fetch_and_store():
    """
    Pull the latest headlines for all 6 topics and insert new ones
    into SQLite. Duplicate headlines (already in the DB from an
    earlier run) are silently skipped — that's expected, not an error.
    """
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE RUN — {run_time}")
    print(f"{'=' * 60}")

    from_date = (datetime.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    to_date = datetime.today().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    total_fetched = 0
    total_inserted = 0
    total_duplicate = 0
    total_failed_topics = 0

    for topic_name, query in SEARCH_QUERIES.items():
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": ARTICLES_PER_QUERY,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"  {topic_name}: request failed ({e}), skipping")
            total_failed_topics += 1
            continue

        if response.status_code != 200:
            print(f"  {topic_name}: HTTP {response.status_code}, skipping")
            total_failed_topics += 1
            continue

        articles = response.json().get("articles", [])
        total_fetched += len(articles)

        inserted_this_topic = 0
        for article in articles:
            headline = article.get("title", "")
            if not headline:
                continue
            try:
                cur.execute(
                    """
                    INSERT INTO news_articles
                        (topic, headline, source, published_at, url, description, sentiment_label, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        topic_name,
                        headline,
                        article.get("source", {}).get("name", "Unknown"),
                        article.get("publishedAt", ""),
                        article.get("url", ""),
                        article.get("description", ""),
                        run_time,
                    ),
                )
                inserted_this_topic += 1
                total_inserted += 1
            except sqlite3.IntegrityError:
                # headline already exists — this is the dedup working as intended
                total_duplicate += 1

        print(f"  {topic_name:<16} fetched {len(articles):>3}  |  new {inserted_this_topic:>3}")

    conn.commit()

    total_in_db = cur.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    unlabeled = cur.execute(
        "SELECT COUNT(*) FROM news_articles WHERE sentiment_label IS NULL"
    ).fetchone()[0]
    conn.close()

    print(f"{'─' * 60}")
    print(f"  Fetched this run   : {total_fetched}")
    print(f"  New articles saved : {total_inserted}")
    print(f"  Duplicates skipped : {total_duplicate}")
    if total_failed_topics:
        print(f"  Topics that failed : {total_failed_topics}")
    print(f"  Total in database  : {total_in_db}")
    print(f"  Awaiting sentiment : {unlabeled}  (Week 4's FinBERT will fill these in)")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────
# STEP 3: Entry point
# ─────────────────────────────────────────────

def run_once():
    print("Running a single fetch cycle (--once mode)...")
    init_db()
    fetch_and_store()
    print("\nDone. Database saved to:", DB_FILE)
    print("Run without --once to start the real hourly pipeline.")


def run_scheduler():
    init_db()
    print("=" * 60)
    print("  PHASE 0 — WEEK 3: Automated News Pipeline")
    print("=" * 60)
    print(f"\nFetching every {FETCH_INTERVAL_HOURS} hour(s).")
    print("Press Ctrl+C to stop.\n")

    # Run once immediately — don't make the user wait an hour to see it work
    fetch_and_store()

    scheduler = BlockingScheduler()
    scheduler.add_job(fetch_and_store, "interval", hours=FETCH_INTERVAL_HOURS)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\nPipeline stopped. Database saved at:", DB_FILE)
        print("Your data is safe — just run the script again to resume collecting.")


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
    else:
        run_scheduler()
