"""
PHASE 0 — WEEK 2
Project: Global Financial News → Indian Stock Market Predictor
Goal: Pull real financial news headlines using NewsAPI and understand them

What this file teaches you:
  - How to call a real API with authentication
  - How API responses (JSON) are structured
  - How to extract and clean headline data
  - How to manually label headlines (groundwork for Phase 1's FinBERT)

IMPORTANT: Before running, paste your NewsAPI key into config.py
Run: python3 week2_news.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

from config import NEWS_API_KEY


# ─────────────────────────────────────────────
# DECISIONS
# ─────────────────────────────────────────────

# NewsAPI free tier limits:
#   - 1,000 requests/day
#   - Only last 30 days of articles (no historical access)
#   - 100 articles max per request

# Search queries — each one is a separate API call
# These cover the main topics relevant to your project
SEARCH_QUERIES = {
    "RBI":            "RBI repo rate India",
    "NIFTY":          "NIFTY 50 stock market India",
    "Indian_Banking": "Indian banking sector HDFC ICICI SBI",
    "Indian_IT":      "Indian IT sector TCS Infosys Wipro",
    "US_Fed":         "Federal Reserve interest rate decision",
    "Global_Markets":"global stock market crash rally",
}

DAYS_BACK = 7  # how many days of news to pull (free tier max useful window)
ARTICLES_PER_QUERY = 20  # keep modest to conserve daily request quota


# ─────────────────────────────────────────────
# STEP 1: Test the API connection
# ─────────────────────────────────────────────

print("=" * 60)
print("  PHASE 0 — WEEK 2: Financial News Explorer")
print("=" * 60)

BASE_URL = "https://newsapi.org/v2/everything"

print("\nTesting NewsAPI connection...")

test_params = {
    "q": "NIFTY 50",
    "apiKey": NEWS_API_KEY,
    "pageSize": 1,
}

test_response = requests.get(BASE_URL, params=test_params)

if test_response.status_code == 200:
    print("✓ Connection successful! API key is working.")
    data = test_response.json()
    print(f"  Total articles available for 'NIFTY 50': {data.get('totalResults', 0):,}")
elif test_response.status_code == 401:
    print("✗ Connection failed — API key is invalid or missing.")
    print("  Check that you pasted your key correctly in config.py")
    exit()
elif test_response.status_code == 426:
    print("✗ Connection failed — free tier doesn't support this query type.")
    exit()
else:
    print(f"✗ Connection failed — status code {test_response.status_code}")
    print(f"  Response: {test_response.text[:200]}")
    exit()


# ─────────────────────────────────────────────
# STEP 2: Pull headlines for each search query
# ─────────────────────────────────────────────

print(f"\nPulling news for {len(SEARCH_QUERIES)} topics, last {DAYS_BACK} days...\n")

from_date = (datetime.today() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
to_date = datetime.today().strftime("%Y-%m-%d")

all_articles = []

for topic_name, query in SEARCH_QUERIES.items():
    print(f"  Fetching: {topic_name} ({query!r})...")

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": ARTICLES_PER_QUERY,
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"    Warning: request failed ({response.status_code}), skipping")
        continue

    data = response.json()
    articles = data.get("articles", [])
    print(f"    Got {len(articles)} articles")

    for article in articles:
        all_articles.append({
            "topic":        topic_name,
            "headline":     article.get("title", ""),
            "source":       article.get("source", {}).get("name", "Unknown"),
            "published_at": article.get("publishedAt", ""),
            "url":          article.get("url", ""),
            "description":  article.get("description", ""),
        })

    # Be polite to the free API — small delay between calls
    time.sleep(1)


# ─────────────────────────────────────────────
# STEP 3: Clean and structure the data
# ─────────────────────────────────────────────

df = pd.DataFrame(all_articles)

if len(df) == 0:
    print("\nNo articles were retrieved. Check your API key and internet connection.")
    exit()

# Remove duplicate headlines (same story from multiple searches)
before_dedup = len(df)
df = df.drop_duplicates(subset=["headline"])
after_dedup = len(df)

# Parse the timestamp into a proper datetime
df["published_at"] = pd.to_datetime(df["published_at"])
df = df.sort_values("published_at", ascending=False)

print(f"\n{'─' * 60}")
print(f"NEWS DATA SUMMARY")
print(f"{'─' * 60}")
print(f"Total articles fetched:  {before_dedup}")
print(f"After removing duplicates: {after_dedup}")
print(f"Date range: {df['published_at'].min().date()} to {df['published_at'].max().date()}")

print(f"\nArticles per topic:")
print(df["topic"].value_counts().to_string())

print(f"\nArticles per source (top 10):")
print(df["source"].value_counts().head(10).to_string())


# ─────────────────────────────────────────────
# STEP 4: Show real examples
# ─────────────────────────────────────────────

print(f"\n{'─' * 60}")
print(f"SAMPLE HEADLINES (5 most recent):")
print(f"{'─' * 60}")

for _, row in df.head(5).iterrows():
    print(f"\n[{row['topic']}] {row['source']} — {row['published_at'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  {row['headline']}")


# ─────────────────────────────────────────────
# STEP 5: Save to CSV — and prepare for manual labeling
# ─────────────────────────────────────────────

# Save full dataset
df.to_csv("week2_news_raw.csv", index=False)

# Create a labeling template — empty sentiment column
# you'll fill in by hand for Week 2 homework
labeling_df = df[["headline", "source", "topic", "published_at"]].copy()
labeling_df["sentiment_label"] = ""  # you fill this: positive / negative / neutral
labeling_df.to_csv("week2_labeling_template.csv", index=False)

print(f"\n{'─' * 60}")
print(f"FILES SAVED")
print(f"{'─' * 60}")
print(f"  week2_news_raw.csv          — {len(df)} real headlines, full data")
print(f"  week2_labeling_template.csv — same headlines, empty label column")

print(f"\n{'=' * 60}")
print(f"  WEEK 2 COMPLETE")
print(f"{'=' * 60}")
print(f"""
What you just built:
  ✓ Connected to a real financial news API
  ✓ Pulled {len(df)} real headlines across {len(SEARCH_QUERIES)} topics
  ✓ Cleaned and deduplicated the data
  ✓ Saved a labeling template for manual sentiment tagging

Your Week 2 homework:
  1. Open week2_labeling_template.csv in Excel/Google Sheets
  2. For each headline, fill sentiment_label with:
     "positive", "negative", or "neutral"
  3. Label at least 30 headlines
  4. This manual labeling is exactly what FinBERT will
     automate in Week 4 — you're doing it by hand first
     so you understand what "good" sentiment labeling looks like

Next: Week 3 — automate this into an hourly news pipeline
""")
