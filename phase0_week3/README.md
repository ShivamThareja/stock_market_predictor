# Week 3: Automated News Pipeline

## What's in this folder

| File | Purpose |
|---|---|
| `setup.py` | Run once — installs `requests` and `apscheduler` |
| `config.py` | Paste your NewsAPI key here (same key as Week 2) |
| `week3_pipeline.py` | Main script — fetches news automatically on a schedule |

## What changed from Week 2

- CSV overwritten every run → SQLite database (`news_pipeline.db`), append-only
- Manual "run it when I remember" → APScheduler runs the fetch automatically every hour
- No duplicate handling → `UNIQUE` constraint on headline means every run can
  safely re-fetch an overlapping time window without creating duplicate rows
- Same 6 topics from Week 2, same `sentiment_label` column — left `NULL` for
  now, Week 4's FinBERT will fill it in automatically

## How to run

### Step 1 — Install dependencies
```bash
python3 setup.py
```

### Step 2 — Add your API key
Open `config.py` and paste in the same NewsAPI key you used for Week 2:
```python
NEWS_API_KEY = "your_real_key_here"
```

### Step 3 — Test with a single run first
```bash
python3 week3_pipeline.py --once
```
This fetches once, saves to the database, and exits — good for confirming
everything works before starting the real hourly loop.

### Step 4 — Run the real pipeline
```bash
python3 week3_pipeline.py
```
This fetches immediately, then again every hour, forever, until you press
`Ctrl+C`. Leave it running in a terminal tab while you work on other things.

If you want it running in the background instead of tying up a terminal:
```bash
nohup python3 week3_pipeline.py > pipeline.log 2>&1 &
```
Check `pipeline.log` to see what it's doing. To stop it later, find the
process (`ps aux | grep week3_pipeline`) and kill it.

## What the script does, each run

1. Pulls the last 2 days of headlines for the same 6 topics as Week 2
2. Tries to insert each one into SQLite — the `UNIQUE` constraint on headline
   means already-seen articles are silently skipped, not duplicated
3. Prints a summary: fetched / new / duplicates skipped / total in database

## Why 2 days back, not "since last run"

Simpler and safer. If your laptop was asleep or the script crashed for a
few hours, a fixed "last 2 days" window means you don't quietly lose
articles — the dedup constraint makes re-fetching the overlap free.

## Inspecting the database

```bash
sqlite3 news_pipeline.db "SELECT topic, COUNT(*) FROM news_articles GROUP BY topic;"
```

Or in Python:
```python
import sqlite3, pandas as pd
conn = sqlite3.connect("news_pipeline.db")
df = pd.read_sql("SELECT * FROM news_articles", conn)
```

## Quota math

6 topics × 1 request each × 24 runs/day = **144 requests/day**.
NewsAPI's free tier allows 1,000/day — plenty of headroom to add more
search topics later without worrying about hitting the limit.

## Common errors

| Error | Fix |
|---|---|
| "API key is invalid" | Check `config.py` — same fix as Week 2 |
| `ModuleNotFoundError: apscheduler` | Run `setup.py` again |
| Pipeline seems to hang | It's not hanging — `BlockingScheduler` is designed to block. That's the point: it's waiting for the next hourly run. Ctrl+C to stop. |
| Script exits immediately with no error | Check you didn't accidentally leave `--once` in the command |

## Next: Week 4
Use FinBERT to automatically fill in `sentiment_label` for every row where
it's still `NULL`, replacing the manual labeling from Week 2. Compare
FinBERT's labels against your hand-labeled Week 2 data to sanity-check
accuracy before trusting it on new headlines.
