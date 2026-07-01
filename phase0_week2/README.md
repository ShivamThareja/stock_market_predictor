# Week 2: Financial News Explorer

## What's in this folder

| File | Purpose |
|---|---|
| `setup.py` | Run once — installs `requests` and `pandas` |
| `config.py` | Paste your NewsAPI key here |
| `week2_news.py` | Main script — pulls real news headlines |

## How to run

### Step 1 — Install dependencies
```bash
python3 setup.py
```

### Step 2 — Add your API key
Open `config.py` in any text editor:
```bash
nano config.py
```
Replace `PASTE_YOUR_API_KEY_HERE` with your real key from newsapi.org, keeping the quotes:
```python
NEWS_API_KEY = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
```
Save and exit.

### Step 3 — Run the script
```bash
python3 week2_news.py
```

## What the script does

1. Tests your API key works
2. Pulls news for 6 topics: RBI, NIFTY, Indian Banking, Indian IT, US Fed, Global Markets
3. Removes duplicate headlines
4. Prints a summary — articles per topic, articles per source
5. Shows 5 sample real headlines
6. Saves two CSV files

## Files created after running

| File | Purpose |
|---|---|
| `week2_news_raw.csv` | All headlines with full metadata |
| `week2_labeling_template.csv` | Same headlines, empty label column for your homework |

## Your homework

Open `week2_labeling_template.csv` in Excel or Google Sheets.
For each headline, fill the `sentiment_label` column with:
- `positive` — good news for markets
- `negative` — bad news for markets
- `neutral` — no clear market impact

Label at least 30 headlines by hand. This teaches you what FinBERT
will automate in Week 4 — you should understand what good labeling
looks like before letting an AI do it automatically.

## Common errors

| Error | Fix |
|---|---|
| "API key is invalid" | Check config.py — make sure you pasted the key inside the quotes |
| "No articles were retrieved" | Check your internet connection |
| Rate limit / 429 error | You've hit the 1000 req/day limit — wait until tomorrow |
| ModuleNotFoundError: requests | Run setup.py again |

## Next: Week 3
Automate this into a pipeline that runs every hour automatically,
using APScheduler and storing data in SQLite.
