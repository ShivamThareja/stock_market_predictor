"""
PHASE 1 (continued): Global macro data
Project: Global Financial News → Indian Stock Market Predictor
Goal: Pull the small set of cross-market series Phase 3 uses as
      "overnight signal" features — the same cross-timezone idea
      the project's README describes (US close -> India open) but
      as price data instead of news.

Series tracked:
  S&P 500   (^GSPC)     - US market close, the overnight signal
  Crude Oil (CL=F)      - WTI futures, moves Energy/transport stocks
  USD/INR   (INR=X)     - currency, moves IT/export-heavy sectors
  India VIX (^INDIAVIX) - local volatility/fear gauge

Kept separate from phase1_close_prices_all.csv on purpose: these
aren't NIFTY50 constituents or sector indices (Phase 1's actual
subject), they're auxiliary market context Phase 3 joins in by date.

Run: python3 phase1_macro_data.py
"""

import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

MACRO_TICKERS = {
    "^GSPC": "S&P 500",
    "CL=F": "Crude Oil",
    "INR=X": "USD/INR",
    "^INDIAVIX": "India VIX",
}

print("=" * 60)
print("  PHASE 1: Macro data (S&P 500, crude oil, USD/INR, India VIX)")
print("=" * 60)

print(f"\nDownloading {len(MACRO_TICKERS)} series from {START_DATE} to {END_DATE}...")

parts = []
loaded, failed = [], []
for ticker, name in MACRO_TICKERS.items():
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if raw.empty or "Close" not in raw.columns:
            failed.append(name)
            continue
        series = raw["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series.name = name
        parts.append(series)
        loaded.append(name)
    except Exception as e:
        print(f"  Warning: {name} ({ticker}) failed: {e}")
        failed.append(name)

macro = pd.concat(parts, axis=1) if parts else pd.DataFrame()
macro = macro.dropna(how="all")
macro.index.name = "Date"

print(f"\nLoaded: {loaded}")
if failed:
    print(f"Failed (skipped, feature_engineering.py handles missing series gracefully): {failed}")
print(f"Shape: {macro.shape}")
if not macro.empty:
    print(f"Date range: {macro.index.min().date()} -> {macro.index.max().date()}")

macro.to_csv("phase1_macro_data.csv")
print("\nSaved -> phase1_macro_data.csv")

print("\n" + "=" * 60)
print("  DONE — phase3_prediction/feature_engineering.py merges this")
print("  into the price table it already uses for stock momentum.")
print("=" * 60)
