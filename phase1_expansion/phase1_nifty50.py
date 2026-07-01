"""
PHASE 1 — Expand to all NIFTY50 stocks + sector indices
Project: Global Financial News → Indian Stock Market Predictor

What changed from Week 1:
  - 6 tickers → ~55 tickers (all 50 NIFTY50 stocks + 5 sector indices)
  - Same code structure, same decisions (2015 start, 90day window, 0.5% noise)
  - Added sector grouping so news can be mapped to sectors automatically
  - Correlation matrix now shows sector clustering clearly

Run: python3 phase1_nifty50.py
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DECISIONS — same as Week 1, carried forward
# ─────────────────────────────────────────────

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ROLLING_WINDOW = 90
NOISE_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# ALL 50 NIFTY50 STOCKS — organized by sector
# ─────────────────────────────────────────────
# This sector grouping is what powers Phase 2's
# "news → sector → stocks" automatic mapping.

SECTOR_MAP = {
    "Banking": {
        "HDFCBANK.NS":   "HDFC Bank",
        "ICICIBANK.NS":  "ICICI Bank",
        "SBIN.NS":       "SBI",
        "KOTAKBANK.NS":  "Kotak Bank",
        "AXISBANK.NS":   "Axis Bank",
        "INDUSINDBK.NS": "IndusInd Bank",
    },
    "IT": {
        "TCS.NS":     "TCS",
        "INFY.NS":    "Infosys",
        "WIPRO.NS":   "Wipro",
        "HCLTECH.NS": "HCL Tech",
        "TECHM.NS":   "Tech Mahindra",
    },
    "Energy": {
        "RELIANCE.NS": "Reliance",
        "ONGC.NS":     "ONGC",
        "BPCL.NS":     "BPCL",
        "COALINDIA.NS":"Coal India",
        "NTPC.NS":     "NTPC",
        "POWERGRID.NS":"Power Grid",
    },
    "Auto": {
        "TATAMOTORS.NS": "Tata Motors",
        "MARUTI.NS":     "Maruti Suzuki",
        "EICHERMOT.NS":  "Eicher Motors",
        "HEROMOTOCO.NS": "Hero MotoCorp",
        "BAJAJ-AUTO.NS": "Bajaj Auto",
        "M&M.NS":        "Mahindra & Mahindra",
    },
    "Pharma": {
        "SUNPHARMA.NS": "Sun Pharma",
        "DRREDDY.NS":   "Dr Reddy's",
        "CIPLA.NS":     "Cipla",
        "DIVISLAB.NS":  "Divi's Labs",
        "APOLLOHOSP.NS":"Apollo Hospitals",
    },
    "FMCG": {
        "HINDUNILVR.NS": "HUL",
        "ITC.NS":        "ITC",
        "NESTLEIND.NS":  "Nestle India",
        "BRITANNIA.NS":  "Britannia",
        "TATACONSUM.NS": "Tata Consumer",
    },
    "Finance_NBFC": {
        "BAJFINANCE.NS":  "Bajaj Finance",
        "BAJAJFINSV.NS":  "Bajaj Finserv",
        "SBILIFE.NS":     "SBI Life",
        "HDFCLIFE.NS":    "HDFC Life",
    },
    "Materials": {
        "JSWSTEEL.NS":   "JSW Steel",
        "TATASTEEL.NS":  "Tata Steel",
        "HINDALCO.NS":   "Hindalco",
        "ULTRACEMCO.NS": "UltraTech Cement",
        "GRASIM.NS":     "Grasim",
        "SHREECEM.NS":   "Shree Cement",
    },
    "Consumer_Other": {
        "ASIANPAINT.NS": "Asian Paints",
        "TITAN.NS":      "Titan",
        "UPL.NS":        "UPL",
    },
    "Conglomerate_Infra": {
        "LT.NS":         "Larsen & Toubro",
        "ADANIENT.NS":   "Adani Enterprises",
        "ADANIPORTS.NS": "Adani Ports",
        "BHARTIARTL.NS": "Bharti Airtel",
    },
}

# Sector indices — track the sector as a whole, not just individual stocks
SECTOR_INDICES = {
    "^NSEBANK":   "Nifty Bank Index",
    "^CNXIT":     "Nifty IT Index",
    "^CNXAUTO":   "Nifty Auto Index",
    "^CNXPHARMA": "Nifty Pharma Index",
    "^CNXFMCG":   "Nifty FMCG Index",
}

# Build the flat list of all tickers (for yfinance download)
ALL_STOCK_TICKERS = {}
for sector, stocks in SECTOR_MAP.items():
    ALL_STOCK_TICKERS.update(stocks)

# Reverse map: ticker → sector (used to tag news events later)
TICKER_TO_SECTOR = {}
for sector, stocks in SECTOR_MAP.items():
    for ticker in stocks:
        TICKER_TO_SECTOR[ticker] = sector

ALL_TICKERS = {**{"^NSEI": "NIFTY 50 Index"}, **ALL_STOCK_TICKERS, **SECTOR_INDICES}

print("=" * 60)
print("  PHASE 1: Expanding to NIFTY50 + Sector Indices")
print("=" * 60)
print(f"\nTotal tickers to download: {len(ALL_TICKERS)}")
print(f"  - 1 main index (NIFTY50)")
print(f"  - {len(ALL_STOCK_TICKERS)} individual stocks across {len(SECTOR_MAP)} sectors")
print(f"  - {len(SECTOR_INDICES)} sector indices")


# ─────────────────────────────────────────────
# STEP 1: Download all data (chunked — yfinance
# struggles with 50+ tickers in one call)
# ─────────────────────────────────────────────

print(f"\nDownloading data from {START_DATE} to {END_DATE}...")
print("This will take 1-2 minutes for 55 tickers...\n")

all_tickers_list = list(ALL_TICKERS.keys())
chunk_size = 15  # yfinance is more reliable in smaller batches
close_prices_parts = []

for i in range(0, len(all_tickers_list), chunk_size):
    chunk = all_tickers_list[i:i + chunk_size]
    print(f"  Downloading batch {i//chunk_size + 1}: {chunk}")
    try:
        raw = yf.download(
            tickers=chunk,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
            group_by="ticker"
        )
        if len(chunk) == 1:
            # single ticker returns flat columns
            part = raw[["Close"]].copy()
            part.columns = [chunk[0]]
        else:
            part = pd.DataFrame({t: raw[t]["Close"] for t in chunk if t in raw.columns.get_level_values(0)})
        close_prices_parts.append(part)
    except Exception as e:
        print(f"    Warning: batch failed ({e}), skipping")

close_prices = pd.concat(close_prices_parts, axis=1)
close_prices.columns = [ALL_TICKERS.get(c, c) for c in close_prices.columns]
close_prices = close_prices.dropna(how="all")

print(f"\nData downloaded successfully.")
print(f"Shape: {close_prices.shape}")
print(f"Date range: {close_prices.index[0].date()}  →  {close_prices.index[-1].date()}")
print(f"Tickers successfully loaded: {close_prices.shape[1]} / {len(ALL_TICKERS)}")


# ─────────────────────────────────────────────
# STEP 2: Daily returns + noise filter
# ─────────────────────────────────────────────

daily_returns = close_prices.pct_change() * 100
daily_returns = daily_returns.dropna(how="all")

print("\n" + "─" * 60)
print("SECTOR-LEVEL SUMMARY — average daily volatility per sector:")
print("─" * 60)

for sector, stocks in SECTOR_MAP.items():
    sector_names = [v for v in stocks.values() if v in daily_returns.columns]
    if sector_names:
        avg_vol = daily_returns[sector_names].std().mean()
        print(f"  {sector:<22} avg volatility: {avg_vol:.2f}%  ({len(sector_names)} stocks)")


# ─────────────────────────────────────────────
# STEP 3: Sector correlation — which sectors
# move together? (foundation for contagion matrix)
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("SECTOR AVERAGE RETURNS — daily correlation with NIFTY50:")
print("─" * 60)

if "NIFTY 50 Index" in daily_returns.columns:
    nifty_returns = daily_returns["NIFTY 50 Index"]
    for sector, stocks in SECTOR_MAP.items():
        sector_names = [v for v in stocks.values() if v in daily_returns.columns]
        if sector_names:
            sector_avg = daily_returns[sector_names].mean(axis=1)
            corr_with_nifty = sector_avg.corr(nifty_returns)
            print(f"  {sector:<22} correlation with NIFTY50: {corr_with_nifty:.2f}")


# ─────────────────────────────────────────────
# STEP 4: Save sector-tagged data for Phase 2
# ─────────────────────────────────────────────

close_prices.to_csv("phase1_close_prices_all.csv")
daily_returns.to_csv("phase1_daily_returns_all.csv")

# Save the sector mapping — Phase 2 needs this to
# automatically route news events to affected stocks
sector_lookup = pd.DataFrame([
    {"ticker": t, "name": n, "sector": TICKER_TO_SECTOR.get(t, "Index")}
    for t, n in ALL_TICKERS.items()
])
sector_lookup.to_csv("phase1_sector_map.csv", index=False)

print("\nFiles saved:")
print("  phase1_close_prices_all.csv  — all 55 tickers, 10yr prices")
print("  phase1_daily_returns_all.csv — all 55 tickers, daily returns")
print("  phase1_sector_map.csv        — ticker → sector lookup table")


# ─────────────────────────────────────────────
# STEP 5: Generate sector heatmap chart
# ─────────────────────────────────────────────

print("\nGenerating sector correlation chart...")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 10))

# Build sector-average returns table
sector_avg_returns = pd.DataFrame()
for sector, stocks in SECTOR_MAP.items():
    sector_names = [v for v in stocks.values() if v in daily_returns.columns]
    if sector_names:
        sector_avg_returns[sector] = daily_returns[sector_names].mean(axis=1)

if "NIFTY 50 Index" in daily_returns.columns:
    sector_avg_returns["NIFTY50"] = daily_returns["NIFTY 50 Index"]

sector_corr = sector_avg_returns.corr()

sns.heatmap(
    sector_corr, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=0, vmax=1,
    linewidths=0.5, square=True, ax=ax
)
ax.set_title(
    "Sector-level correlation — which sectors move together",
    fontsize=13, fontweight="bold", pad=12
)
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("phase1_sector_correlation.png", dpi=150, bbox_inches="tight")
print("Chart saved → phase1_sector_correlation.png")
plt.show()

print("\n" + "=" * 60)
print("  PHASE 1 EXPANSION COMPLETE")
print("=" * 60)
print(f"""
What changed from Week 1:
  Tickers tracked: 6 → {close_prices.shape[1]}
  Sectors covered: 3 → {len(SECTOR_MAP)}
  New capability: news → sector → stocks automatic routing

This sector_map.csv is critical for Phase 2 — when a news
headline mentions "RBI" or "repo rate", the system will look
up "Banking" sector and automatically flag HDFC Bank, ICICI
Bank, SBI, Kotak, Axis, and IndusInd Bank — all at once.
""")
