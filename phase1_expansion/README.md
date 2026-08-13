# Phase 1: Expand to all NIFTY50 stocks + sector indices

## What's in this folder

| File | Purpose |
|---|---|
| `phase1_nifty50.py` | Main script — downloads all 56 tickers, builds sector summaries and the sector map |

## What changed from Week 1

- 6 tickers → all 50 NIFTY50 stocks + 5 sector indices + the NIFTY50 index itself (56 total)
- Same decisions carried forward: 2015 start date, 90-day rolling volatility window, ±0.5% noise threshold
- New: stocks grouped into 10 sectors (Banking, IT, Energy, Auto, Pharma, FMCG,
  Finance/NBFC, Materials, Consumer, Conglomerate/Infra)
- New: a ticker → sector lookup table, which later phases use to automatically
  route a news headline (e.g. "RBI hikes repo rate") to every stock in the
  affected sector

## How to run

Uses the same dependencies as Week 1 (`yfinance`, `pandas`, `matplotlib`,
`seaborn`) — if you already ran `phase0_week1/setup.py` you're set. Otherwise:

```bash
pip3 install yfinance pandas matplotlib seaborn --break-system-packages
```

Then:

```bash
python3 phase1_nifty50.py
```

## What the script does

1. Downloads 10+ years of price data for all 56 tickers, in chunks of 15
   (yfinance is unreliable with 50+ tickers in a single call)
2. Reports which tickers actually loaded — a ticker can silently fail
   (delisted, renamed) and leave an all-NaN column, so the script checks
   for real data per ticker rather than just counting columns
3. Prints average daily volatility per sector
4. Prints each sector's correlation with the NIFTY50 index
5. Saves 3 CSVs + a sector correlation heatmap

## Files created after running

| File | Purpose |
|---|---|
| `phase1_close_prices_all.csv` | All 56 tickers, 10yr close prices |
| `phase1_daily_returns_all.csv` | All 56 tickers, daily % returns |
| `phase1_sector_map.csv` | ticker → sector lookup — used by Phase 2's news routing |
| `phase1_sector_correlation.png` | Sector-level correlation heatmap |

## Known ticker changes

`TATAMOTORS.NS` was renamed to `TMPV.NS` (Tata Motors Passenger Vehicles)
after the 2025 commercial/passenger vehicle demerger. `TMPV.NS` carries the
full price history back to 2015, so it's used in place of the old symbol.
If yfinance flags other tickers as delisted in the future, check for a
renamed successor the same way before assuming the sector's stock count
should just shrink.

## Next: Phase 2

FinBERT-based sentiment scoring on the automated news pipeline (Week 3),
using `phase1_sector_map.csv` to route headlines to affected stocks.
