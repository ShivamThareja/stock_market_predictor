"""
PHASE 0 — WEEK 1
Project: Global Financial News → Indian Stock Market Predictor
Goal: Pull 10 years of NIFTY50 + top Indian stocks data and understand it

Decisions made:
  - START_DATE = "2015-01-01" (hardcoded, not dynamic timedelta)
  - 10 years = ML training sweet spot (modern Indian market era)
  - Rolling volatility window = 90 days (appropriate for 10yr data)
  - X-axis ticks every 6 months (readable on 10yr chart)
  - Only NIFTY50 large-caps (better news coverage, more reliable signal)
  - Filter: moves below ±0.5% treated as noise (from research findings)

What this file teaches you:
  - How to download real stock data using yfinance
  - What OHLCV means (Open, High, Low, Close, Volume)
  - How to calculate daily returns
  - How to plot professional-looking charts
  - How to read correlations between stocks

Run: python3 week1_stocks.py
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
# DECISIONS — all in one place, easy to change
# ─────────────────────────────────────────────

# Start date: 2015-01-01
# Why 2015: Modern Indian market era — high FII participation,
# algo trading, strong global correlation. Pre-2015 market
# behaved differently and would confuse the ML model.
START_DATE = "2015-01-01"

# End date: today (always pulls the latest available data)
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Rolling volatility window: 90 days
# Why 90: 30-day window is too noisy on 10 years of data.
# 90-day window shows real volatility cycles clearly.
ROLLING_WINDOW = 90

# Noise filter threshold: 0.5%
# Why 0.5: From LSTM paper research — moves below 0.5% are
# statistical noise. We flag them here for awareness.
# Phase 3 ML model will ONLY train on moves above this.
NOISE_THRESHOLD = 0.5

# Tickers: NIFTY50 index + 5 large-cap stocks
# Why these 6: Covers all major sectors (IT, Banking, Energy)
# Large-caps have better news coverage = more reliable signal
# .NS suffix = NSE listed | ^NSEI = NIFTY50 index itself
TICKERS = {
    "^NSEI":       "NIFTY 50 Index",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS":      "TCS",
    "HDFCBANK.NS": "HDFC Bank",
     "INFY.NS":     "Infosys",
    "ICICIBANK.NS":"ICICI Bank",
}

# Colors for charts — one per ticker
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]


# ─────────────────────────────────────────────
# STEP 1: Download the data
# ─────────────────────────────────────────────

print("=" * 60)
print("  PHASE 0 — WEEK 1: Stock Data Explorer")
print("  Project: Global News → Indian Stock Predictor")
print("=" * 60)
print(f"\nDownloading data from {START_DATE} to {END_DATE}...")
print(f"Tickers: {list(TICKERS.keys())}\n")

raw = yf.download(
    tickers=list(TICKERS.keys()),
    start=START_DATE,
    end=END_DATE,
    progress=False,
    auto_adjust=True,  # adjusts for splits and dividends
)

# Extract Close prices into a clean DataFrame
close_prices = raw["Close"].copy()
close_prices.columns = [TICKERS[t] for t in close_prices.columns]

# Drop rows where ALL values are NaN (market holidays)
close_prices = close_prices.dropna(how="all")

print(f"Data downloaded successfully.")
print(f"Shape: {close_prices.shape}")
print(f"Date range: {close_prices.index[0].date()}  →  {close_prices.index[-1].date()}")
print(f"Trading days: {len(close_prices)}")
print(f"Years of data: {round(len(close_prices) / 252, 1)} years")
print(f"\nFirst 3 rows:")
print(close_prices.head(3).to_string())
print(f"\nLast 3 rows:")
print(close_prices.tail(3).to_string())


# ─────────────────────────────────────────────
# STEP 2: Understand OHLCV for one stock (TCS)
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("OHLCV EXPLAINED — TCS latest trading day:")
print("─" * 60)

tcs_full = yf.download(
    "TCS.NS",
    start=START_DATE,
    end=END_DATE,
    progress=False,
    auto_adjust=True
)
# drop rows where Close is NaN (incomplete trading days)
tcs_full = tcs_full.dropna(subset=[("Close", "TCS.NS")] 
                           if isinstance(tcs_full.columns, pd.MultiIndex) 
                           else ["Close"])
latest = tcs_full.iloc[-1]

print(f"  Date   {tcs_full.index[-1].date()}")
print(f"  Open   ₹{float(latest['Open'].iloc[0]):.2f}   — price when NSE opened at 9:15 AM IST")
print(f"  High   ₹{float(latest['High'].iloc[0]):.2f}   — highest price touched that day")
print(f"  Low    ₹{float(latest['Low'].iloc[0]):.2f}   — lowest price touched that day")
print(f"  Close  ₹{float(latest['Close'].iloc[0]):.2f}  — price when NSE closed at 3:30 PM IST")
print(f"  Volume {int(latest['Volume'].iloc[0]):,}   — shares bought/sold that day")


# ─────────────────────────────────────────────
# STEP 3: Calculate daily returns
# ─────────────────────────────────────────────

# pct_change() = (today - yesterday) / yesterday * 100
# This tells you how much each stock moved each day
daily_returns = close_prices.pct_change() * 100
daily_returns = daily_returns.dropna()

print("\n" + "─" * 60)
print("DAILY RETURNS — Summary statistics (10 years):")
print("─" * 60)

stats = daily_returns.describe().T[["mean", "std", "min", "max"]]
stats.columns = ["Avg daily %", "Volatility %", "Worst day %", "Best day %"]
stats = stats.round(3)
print(stats.to_string())

# Noise filter check
print(f"\nNoise filter (±{NOISE_THRESHOLD}%):")
for col in daily_returns.columns:
    meaningful = (daily_returns[col].abs() > NOISE_THRESHOLD).sum()
    total = len(daily_returns[col].dropna())
    pct = round(meaningful / total * 100, 1)
    print(f"  {col:<25} {meaningful}/{total} days had moves > ±{NOISE_THRESHOLD}%  ({pct}%)")

print(f"\nNote: Phase 3 ML model will ONLY train on the {NOISE_THRESHOLD}%+ move days.")
print("Smaller moves are noise — confirmed by LSTM research paper.")

# ─────────────────────────────────────────────
# STEP 4: Normalized performance comparison
# ─────────────────────────────────────────────

# Problem: NIFTY is at ~22,000 but HDFC Bank is at ~800
# Can't compare directly. Solution: normalize to base 100.
#if pd.isna(perf):
first_valid = close_prices.apply(lambda col: col.dropna().iloc[0])
normalized = (close_prices / first_valid) * 100
final = (normalized.iloc[-1] - 100).round(1).sort_values(ascending=False)

print("\n" + "─" * 60)
print("NORMALIZED PERFORMANCE (base = 100 on 2015-01-01):")
print("─" * 60)

for name, perf in final.items():
    sign = "+" if (not pd.isna(perf) and perf > 0) else ""
    bar = "█" * min(int(abs(perf) / 10), 30) if not pd.isna(perf) else ""
    if pd.isna(perf):
        print(f"  {name:<25} no data on 2015-01-01 (market holiday)")
    else:
        print(f"  {name:<25} {sign}{perf:.1f}%  {bar}")


# ─────────────────────────────────────────────
# STEP 5: Correlation analysis
# ─────────────────────────────────────────────


corr_matrix = daily_returns.corr()

print("\n" + "─" * 60)
print("CORRELATION MATRIX (daily returns):")
print("─" * 60)
print(corr_matrix.round(2).to_string())
print("\nKey insight: TCS + Infosys should be ~0.8+ (both IT stocks)")
print("This is the foundation of cross-market prediction in Phase 3.")


# ─────────────────────────────────────────────
# STEP 6: Generate 5 charts
# ─────────────────────────────────────────────

print("\nGenerating charts...")

sns.set_theme(style="whitegrid", palette="husl")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#FAFAFA")

# ── Chart 1: Normalized performance (full top row) ──
ax1 = fig.add_subplot(3, 2, (1, 2))
first_valid = close_prices.apply(lambda col: col.dropna().iloc[0])
normalized = (close_prices / first_valid) * 100

for i, col in enumerate(normalized.columns):
    ax1.plot(
        normalized.index,
        normalized[col],
        label=col,
        color=COLORS[i],
        linewidth=1.8
    )
ax1.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_title(
    "Normalized price performance — 10 years (base = 100 on Jan 2015)",
    fontsize=13, fontweight="bold", pad=12
)
ax1.set_ylabel("Indexed price (start = 100)")
ax1.legend(loc="upper left", fontsize=9, ncol=3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

# Annotate major Indian market events
events = {
    "2016-11-08": "Demonetisation",
    "2020-03-23": "Covid crash",
    "2022-02-24": "Russia-Ukraine",
    "2024-06-04": "Election result",
}
for date_str, label in events.items():
    try:
        event_date = pd.Timestamp(date_str)
        ax1.axvline(
            x=event_date, color="red",
            linestyle=":", linewidth=0.8, alpha=0.6
        )
        ax1.text(
            event_date, ax1.get_ylim()[1] * 0.95,
            label, fontsize=7, color="red",
            rotation=90, va="top", ha="right"
        )
    except Exception:
        pass


# ── Chart 2: NIFTY50 price with area fill ──
ax2 = fig.add_subplot(3, 2, 3)
nifty_data = yf.download(
    "^NSEI",
    start=START_DATE,
    end=END_DATE,
    progress=False,
    auto_adjust=True
)
ax2.plot(
    nifty_data.index, nifty_data["Close"],
    color="#2563EB", linewidth=1.8
)

close_vals = nifty_data["Close"].squeeze()
ax2.fill_between(
    nifty_data.index, close_vals,
    close_vals.min(),
    alpha=0.08, color="#2563EB"
)

ax2.set_title(
    "NIFTY 50 index — 10 years (2015–2025)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Index value")
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)


# ── Chart 3: Daily returns histogram (NIFTY) ──
ax3 = fig.add_subplot(3, 2, 4)
nifty_returns = nifty_data["Close"].pct_change().dropna() * 100
ax3.hist(
    nifty_returns, bins=80,
    color="#2563EB", alpha=0.7, edgecolor="white"
)
ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
ax3.axvline(
    x=nifty_returns.mean().iloc[0],
    color="red", linestyle="--", linewidth=1.2,
    label=f"Mean: {nifty_returns.mean().iloc[0]:.3f}%"
)
ax3.axvline(
    x=NOISE_THRESHOLD, color="orange",
    linestyle=":", linewidth=1,
    label=f"Noise threshold: ±{NOISE_THRESHOLD}%"
)
ax3.axvline(
    x=-NOISE_THRESHOLD, color="orange",
    linestyle=":", linewidth=1
)
ax3.set_title(
    "NIFTY 50 — daily returns distribution (10 years)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Daily return (%)")
ax3.set_ylabel("Number of days")
ax3.legend(fontsize=8)


# ── Chart 4: Correlation heatmap ──
ax4 = fig.add_subplot(3, 2, 5)
mask = pd.DataFrame(
    False,
    index=corr_matrix.index,
    columns=corr_matrix.columns
)
for i in range(len(mask)):
    for j in range(i + 1, len(mask.columns)):
        mask.iloc[i, j] = True

sns.heatmap(
    corr_matrix, ax=ax4,
    annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=0, vmax=1,
    mask=mask,
    linewidths=0.5, square=True,
    annot_kws={"size": 8}
)
ax4.set_title(
    "Return correlations — foundation of cross-market prediction",
    fontsize=11, fontweight="bold"
)
ax4.set_xticklabels(
    ax4.get_xticklabels(), rotation=40, ha="right", fontsize=8
)
ax4.set_yticklabels(
    ax4.get_yticklabels(), rotation=0, fontsize=8
)


# ── Chart 5: 90-day rolling volatility ──
ax5 = fig.add_subplot(3, 2, 6)
for i, col in enumerate(daily_returns.columns):
    rolling_vol = daily_returns[col].rolling(window=ROLLING_WINDOW).std()
    ax5.plot(
        rolling_vol.index, rolling_vol,
        label=col, color=COLORS[i], linewidth=1.5
    )
ax5.set_title(
    f"{ROLLING_WINDOW}-day rolling volatility — which periods were volatile?",
    fontsize=11, fontweight="bold"
)
ax5.set_ylabel("Volatility (std of daily returns %)")
ax5.legend(fontsize=8, loc="upper right")
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax5.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30)


plt.suptitle(
    "Phase 0 — Week 1: Indian Stock Market Explorer (10 Years, 2015–2025)",
    fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout(pad=2.5)

output_file = "week1_charts.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
print(f"Charts saved → {output_file}")
plt.show()


# ─────────────────────────────────────────────
# STEP 7: Save CSVs for Week 2 onward
# ─────────────────────────────────────────────

close_prices.to_csv("week1_close_prices.csv")
daily_returns.to_csv("week1_daily_returns.csv")
corr_matrix.to_csv("week1_correlations.csv")

print("\nCSV files saved:")
print("  week1_close_prices.csv   — 10yr close prices (use in Phase 2+)")
print("  week1_daily_returns.csv  — daily % changes (use in Phase 3 ML)")
print("  week1_correlations.csv   — correlation matrix (use in Phase 3)")

print("\n" + "=" * 60)
print("  WEEK 1 COMPLETE")
print("=" * 60)
print(f"""
What you just built:
  ✓ 10 years of real NSE/NIFTY data downloaded
  ✓ OHLCV understood for TCS
  ✓ Daily returns calculated for all 6 tickers
  ✓ Noise threshold identified (±{NOISE_THRESHOLD}%)
  ✓ Correlations between all stocks measured
  ✓ 5 charts generated with major event markers
  ✓ 3 CSV files saved for use in later phases

Key numbers to remember:
  Trading days downloaded : {len(close_prices)}
  Years of data           : {round(len(close_prices) / 252, 1)}
  Start date              : {START_DATE}  (hardcoded — reproducible)
  Rolling window used     : {ROLLING_WINDOW} days
  Noise threshold         : ±{NOISE_THRESHOLD}%

Your Week 1 homework:
  1. Open week1_charts.png
  2. Find the single biggest drop on the NIFTY chart
  3. Google that date — find what news caused it
  4. That manual step IS the core idea of this project
     (Phase 2 will automate this at scale)

Next: Week 2 — pull financial news headlines via NewsAPI
""")
