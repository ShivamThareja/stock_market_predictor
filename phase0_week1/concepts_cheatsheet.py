"""
FINANCE CONCEPTS — WEEK 1 CHEATSHEET
Read this before running week1_stocks.py. Takes 5 minutes.
These concepts are the foundation of the entire project.
"""

# ─────────────────────────────────────────────────────
# WHAT IS NIFTY 50?
# ─────────────────────────────────────────────────────
# NIFTY 50 = index tracking the 50 largest companies on NSE
# Think of it like an "average score" of Indian market health
# When people say "market is up 1%" they mean NIFTY is up 1%
#
# yfinance ticker : "^NSEI"
# NSE stocks      : add ".NS" → "RELIANCE.NS", "TCS.NS"
# BSE stocks      : add ".BO" → "RELIANCE.BO"
#
# Global equivalents:
#   S&P 500    = US version of NIFTY (ticker: ^GSPC)
#   FTSE 100   = UK version (ticker: ^FTSE)
#   Nikkei 225 = Japan version (ticker: ^N225)
# These are what Phase 3 will monitor overnight to
# predict NIFTY's opening direction next morning.


# ─────────────────────────────────────────────────────
# WHAT IS OHLCV?
# ─────────────────────────────────────────────────────
# Every trading day produces 5 numbers per stock:
#
# O — Open   = price at 9:15 AM IST when NSE opens
# H — High   = highest price touched all day
# L — Low    = lowest price touched all day
# C — Close  = price at 3:30 PM IST when NSE closes  ← most important
# V — Volume = how many shares were bought/sold
#
# Example for TCS on a random day:
#   Open:   3,820  ← opened here
#   High:   3,891  ← shot up to here at some point
#   Low:    3,798  ← fell to here at some point
#   Close:  3,865  ← ended here  ← this is what we use


# ─────────────────────────────────────────────────────
# WHAT IS DAILY RETURN?
# ─────────────────────────────────────────────────────
# How much did the stock move today vs yesterday?
# Formula: (today_close - yesterday_close) / yesterday_close × 100
#
# Example:
#   Yesterday TCS closed at 3,800
#   Today TCS closed at 3,876
#   Daily return = (3876 - 3800) / 3800 × 100 = +2.0%
#
# In pandas: daily_returns = close_prices.pct_change() * 100
#
# PROJECT CONNECTION:
# This is what your Phase 3 model will PREDICT.
# "Given tonight's global news, will tomorrow's return be
# positive (UP), negative (DOWN), or near-zero (FLAT)?"


# ─────────────────────────────────────────────────────
# WHAT IS THE ±0.5% NOISE THRESHOLD?
# ─────────────────────────────────────────────────────
# Research finding from the LSTM paper we studied:
# Moves smaller than ±0.5% are statistical noise.
# They're caused by random trading, not real news events.
# Training the ML model on noise = teaching it random patterns.
#
# So in Phase 3 we ONLY train on days where:
#   daily_return > +0.5%  → label as UP
#   daily_return < -0.5%  → label as DOWN
#   between ±0.5%         → label as FLAT (or discard)
#
# This makes the model learn from REAL signal only.
# Directly improves prediction accuracy.


# ─────────────────────────────────────────────────────
# WHAT IS VOLATILITY?
# ─────────────────────────────────────────────────────
# How much does the stock's price jump around?
# Measured as: standard deviation of daily returns
#
# Low volatility (~0.5% std)  → stable (HDFC Bank on normal days)
# High volatility (~2-3% std) → wild (any stock during Covid crash)
#
# PROJECT CONNECTION:
# High volatility periods = markets reacting to news.
# The 90-day rolling volatility chart shows EXACTLY when
# news drove the market. Those are the patterns your model learns.
#
# We use 90-day rolling window (not 30-day) because:
# On 10 years of data, 30 days is too noisy.
# 90 days smooths out daily noise and shows real cycles.


# ─────────────────────────────────────────────────────
# WHAT IS CORRELATION?
# ─────────────────────────────────────────────────────
# Do two stocks move together?
# Range: -1.0 (opposite) to +1.0 (identical)
#
#  0.85 = move together strongly (TCS + Infosys — both IT)
#  0.50 = somewhat related
#  0.10 = mostly independent
# -0.30 = tend to move opposite
#
# PROJECT CONNECTION — this is CRITICAL:
# If TCS and NASDAQ have correlation 0.75:
# → When NASDAQ falls at night, TCS likely opens lower
# → That is your cross-timezone prediction logic
#
# If HDFC Bank and ICICI Bank have correlation 0.88:
# → Bad news for one = warning for both
# → That is your sector contagion matrix


# ─────────────────────────────────────────────────────
# KEY SECTOR → STOCK MAPPING (used in Phase 2)
# ─────────────────────────────────────────────────────
#
# IT Sector       → TCS, Infosys, Wipro, HCL Tech
#   News triggers : US tech layoffs, NASDAQ moves, USD/INR rate
#
# Banking Sector  → HDFC Bank, ICICI Bank, SBI, Kotak
#   News triggers : RBI repo rate, CRR changes, NPA data
#
# Energy Sector   → Reliance, ONGC, BPCL
#   News triggers : Crude oil price, OPEC decisions
#
# Auto Sector     → Tata Motors, M&M, Maruti
#   News triggers : EV policy, fuel prices, GST changes
#
# Pharma Sector   → Sun Pharma, Dr Reddy's, Cipla
#   News triggers : FDA approvals/rejections, drug pricing
#
# This mapping is what Phase 2 will automate —
# "RBI rate hike" → flag Banking sector → alert HDFC, ICICI, SBI


# ─────────────────────────────────────────────────────
# MAJOR EVENTS VISIBLE IN YOUR 10-YEAR CHART
# ─────────────────────────────────────────────────────
# These are marked with red dotted lines on Chart 1:
#
# Nov 2016  — Demonetisation
#             Modi announced ₹500/₹1000 notes invalid overnight
#             NIFTY fell ~6% next day — pure domestic news event
#             This is a SAME-TIMEZONE prediction scenario
#
# Mar 2020  — Covid crash
#             NIFTY fell 38% in 6 weeks — fastest crash ever
#             US/global markets fell first, India followed
#             This is a CROSS-TIMEZONE prediction scenario
#
# Feb 2022  — Russia-Ukraine war
#             Oil spiked $30 → aviation stocks hit, oil cos gained
#             Happened overnight → affected India market open
#             CROSS-TIMEZONE scenario
#
# Jun 2024  — Indian election results
#             BJP won fewer seats than expected, NIFTY fell 8%
#             Single biggest intraday drop in years
#             SAME-TIMEZONE scenario
#
# YOUR HOMEWORK: Find the EXACT date of the Covid crash low
# on your chart. Google "NIFTY 50 lowest point 2020 news"
# That is one data point of what your project automates.
