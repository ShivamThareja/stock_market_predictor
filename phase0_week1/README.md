# Stock Market Predictor

A global financial news → Indian stock market predictor.

Reads financial news from around the world, uses AI to understand
sentiment, and predicts how Indian stocks (NIFTY50) will react —
both from same-timezone Indian news (RBI, NSE announcements) and
cross-timezone overnight news (US Fed, Nikkei, global markets) —
before NSE opens each morning.

## Project status
Currently building Phase 0 — learning the fundamentals with
real Indian stock market data and financial news APIs.

## Tech stack
Python, yfinance, pandas, matplotlib, seaborn, NewsAPI, FinBERT
(coming in Phase 1), XGBoost (coming in Phase 3), Streamlit
(coming in Phase 1)

## Structure
- `phase0_week1/` — Stock data exploration, OHLCV, correlations
- `phase0_week2/` — Financial news fetching and labeling

Built incrementally, one phase at a time, by a BTech CSE student
learning finance + ML + NLP through this project.
