# Stock Market Predictor

> Global financial news → Indian stock market direction predictor

A complete, end-to-end financial intelligence system that reads news from around the world 24/7, uses AI to understand sentiment, and predicts how Indian stocks (NIFTY50) will react — both from same-timezone Indian news (RBI, NSE announcements) and cross-timezone overnight news (US Fed, Nikkei, global markets) — before NSE opens each morning.

---

## Live results

| Metric | Value |
|---|---|
| Headlines collected | 343 labeled headlines |
| Stock-news pairs routed | 973 (headline, stock) pairs |
| Tickers tracked | 56 (all NIFTY50 + sector indices) |
| Model accuracy | 42.2% (+8.9pts above 33.3% random baseline) |
| Training data | ~28 trading days (accuracy improves as pipeline runs longer) |

> **Honest note on accuracy:** 42.2% on 28 days of data is a proof of concept, not a finished result. The pipeline runs hourly — as it accumulates more real trading days across varied market conditions (crashes, rallies, RBI surprises), the model retrains and accuracy improves passively. Renaissance Technologies achieves ~66% after 30 years. We're being honest about where we are.

---

## What it does

```
Global News (24/7)
       ↓
week3_pipeline.py     — fetches headlines every hour from 8+ sources
       ↓
week4_finbert.py      — FinBERT AI labels each headline: positive / negative / neutral
       ↓
phase2_routing.py     — maps each headline to affected NIFTY50 stocks and sectors
       ↓
phase3_xgboost.py     — XGBoost model predicts: UP / DOWN / FLAT for next session
       ↓
Streamlit Dashboard   — live view of pipeline, NIFTY chart, model accuracy
```

---

## Data sources (8+ active)

| Source | What it provides |
|---|---|
| NewsAPI.org | 1,000 req/day, broad global financial news |
| Bloomberg RSS | Free premium headlines — most market-moving |
| Reuters RSS | Breaking global financial news |
| Economic Times RSS | India-specific market news |
| RBI RSS | Direct repo rate decisions — faster than any news article |
| Mint RSS | Indian business news |
| yfinance | 10 years of NSE/BSE/global price data |
| GDELT (Phase 2) | Historical news archive, 65 languages |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data collection | `yfinance`, `NewsAPI`, `feedparser`, `requests` |
| Scheduling | `APScheduler` — runs hourly, 24/7 |
| Storage | `SQLite` — all headlines, prices, predictions |
| NLP / Sentiment | `FinBERT` (ProsusAI/finbert) via HuggingFace Transformers |
| Entity extraction | `spaCy` — identifies which company/sector is mentioned |
| ML prediction | `XGBoost` — gradient boosted trees |
| Dashboard | `Streamlit` |
| Language | Python 3.12 |

---

## Project structure

```
stock_market_predictor/
├── phase0_week1/          # Stock data exploration — 10yr NIFTY50 data, OHLCV, correlations
├── phase0_week2/          # News fetching — NewsAPI integration, manual sentiment labeling
├── phase0_week3/          # Automated pipeline — APScheduler, SQLite, multi-source RSS
├── phase0_week4/          # FinBERT sentiment — AI labeling, validation vs human labels
├── phase1_expansion/      # All 56 NIFTY50 tickers + sector indices across 10 sectors
├── phase2_routing/        # News → sector → stock automatic mapping
├── phase3_prediction/     # XGBoost model — UP/DOWN/FLAT prediction
├── final_dashboard/       # Streamlit dashboard tying everything together
└── venv/                  # Python virtual environment
```

---

## How to run locally

```bash
# Clone the repo
git clone https://github.com/ShivamThareja/stock_market_predictor.git
cd stock_market_predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your NewsAPI key
cp phase0_week2/config_template.py phase0_week2/config.py
# Edit config.py and add your key from newsapi.org

# Run the pipeline (fetches latest news)
python3 phase0_week3/week3_pipeline.py --once

# Run the dashboard
cd final_dashboard
streamlit run dashboard.py
```

---

## Key decisions and research backing

| Decision | Reason |
|---|---|
| FinBERT over generic BERT | Trained specifically on financial text — proven better for this domain |
| Headlines only, no full article | LSTM research paper: full article text performs worse than headlines alone |
| ±0.5% noise threshold for ML training | Moves below ±0.5% are random noise — from LSTM paper findings |
| 2015 as ML training start | Pre-2015 Indian market had different structure — less FII, weaker global correlation |
| XGBoost over LSTM | Outperforms deep learning on small tabular datasets — industry standard in quant finance |
| Honest accuracy reporting | 42.2% on limited data documented transparently — overfitting to fake 90% is worse |

---

## What makes this different

Most existing financial NLP projects:
- Cover US markets only — this covers NSE/NIFTY50 specifically
- Use a single news source — this aggregates 8+ sources
- Are research papers, not deployable tools — this runs locally with one command
- Ignore cross-timezone effects — this is built around the US close → India open signal

---

## Roadmap / what's next

- [ ] Deploy dashboard to Streamlit Cloud (public URL)
- [ ] Cron job for fully automated 24/7 data collection
- [ ] Fine-tune FinBERT on 200+ labeled examples (currently 343 collected, labeling in progress)
- [ ] Cross-timezone prediction: US/Europe overnight news → NSE open direction
- [ ] Sector-level models (separate XGBoost per sector)
- [ ] LangChain RAG layer for natural language "why did this stock move?" explanations
- [ ] Knowledge graph of NIFTY50 company relationships for contagion detection

---

## Built by

Shivam Thareja — BTech CSE, PES University Bengaluru (graduating May 2027)

Built incrementally over 4 weeks, learning finance, NLP, and ML through the project itself.
Starting point: zero finance knowledge. Current state: working end-to-end financial intelligence system.

---

*All predictions are for educational purposes only. Not financial advice.*
