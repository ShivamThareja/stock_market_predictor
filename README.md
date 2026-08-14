# Stock Market Predictor

> Global financial news → Indian stock market direction predictor

A complete, end-to-end financial intelligence system that reads news from around the world 24/7, uses AI to understand sentiment, and predicts how Indian stocks (NIFTY50) will react — both from same-timezone Indian news (RBI, NSE announcements) and cross-timezone overnight news (US Fed, Nikkei, global markets) — before NSE opens each morning.

---

## Live results

| Metric | Value |
|---|---|
| Headlines collected | 386 labeled headlines |
| Stock-news pairs routed | 1,037 (headline, stock) pairs |
| Tickers tracked | 56 (all NIFTY50 + sector indices) |
| Model accuracy | 43.5% (+10.9pts above 32.6% random baseline) |
| Training data | ~29 trading days (accuracy improves as pipeline runs longer) |

> **Honest note on accuracy:** 43.5% on 29 days of data is a proof of concept, not a finished result. A round of real improvement work — rule-based sentiment corrections, 5 new macro features (S&P 500, crude oil, USD/INR, India VIX, day-of-week), and a FinBERT fine-tuning attempt — moved this from an earlier 42.2% baseline by just over a point, which is itself the finding: more data (accumulated over time, not engineered around) is what's actually left to move this number meaningfully. Full writeup in `phase3_prediction/README.md` and `phase0_week4/README.md`, including the fine-tuning attempt that honestly did NOT beat zero-shot FinBERT.

---

## What it does

```
Global News
       ↓
week3_pipeline.py     — fetches headlines hourly across 6 topic queries (NewsAPI)
       ↓
week4_finbert.py      — FinBERT AI labels each headline: positive / negative / neutral
week4_label_rules.py  — targeted corrections for known FinBERT failure patterns
       ↓
phase2_news_routing.py — maps each headline to affected NIFTY50 stocks and sectors
       ↓
phase3_xgboost_model.py — XGBoost model predicts: UP / DOWN / FLAT for next session
       ↓
Streamlit Dashboard   — live view of pipeline, NIFTY chart, model accuracy
```

---

## Data sources

| Source | What it provides |
|---|---|
| NewsAPI.org | Global financial news across 6 topic queries (RBI, NIFTY, Indian Banking, Indian IT, US Fed, Global Markets) |
| yfinance | 10 years of NSE/BSE price data (56 NIFTY50 tickers + sector indices), plus S&P 500, crude oil, USD/INR, India VIX |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data collection | `yfinance`, `NewsAPI`, `requests` |
| Scheduling | `APScheduler` (manual/scheduler mode) + `cron` for hands-off hourly runs |
| Storage | `SQLite` — headlines + sentiment; prices/routing/model outputs as CSV |
| NLP / Sentiment | `FinBERT` (ProsusAI/finbert) via HuggingFace Transformers, plus targeted rule-based corrections |
| Entity/sector matching | Word-boundary regex over the NIFTY50 name list (`phase2_news_routing.py`) |
| ML prediction | `XGBoost` — gradient boosted trees |
| Dashboard | `Streamlit` + `Plotly` |
| Language | Python 3.9 |

---

## Project structure

```
stock_market_predictor/
├── phase0_week1/          # Stock data exploration — 10yr NIFTY50 data, OHLCV, correlations
├── phase0_week2/          # News fetching — NewsAPI integration, manual sentiment labeling
├── phase0_week3/          # Automated pipeline — APScheduler, SQLite, NewsAPI
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

# Add your NewsAPI key (week3_pipeline.py reads phase0_week3/config.py specifically)
cp phase0_week3/config_template.py phase0_week3/config.py
# Edit config.py and add your key from newsapi.org

# Run the pipeline (fetches latest news)
cd phase0_week3 && python3 week3_pipeline.py --once && cd ..

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
| Honest accuracy reporting | 43.5% on limited data documented transparently, including a fine-tuning attempt that didn't beat baseline — overfitting to fake 90% is worse |

---

## What makes this different

Most existing financial NLP projects:
- Cover US markets only — this covers NSE/NIFTY50 specifically
- Are research papers, not deployable tools — this runs locally with one command (and has a live dashboard)
- Ignore cross-timezone effects — this is built around the US close → India open signal (S&P 500 prior-day return is a direct model feature, not just a news topic)

---

## Roadmap / what's next

- [ ] Deploy dashboard to Streamlit Cloud (public URL)
- [ ] Cron job for fully automated hourly data collection (command ready — see below; needs to be run once from a real terminal, not this sandboxed one)
- [x] Fine-tune FinBERT on 200+ labeled examples — done at 386; honestly tied zero-shot (69.6%) rather than beating it, see `phase0_week4/README.md`
- [x] Macro/cross-timezone features: S&P 500, crude oil, USD/INR, India VIX prior-day change as direct model inputs
- [ ] Sector-level models (separate XGBoost per sector)
- [ ] LangChain RAG layer for natural language "why did this stock move?" explanations
- [ ] Knowledge graph of NIFTY50 company relationships for contagion detection

To start the automated hourly pipeline yourself:
```bash
crontab -e
# add:
7 * * * * cd /path/to/stock_market_predictor-main/phase0_week3 && /path/to/stock_market_predictor-main/venv/bin/python3 week3_pipeline.py --once >> ~/pipeline.log 2>&1
```

---

## Built by

Shivam Thareja — BTech CSE, PES University Bengaluru (graduating May 2027)

Built incrementally over 4 weeks, learning finance, NLP, and ML through the project itself.
Starting point: zero finance knowledge. Current state: working end-to-end financial intelligence system.

---

*All predictions are for educational purposes only. Not financial advice.*
