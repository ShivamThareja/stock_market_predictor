"""
FINAL PHASE: Streamlit dashboard
Project: Global Financial News → Indian Stock Market Predictor
Goal: Tie every earlier phase together into one view — news
      collection, sentiment labeling, sector/stock routing, and
      the Phase 3 prediction model — instead of six separate
      scripts each printing to their own terminal.

Reads (never re-fetches/re-trains on its own — see the README for
why, and how to refresh each layer):
  phase0_week3/news_pipeline.db        - raw + FinBERT-labeled headlines
  phase2_routing/phase2_routed_events.csv - headline -> stock routing
  phase1_expansion/phase1_close_prices_all.csv - price history
  phase1_expansion/phase1_sector_map.csv - ticker -> sector/name
  phase3_prediction/*.csv              - Phase 3's last training run

The one thing this dashboard computes itself (not just displays):
"live" predictions for news that's too recent to have a resolved
next-session close yet — reusing phase3_prediction/feature_engineering.py
so live inference and the trained/evaluated model never drift apart.

Run: streamlit run dashboard.py
(needs DYLD_LIBRARY_PATH set on macOS without Homebrew — see README)
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

PHASE3_DIR = Path(__file__).resolve().parent.parent / "phase3_prediction"
sys.path.insert(0, str(PHASE3_DIR))
from feature_engineering import FEATURE_COLS, build_dataset, fit_encoders, make_model, model_matrix  # noqa: E402

NEWS_DB = Path(__file__).resolve().parent.parent / "phase0_week3" / "news_pipeline.db"
ROUTED_CSV = Path(__file__).resolve().parent.parent / "phase2_routing" / "phase2_routed_events.csv"
PRICES_CSV = Path(__file__).resolve().parent.parent / "phase1_expansion" / "phase1_close_prices_all.csv"
SECTOR_MAP_CSV = Path(__file__).resolve().parent.parent / "phase1_expansion" / "phase1_sector_map.csv"
METRICS_CSV = PHASE3_DIR / "phase3_metrics.csv"
IMPORTANCE_CSV = PHASE3_DIR / "phase3_feature_importance.csv"
TEST_PRED_CSV = PHASE3_DIR / "phase3_test_predictions.csv"

# ─────────────────────────────────────────────
# Fixed color language, used consistently everywhere on this page:
#   green  = positive sentiment / predicted UP   (bullish)
#   red    = negative sentiment / predicted DOWN (bearish)
#   gray   = neutral sentiment  / predicted FLAT
# Same meaning every time these colors appear, never reused for
# anything else. Topics get their own identity palette so they're
# never confused with the sentiment/direction colors above.
# ─────────────────────────────────────────────
STATUS_COLORS = {"positive": "#0ca30c", "negative": "#d03b3b", "neutral": "#898781"}
DIRECTION_COLORS = {"UP": "#0ca30c", "DOWN": "#d03b3b", "FLAT": "#898781"}
TOPIC_COLORS = {
    "RBI": "#2a78d6", "NIFTY": "#eb6834", "Indian_Banking": "#1baf7a",
    "Indian_IT": "#eda100", "US_Fed": "#e87ba4", "Global_Markets": "#4a3aa7",
}
SEQUENTIAL_BLUE = "#2a78d6"

st.set_page_config(page_title="News -> NIFTY Predictor", layout="wide", page_icon="📈")


# ─────────────────────────────────────────────
# Cached loaders — each reads whatever the last
# pipeline run left on disk. Cache clears on the
# sidebar "Refresh data" button.
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_news():
    if not NEWS_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(NEWS_DB)
    df = pd.read_sql(
        "SELECT topic, headline, source, published_at, sentiment_label, sentiment_confidence "
        "FROM news_articles WHERE sentiment_label IS NOT NULL", conn,
    )
    conn.close()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(ttl=300)
def load_routed():
    if not ROUTED_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ROUTED_CSV)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(ttl=300)
def load_prices():
    if not PRICES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True).sort_index()


@st.cache_data(ttl=300)
def load_sector_map():
    if not SECTOR_MAP_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SECTOR_MAP_CSV)


@st.cache_data(ttl=300)
def load_phase3_artifacts():
    metrics = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else None
    importance = pd.read_csv(IMPORTANCE_CSV) if IMPORTANCE_CSV.exists() else None
    test_pred = pd.read_csv(TEST_PRED_CSV) if TEST_PRED_CSV.exists() else None
    return metrics, importance, test_pred


@st.cache_data(ttl=300)
def compute_full_dataset(routed, prices, sector_map):
    if routed.empty or prices.empty or sector_map.empty:
        return pd.DataFrame()
    return build_dataset(routed, prices, sector_map)


@st.cache_resource(ttl=300)
def train_live_model(dataset_labeled_json):
    """Train on ALL resolved (ticker, day) rows — not the time-split
    used for evaluation — so live in-flight predictions use every bit
    of real data available. Cached on the dataset's own content, so a
    fresh news day invalidates it automatically."""
    dataset = pd.read_json(dataset_labeled_json)
    if len(dataset) < 20 or dataset["label"].nunique() < 2:
        return None
    sector_encoder, label_encoder = fit_encoders(dataset)
    X = model_matrix(dataset, sector_encoder)
    y = label_encoder.transform(dataset["label"])
    model = make_model()
    model.fit(X, y)
    return model, sector_encoder, label_encoder


# ─────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────

news = load_news()
routed = load_routed()
prices = load_prices()
sector_map = load_sector_map()
metrics, importance, test_pred = load_phase3_artifacts()
full_dataset = compute_full_dataset(routed, prices, sector_map)

if not full_dataset.empty:
    labeled = full_dataset[full_dataset["label"].notna()].reset_index(drop=True)
    live_rows = full_dataset[full_dataset["label"].isna()].reset_index(drop=True)
else:
    labeled, live_rows = pd.DataFrame(), pd.DataFrame()

name_lookup = sector_map.set_index("ticker")["name"] if not sector_map.empty else pd.Series(dtype=str)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Pipeline status")
    st.caption("Global news → Indian stock market predictor")

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**Data freshness**")
    if not news.empty:
        st.caption(f"News: {len(news)} labeled headlines")
        st.caption(f"Latest: {news['published_at'].max():%Y-%m-%d %H:%M}")
    else:
        st.caption("News: no data — run phase0_week3 + week4_finbert.py")
    if not routed.empty:
        st.caption(f"Routed events: {len(routed)}")
    if not prices.empty:
        st.caption(f"Prices through: {prices.index.max():%Y-%m-%d}")
    if metrics is not None and len(metrics):
        st.caption(f"Model last trained: {pd.to_datetime(metrics.iloc[-1]['run_at']):%Y-%m-%d %H:%M}")

    st.divider()
    st.markdown("**To pull in new data**, run (in order):")
    st.code(
        "phase0_week3/week3_pipeline.py --once\n"
        "phase0_week4/week4_finbert.py\n"
        "phase2_routing/phase2_news_routing.py\n"
        "phase1_expansion/phase1_nifty50.py\n"
        "phase3_prediction/phase3_xgboost_model.py",
        language="bash",
    )
    st.caption("then hit Refresh data above.")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_overview, tab_news, tab_routing, tab_predictions = st.tabs(
    ["Overview", "News & Sentiment", "Routing", "Predictions"]
)


# ---------- Overview ----------
with tab_overview:
    st.header("Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labeled headlines", len(news) if not news.empty else 0)
    c2.metric("Routed (headline, stock) rows", len(routed) if not routed.empty else 0)
    c3.metric("Tickers tracked", int(prices.shape[1]) if not prices.empty else 0)
    c4.metric(
        "Model accuracy (test)",
        f"{metrics.iloc[-1]['model_accuracy']:.1%}" if metrics is not None and len(metrics) else "—",
        delta=f"{metrics.iloc[-1]['lift_pts']:+.1f} pts vs baseline" if metrics is not None and len(metrics) else None,
    )

    if not prices.empty and "NIFTY 50 Index" in prices.columns:
        st.subheader("NIFTY 50 — last 6 months")
        nifty_series = prices["NIFTY 50 Index"].dropna()
        recent = nifty_series[nifty_series.index >= nifty_series.index.max() - pd.Timedelta(days=180)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent.values, mode="lines",
            line=dict(color=SEQUENTIAL_BLUE, width=2), fill="tozeroy",
            fillcolor="rgba(42,120,214,0.08)", hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title="Close",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data yet — run phase1_expansion/phase1_nifty50.py")

    st.subheader("How the pieces connect")
    st.markdown(
        "```\n"
        "week3_pipeline.py  -->  week4_finbert.py  -->  phase2_news_routing.py  -->  phase3_xgboost_model.py\n"
        "  (fetch headlines)     (label sentiment)      (route to stocks)          (predict direction)\n"
        "```\n"
        "This dashboard reads the output each script already writes — it doesn't "
        "re-fetch news or re-train the model on its own. Hit **Refresh data** in "
        "the sidebar after re-running any of them."
    )


# ---------- News & Sentiment ----------
with tab_news:
    st.header("News & Sentiment")

    if news.empty:
        st.info("No labeled news yet.")
    else:
        col_a, col_b = st.columns([1, 1])
        topics = sorted(news["topic"].unique())
        sentiments = sorted(news["sentiment_label"].unique())
        pick_topics = col_a.multiselect("Topic", topics, default=topics)
        pick_sentiments = col_b.multiselect("Sentiment", sentiments, default=sentiments)

        filtered = news[news["topic"].isin(pick_topics) & news["sentiment_label"].isin(pick_sentiments)]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment distribution")
            counts = filtered["sentiment_label"].value_counts().reindex(["positive", "neutral", "negative"]).dropna()
            fig = go.Figure(go.Bar(
                x=counts.values, y=counts.index, orientation="h",
                marker_color=[STATUS_COLORS[s] for s in counts.index],
                hovertemplate="%{y}: %{x}<extra></extra>",
            ))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Headlines by topic")
            tcounts = filtered["topic"].value_counts()
            fig = go.Figure(go.Bar(
                x=tcounts.index, y=tcounts.values,
                marker_color=[TOPIC_COLORS.get(t, "#898781") for t in tcounts.index],
                hovertemplate="%{x}: %{y}<extra></extra>",
            ))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment over time")
        daily = filtered.copy()
        daily["day"] = daily["published_at"].dt.normalize()
        daily_counts = daily.groupby(["day", "sentiment_label"]).size().reset_index(name="count")
        fig = go.Figure()
        for label in ["positive", "neutral", "negative"]:
            sub = daily_counts[daily_counts["sentiment_label"] == label]
            fig.add_trace(go.Scatter(
                x=sub["day"], y=sub["count"], mode="lines", name=label,
                line=dict(color=STATUS_COLORS[label], width=2), stackgroup="one",
            ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend_title=None,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Headlines ({len(filtered)})")
        show = filtered.sort_values("published_at", ascending=False)[
            ["published_at", "topic", "sentiment_label", "sentiment_confidence", "source", "headline"]
        ]
        st.dataframe(show, use_container_width=True, hide_index=True, height=350)


# ---------- Routing ----------
with tab_routing:
    st.header("Routing — who does this news actually affect")

    if routed.empty:
        st.info("No routed events yet — run phase2_routing/phase2_news_routing.py")
    else:
        routed_local = routed.copy()
        routed_local["sign"] = routed_local["sentiment_label"].map({"positive": 1, "negative": -1, "neutral": 0})

        st.subheader("Net sentiment by sector")
        sector_sent = routed_local.groupby("sector")["sign"].mean().sort_values()
        colors = ["#d03b3b" if v < -0.05 else "#0ca30c" if v > 0.05 else "#898781" for v in sector_sent.values]
        fig = go.Figure(go.Bar(
            x=sector_sent.values, y=sector_sent.index, orientation="h",
            marker_color=colors, hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#c3c2b7")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="net sentiment (-1 to +1)",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Most-covered stocks")
        top_tickers = routed_local["company_name"].value_counts().head(15)
        fig = go.Figure(go.Bar(
            x=top_tickers.values, y=top_tickers.index, orientation="h",
            marker_color=SEQUENTIAL_BLUE, hovertemplate="%{y}: %{x}<extra></extra>",
        ))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(autorange="reversed"),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest routed events")
        show = routed_local.sort_values("published_at", ascending=False)[
            ["published_at", "company_name", "sector", "sentiment_label", "matched_via", "headline"]
        ].head(50)
        st.dataframe(show, use_container_width=True, hide_index=True, height=350)


# ---------- Predictions ----------
with tab_predictions:
    st.header("Phase 3 model — next-session direction")

    if metrics is None or not len(metrics):
        st.info("No trained model yet — run phase3_prediction/phase3_xgboost_model.py")
    else:
        m = metrics.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Model accuracy (test)", f"{m['model_accuracy']:.1%}")
        c2.metric("Majority-class baseline", f"{m['baseline_accuracy']:.1%}", help=f"always predicting '{m['majority_label']}'")
        c3.metric("Lift", f"{m['lift_pts']:+.1f} pts")
        st.caption(
            f"Trained on {int(m['train_rows'])} rows, tested on {int(m['test_rows'])} rows "
            f"from {int(m['n_news_days'])} total news days (time-based split at {m['split_date']}). "
            "Small dataset — treat this as a directional signal, not a stable production number. "
            "See phase3_prediction/README.md for the full honest writeup."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if test_pred is not None and len(test_pred):
                st.subheader("Confusion matrix (test set)")
                order = ["DOWN", "FLAT", "UP"]
                cm = pd.crosstab(test_pred["label"], test_pred["predicted_label"]).reindex(
                    index=order, columns=order, fill_value=0
                )
                fig = px.imshow(
                    cm.values, x=order, y=order, text_auto=True,
                    color_continuous_scale=[[0, "#fcfcfb"], [1, SEQUENTIAL_BLUE]],
                    labels=dict(x="predicted", y="actual", color="count"),
                )
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if importance is not None and len(importance):
                st.subheader("Feature importance")
                imp = importance.sort_values("importance", ascending=True)
                fig = go.Figure(go.Bar(
                    x=imp["importance"], y=imp["feature"], orientation="h",
                    marker_color=SEQUENTIAL_BLUE,
                ))
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Live predictions — news too recent to have a next-session close yet")

        if live_rows.empty:
            st.caption("Nothing in-flight right now — every routed (ticker, day) pair already has a "
                       "resolved next-session outcome. Check back after new headlines come in.")
        else:
            bundle = train_live_model(labeled.to_json())
            if bundle is None:
                st.caption("Not enough labeled history yet to train a live model.")
            else:
                model, sector_encoder, label_encoder = bundle
                X_live = model_matrix(live_rows, sector_encoder)
                proba = model.predict_proba(X_live)
                pred_idx = proba.argmax(axis=1)
                pred_labels = label_encoder.inverse_transform(pred_idx)
                confidence = proba.max(axis=1)

                live_out = live_rows[["ticker", "sector", "date", "n_headlines", "net_sentiment"]].copy()
                live_out["company_name"] = live_out["ticker"].map(name_lookup).fillna(live_out["ticker"])
                live_out["predicted_direction"] = pred_labels
                live_out["confidence"] = confidence
                live_out = live_out.sort_values("confidence", ascending=False)

                st.caption(
                    f"Model trained on all {len(labeled)} resolved (ticker, day) rows to date "
                    f"(not the time-split held-out model above) — this is the closest thing this "
                    f"project has to a real forward prediction right now."
                )

                for _, row in live_out.iterrows():
                    color = DIRECTION_COLORS.get(row["predicted_direction"], "#898781")
                    cols = st.columns([3, 2, 2, 2, 3])
                    cols[0].markdown(f"**{row['company_name']}** ({row['sector']})")
                    cols[1].markdown(f"news date: {pd.to_datetime(row['date']).date()}")
                    cols[2].markdown(f"{int(row['n_headlines'])} headline(s)")
                    cols[3].markdown(
                        f"<span style='color:{color};font-weight:600'>{row['predicted_direction']}</span>",
                        unsafe_allow_html=True,
                    )
                    cols[4].progress(float(row["confidence"]), text=f"{row['confidence']:.0%} confidence")
