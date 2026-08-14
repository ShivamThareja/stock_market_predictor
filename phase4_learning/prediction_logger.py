"""
PHASE 4: Self-learning engine — morning prediction
Project: Global Financial News → Indian Stock Market Predictor

Meant to run once every morning before NSE opens (7 AM IST, via
scheduler.py). Predicts NIFTY 50's next-session direction from
overnight news and logs the full reasoning trail — every number in
prediction_log is something learning_engine.py can later check itself
against, not a black-box guess.

Step 0 (new, not in the original spec, added for correctness): runs
the existing hourly pipeline scripts once via subprocess FIRST — the
scheduler's hourly cadence (:07/:15/:20) might not have run yet this
hour when this fires at 7:00 sharp, and predicting on stale overnight
news would defeat the point. Calls week3_pipeline.py, week4_finbert.py,
week4_label_rules.py, phase2_news_routing.py exactly as they already
exist — never imports or edits them, only invokes them as separate
processes, same as any other cron-style caller would.

Run: python3 prediction_logger.py
"""

import json
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import schema  # noqa: E402
import signals  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "phase3_prediction"))
import feature_engineering as fe  # noqa: E402

PYTHON = sys.executable


def run_step(label, cwd, args):
    print(f"  -> {label}...")
    result = subprocess.run([PYTHON] + args, cwd=cwd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"     WARNING: {label} exited {result.returncode}: {result.stderr[-300:]}")
    return result


def refresh_pipeline():
    print("Step 0: refreshing data before predicting (existing scripts, untouched, called as subprocesses)")
    run_step("week3_pipeline.py --once", REPO_ROOT / "phase0_week3", ["week3_pipeline.py", "--once"])
    run_step("week4_finbert.py", REPO_ROOT / "phase0_week4", ["week4_finbert.py"])
    run_step("week4_label_rules.py", REPO_ROOT / "phase0_week4", ["week4_label_rules.py"])
    run_step("phase2_news_routing.py", REPO_ROOT / "phase2_routing", ["phase2_news_routing.py"])


def load_data():
    routed = pd.read_csv(REPO_ROOT / "phase2_routing" / "phase2_routed_events.csv")
    prices = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_close_prices_all.csv", index_col="Date", parse_dates=True).sort_index()
    macro_csv = REPO_ROOT / "phase1_expansion" / "phase1_macro_data.csv"
    if macro_csv.exists():
        macro = pd.read_csv(macro_csv, index_col="Date", parse_dates=True).sort_index()
        prices = prices.join(macro, how="outer")
    sector_map = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_sector_map.csv")
    return routed, prices, sector_map


def main():
    print("=" * 60)
    print("  PHASE 4: Morning prediction")
    print("=" * 60)

    schema.create_tables()
    refresh_pipeline()

    print("\nStep 1: building today's NIFTY feature row")
    routed, prices, sector_map = load_data()
    full_dataset = fe.build_dataset(routed, prices, sector_map)
    nifty_rows = full_dataset[full_dataset["ticker"] == signals.NIFTY_TICKER].sort_values("date")
    live_rows = nifty_rows[nifty_rows["label"].isna()]

    if live_rows.empty:
        print("  No unresolved NIFTY (ticker, date) row right now — either no fresh news yet today, "
              "or today's news already has a resolved next-session close. Nothing to predict yet.")
        return

    today_row = live_rows.iloc[[-1]]
    news_date = today_row["date"].iloc[0]
    print(f"  Using news date {news_date.date()} (most recent unresolved NIFTY row)")

    print("\nStep 2: loading model (retraining fresh if none saved yet)")
    state = signals.load_model_state()
    labeled = full_dataset[full_dataset["label"].notna()]
    if state is None or len(labeled) < 20:
        if len(labeled) < 20:
            print(f"  Only {len(labeled)} labeled rows — too few to train yet.")
            return
        print("  No saved model state — training one now.")
        model, sector_encoder, label_encoder, weights = signals.train_weighted_model(labeled)
        signals.save_model_state(model, sector_encoder, label_encoder, fe.FEATURE_COLS,
                                  {"trained_at": datetime.now().isoformat(), "n_rows": len(labeled)})
    else:
        model, sector_encoder, label_encoder = state["model"], state["sector_encoder"], state["label_encoder"]
        weights = signals.get_signal_weights()

    print("\nStep 3: predicting")
    X_today = fe.model_matrix(today_row, sector_encoder)
    proba = model.predict_proba(X_today)[0]
    pred_idx = int(proba.argmax())
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])
    print(f"  Prediction: {prediction}  (confidence {confidence:.1%})")

    print("\nStep 4: explaining the prediction")
    per_signal = signals.signal_contributions(model, X_today, fe.FEATURE_COLS, pred_idx)
    top_signals = sorted(per_signal.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    top_signals_out = [{"signal": s, "contribution": round(c, 4)} for s, c in top_signals]
    print(f"  Top signals: {top_signals_out}")

    conn = signals.get_conn()
    routed_today = routed[routed["ticker"] == signals.NIFTY_TICKER].copy()
    routed_today["published_at"] = pd.to_datetime(routed_today["published_at"], utc=True).dt.tz_localize(None)
    routed_today = routed_today[routed_today["published_at"].dt.normalize() == news_date]
    routed_today["strength"] = routed_today["sentiment_confidence"] * routed_today["sentiment_label"].map(
        {"positive": 1, "negative": -1, "neutral": 0}
    ).abs().clip(lower=0.01)
    influential = routed_today.sort_values("strength", ascending=False).head(5)
    influential_out = [
        {"headline": r.headline, "source": r.source, "sentiment": r.sentiment_label}
        for r in influential.itertuples(index=False)
    ]

    print("\nStep 5: finding similar past headlines (TF-IDF)")
    history = pd.read_sql(
        "SELECT headline, published_at FROM news_articles WHERE date(published_at) < date(?)",
        conn, params=(str(news_date.date()),),
    )
    similar = signals.find_similar_past_events(routed_today["headline"].tolist(), history, prices)

    source_weights = signals.get_source_weights(conn)

    print("\nStep 6: logging")
    today_str = str(datetime.now().date())
    conn.execute(
        "INSERT INTO prediction_log (date, timestamp, prediction, confidence, top_signals, "
        "influential_headlines, similar_past_events, signal_weights_snapshot, source_weights_snapshot, news_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(date) DO UPDATE SET timestamp=excluded.timestamp, prediction=excluded.prediction, "
        "confidence=excluded.confidence, top_signals=excluded.top_signals, "
        "influential_headlines=excluded.influential_headlines, similar_past_events=excluded.similar_past_events, "
        "signal_weights_snapshot=excluded.signal_weights_snapshot, source_weights_snapshot=excluded.source_weights_snapshot, "
        "news_date=excluded.news_date",
        (
            today_str, datetime.now().isoformat(), prediction, confidence,
            json.dumps(top_signals_out), json.dumps(influential_out), json.dumps(similar),
            json.dumps(weights), json.dumps(source_weights), str(news_date.date()),
        ),
    )
    conn.commit()
    conn.close()

    print(f"\nLogged prediction for {today_str}: {prediction} ({confidence:.1%} confidence)")
    print("\n" + "=" * 60)
    print("  DONE — learning_engine.py will check this against reality this evening")
    print("=" * 60)


if __name__ == "__main__":
    main()
