"""
PHASE 4: Self-learning engine — shared logic
Project: Global Financial News → Indian Stock Market Predictor

Imports Phase 3's feature_engineering.py READ-ONLY (never edited) to
build today's feature row the exact same way Phase 3 and the
dashboard already do, then layers weighting/learning on top of it in
this new module — keeping the "only add new files" constraint honest
instead of quietly modifying Phase 3 to make this work.

A technical note that shaped this file's design: multiplying a
feature COLUMN by a constant before training is a no-op for tree
models — XGBoost splits on value ORDER, not magnitude, so scaling
"sentiment" by 0.5 wouldn't change a single split in the tree. The
mechanism that actually works is XGBoost's own `feature_weights` fit
parameter, which biases which features get SAMPLED as split
candidates under column sub-sampling (this project's model already
uses colsample_bytree=0.8, so sampling is active and this has real
effect) — verified this works before building on it, not assumed.
That's what SIGNAL WEIGHT actually controls here.

Signal -> feature-column grouping. Every column in
feature_engineering.FEATURE_COLS is covered by exactly one signal
except sector_enc, which is a categorical identity encoding, not an
economic "signal" — it stays at a fixed weight, never adjusted.
"""

import json
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DB_FILE = REPO_ROOT / "phase0_week3" / "news_pipeline.db"
MODEL_STATE_FILE = HERE / "model_state.pkl"
LEARNING_STATE_FILE = HERE / "learning_state.json"

sys.path.insert(0, str(REPO_ROOT / "phase3_prediction"))
import feature_engineering as fe  # noqa: E402  (read-only use — never edited)

SIGNAL_FEATURE_MAP = {
    "sentiment":        ["net_sentiment", "pct_positive", "pct_negative", "pct_neutral", "avg_confidence"],
    "headline_volume":  ["n_headlines"],
    "stock_momentum":   ["prior_1d_return", "prior_3d_return"],
    "market_sentiment": ["market_net_sentiment"],
    "nifty_momentum":   ["nifty_market_prior_1d_return"],
    "sp500":            ["sp500_prior_1d_return"],
    "crude_oil":        ["crude_oil_prior_1d_return"],
    "usdinr":           ["usdinr_prior_1d_return"],
    "india_vix":        ["india_vix_prior_1d_return"],
    "day_of_week":      ["day_of_week"],
}

MIN_WEIGHT = 0.2
MAX_WEIGHT = 2.0
DEFAULT_WEIGHT = 1.0

NIFTY_TICKER = "^NSEI"


def get_conn():
    return sqlite3.connect(DB_FILE)


# ─────────────────────────────────────────────
# Signal weights
# ─────────────────────────────────────────────

def get_signal_weights(conn=None):
    """Returns {signal_name: current_weight}, creating default rows
    (weight 1.0) for any signal not yet in the table."""
    own_conn = conn is None
    conn = conn or get_conn()
    now = datetime.now().isoformat()
    existing = dict(conn.execute("SELECT signal_name, current_weight FROM signal_weights").fetchall())
    for name in SIGNAL_FEATURE_MAP:
        if name not in existing:
            conn.execute(
                "INSERT INTO signal_weights (signal_name, current_weight, last_updated, weight_history_json) "
                "VALUES (?, ?, ?, ?)",
                (name, DEFAULT_WEIGHT, now, json.dumps([{"date": now, "weight": DEFAULT_WEIGHT, "reason": "init"}])),
            )
            existing[name] = DEFAULT_WEIGHT
    conn.commit()
    if own_conn:
        conn.close()
    return existing


def set_signal_weight(conn, signal_name, new_weight, reason, pre_decay_weight=None, reset_streaks=None):
    new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
    now = datetime.now().isoformat()
    row = conn.execute(
        "SELECT current_weight, weight_history_json FROM signal_weights WHERE signal_name = ?", (signal_name,)
    ).fetchone()
    history = json.loads(row[1]) if row and row[1] else []
    history.append({"date": now, "weight": new_weight, "reason": reason})
    fields = ["current_weight = ?", "last_updated = ?", "weight_history_json = ?"]
    params = [new_weight, now, json.dumps(history)]
    if pre_decay_weight is not None:
        fields.append("pre_decay_weight = ?")
        params.append(pre_decay_weight)
    if reset_streaks == "wrong":
        fields += ["consecutive_wrong = 0"]
    elif reset_streaks == "correct":
        fields += ["consecutive_correct = 0"]
    elif reset_streaks == "both":
        fields += ["consecutive_wrong = 0", "consecutive_correct = 0"]
    params.append(signal_name)
    conn.execute(f"UPDATE signal_weights SET {', '.join(fields)} WHERE signal_name = ?", params)
    conn.commit()
    return new_weight


def feature_weights_array(signal_weights, feature_cols):
    """Broadcast {signal_name: weight} to a per-column array matching
    feature_cols order (+ sector_enc at fixed weight 1.0), for
    XGBoost's feature_weights fit() parameter."""
    col_to_signal = {}
    for signal, cols in SIGNAL_FEATURE_MAP.items():
        for c in cols:
            col_to_signal[c] = signal
    weights = []
    for col in feature_cols:
        signal = col_to_signal.get(col)
        weights.append(signal_weights.get(signal, DEFAULT_WEIGHT) if signal else DEFAULT_WEIGHT)
    weights.append(DEFAULT_WEIGHT)  # sector_enc — identity encoding, never weighted
    return np.array(weights, dtype=float)


# ─────────────────────────────────────────────
# Model persistence — Phase 3's script only writes
# CSVs, so this system keeps its own saved model
# (trained WITH current signal weights applied) for
# the morning prediction to load without retraining
# from scratch every single day.
# ─────────────────────────────────────────────

def save_model_state(model, sector_encoder, label_encoder, feature_cols, meta):
    with open(MODEL_STATE_FILE, "wb") as f:
        pickle.dump({
            "model": model, "sector_encoder": sector_encoder, "label_encoder": label_encoder,
            "feature_cols": feature_cols, "meta": meta,
        }, f)


def load_model_state():
    if not MODEL_STATE_FILE.exists():
        return None
    with open(MODEL_STATE_FILE, "rb") as f:
        return pickle.load(f)


def train_weighted_model(dataset_labeled):
    """Train on ALL labeled history (same pattern as the dashboard's
    live model), but with today's signal_weights biasing feature
    sampling via feature_weights — this is the one place signal
    weights actually reach the model."""
    sector_encoder, label_encoder = fe.fit_encoders(dataset_labeled)
    X = fe.model_matrix(dataset_labeled, sector_encoder)
    y = label_encoder.transform(dataset_labeled["label"])
    weights = get_signal_weights()
    fweights = feature_weights_array(weights, fe.FEATURE_COLS)
    model = fe.make_model()
    model.fit(X, y, feature_weights=fweights)
    return model, sector_encoder, label_encoder, weights


# ─────────────────────────────────────────────
# Learning-engine state (headline count / drift
# tracking for the retrain trigger) — a small JSON
# file rather than a 7th table; purely additive,
# no existing file or table touched either way.
# ─────────────────────────────────────────────

def load_learning_state():
    if LEARNING_STATE_FILE.exists():
        return json.loads(LEARNING_STATE_FILE.read_text())
    return {"last_retrain_headline_count": 0, "last_retrain_at": None,
            "last_retrain_accuracy": None, "cumulative_weight_drift": 0.0}


def save_learning_state(state):
    LEARNING_STATE_FILE.write_text(json.dumps(state, indent=2))


def record_weight_drift(delta):
    state = load_learning_state()
    state["cumulative_weight_drift"] = state.get("cumulative_weight_drift", 0.0) + abs(delta)
    save_learning_state(state)


def check_retrain_trigger(conn):
    """Returns (should_retrain: bool, reason: str|None)."""
    state = load_learning_state()
    total_headlines = conn.execute(
        "SELECT COUNT(*) FROM news_articles WHERE sentiment_label IS NOT NULL"
    ).fetchone()[0]
    new_headlines = total_headlines - state.get("last_retrain_headline_count", 0)
    if new_headlines >= 50:
        return True, f"{new_headlines} new labeled headlines since last training"

    if state.get("cumulative_weight_drift", 0.0) > 0.20:
        return True, f"cumulative signal weight drift {state['cumulative_weight_drift']:.2f} > 0.20"

    last10 = pd.read_sql(
        "SELECT correct FROM prediction_outcomes ORDER BY date DESC LIMIT 10", conn
    )
    if len(last10) >= 10 and last10["correct"].mean() < 0.38:
        return True, f"accuracy over last 10 predictions ({last10['correct'].mean():.0%}) below 38%"

    return False, None


def mark_retrained(headline_count, accuracy):
    state = load_learning_state()
    state["last_retrain_headline_count"] = headline_count
    state["last_retrain_at"] = datetime.now().isoformat()
    state["last_retrain_accuracy"] = accuracy
    state["cumulative_weight_drift"] = 0.0
    save_learning_state(state)


# ─────────────────────────────────────────────
# Per-prediction feature contributions (SHAP via
# XGBoost's own pred_contribs) — used both to report
# "top signals" each morning and to attribute blame
# for a wrong prediction each afternoon.
# ─────────────────────────────────────────────

def signal_contributions(model, X_row, feature_cols, predicted_class_idx):
    """Returns {signal_name: contribution} toward the predicted
    class's margin, for one row. Sums each signal's constituent
    feature columns' SHAP contributions."""
    booster = model.get_booster()
    import xgboost as xgb
    dm = xgb.DMatrix(X_row, feature_names=feature_cols + ["sector_enc"])
    contribs = booster.predict(dm, pred_contribs=True)  # shape: (1, n_classes, n_features+1)
    row_contribs = contribs[0][predicted_class_idx][:-1]  # drop bias term

    col_to_signal = {}
    for signal, cols in SIGNAL_FEATURE_MAP.items():
        for c in cols:
            col_to_signal[c] = signal

    per_signal = {}
    all_cols = feature_cols + ["sector_enc"]
    for col, contrib in zip(all_cols, row_contribs):
        signal = col_to_signal.get(col, "sector")
        per_signal[signal] = per_signal.get(signal, 0.0) + float(contrib)
    return per_signal


# ─────────────────────────────────────────────
# TF-IDF similar past headlines
# ─────────────────────────────────────────────

def find_similar_past_events(today_headlines, all_headlines_df, prices, top_k=3):
    """today_headlines: list of str. all_headlines_df: DataFrame with
    headline, published_at columns (excludes today's own rows already).
    Returns list of {headline, date, similarity, next_session_return}."""
    if len(all_headlines_df) < 5 or not today_headlines:
        return []

    corpus = all_headlines_df["headline"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    try:
        corpus_matrix = vectorizer.fit_transform(corpus)
        today_matrix = vectorizer.transform(today_headlines)
    except ValueError:
        return []

    sims = cosine_similarity(today_matrix, corpus_matrix)
    best_per_past = sims.max(axis=0)  # best similarity to ANY of today's headlines, per past headline
    top_idx = np.argsort(best_per_past)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if best_per_past[idx] <= 0:
            continue
        row = all_headlines_df.iloc[idx]
        past_date = pd.to_datetime(row["published_at"], utc=True).tz_localize(None).normalize()
        ret = fe.next_session_return(prices, "NIFTY 50 Index", past_date)
        results.append({
            "headline": row["headline"],
            "date": str(past_date.date()),
            "similarity": round(float(best_per_past[idx]), 3),
            "next_session_return": round(ret, 2) if ret is not None else None,
            "outcome": fe.label_from_return(ret) if ret is not None else "unresolved",
        })
    return results


# ─────────────────────────────────────────────
# Source credibility
# ─────────────────────────────────────────────

def update_source_credibility(conn, sources, was_correct):
    """Coarse attribution, documented as such: every source that had
    at least one headline routed to NIFTY today shares equal credit
    or blame for today's single outcome. Finer per-headline causal
    attribution isn't possible with a same-day-aggregate model —
    same 'reasonable first pass' caveat Phase 2's company matching
    already carries."""
    now = datetime.now().isoformat()
    changes = []
    for source in set(sources):
        row = conn.execute(
            "SELECT total_predictions, correct_predictions, current_weight FROM source_credibility WHERE source_name = ?",
            (source,),
        ).fetchone()
        total, correct, weight = row if row else (0, 0, DEFAULT_WEIGHT)
        total += 1
        correct += 1 if was_correct else 0
        accuracy = correct / total
        new_weight = weight
        if total >= 20:
            if accuracy < 0.40:
                new_weight = 0.3
            elif accuracy > 0.65:
                new_weight = 1.5
        conn.execute(
            "INSERT INTO source_credibility (source_name, total_predictions, correct_predictions, "
            "accuracy_rate, current_weight, last_updated) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_name) DO UPDATE SET total_predictions=excluded.total_predictions, "
            "correct_predictions=excluded.correct_predictions, accuracy_rate=excluded.accuracy_rate, "
            "current_weight=excluded.current_weight, last_updated=excluded.last_updated",
            (source, total, correct, accuracy, new_weight, now),
        )
        if new_weight != weight:
            changes.append((source, weight, new_weight, accuracy))
    conn.commit()
    return changes


def get_source_weights(conn):
    rows = conn.execute("SELECT source_name, current_weight FROM source_credibility").fetchall()
    return dict(rows)
