"""
PHASE 4: Self-learning engine — outcome recording + learning
Project: Global Financial News → Indian Stock Market Predictor

Meant to run once every afternoon after NSE closes (4 PM IST, via
scheduler.py). For every prediction_log row not yet resolved in
prediction_outcomes:

  1. Pulls NIFTY's actual close (yfinance) and compares to the
     morning's prediction.
  2. If wrong, attributes it to the signal that contributed most to
     the (wrong) predicted class — reusing the exact SHAP-style
     contributions prediction_logger.py already computed and stored
     that morning (top_signals), not a re-guess with a possibly
     different model.
  3. Adjusts that signal's weight: every 5th wrong attribution ->
     -10%; every 5th correct attribution -> +10% (clamped [0.2, 2.0]).
  4. Confidence decay: 3 wrong in a row (a true streak, separate from
     the rolling "every 5" counters above) -> halve immediately;
     restore to the pre-decay weight after 2 correct in a row.
  5. Updates source_credibility for every source that had a headline
     routed to NIFTY that day.
  6. Checks the retrain trigger (50+ new headlines / >20% cumulative
     weight drift / <38% accuracy over the last 10) and retrains the
     Phase 4 model (signals.train_weighted_model) if any fires.
  7. Generates tomorrow's forward-chain watchlist from today's actual
     market moves.

Run: python3 learning_engine.py
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import schema  # noqa: E402
import signals  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "phase3_prediction"))
import feature_engineering as fe  # noqa: E402

DECAY_STREAK = 3
RESTORE_STREAK = 2
ADJUSTMENT_EVERY = 5
ADJUSTMENT_STEP = 0.10

ERROR_REASON_TEMPLATES = {
    "sentiment": "Sentiment signal pointed the wrong way — headline tone didn't match the actual market reaction.",
    "headline_volume": "Headline volume was over-weighted relative to actual signal quality.",
    "stock_momentum": "Prior-day price momentum didn't carry through.",
    "market_sentiment": "Aggregate market-wide news sentiment was misleading.",
    "nifty_momentum": "NIFTY's own prior-day momentum didn't carry through.",
    "sp500": "Overnight S&P 500 signal was misread as the dominant driver.",
    "crude_oil": "Crude oil move wasn't weighted correctly relative to its actual impact.",
    "usdinr": "USD/INR move wasn't weighted correctly relative to its actual impact.",
    "india_vix": "India VIX signal was misread.",
    "day_of_week": "Day-of-week pattern didn't hold today.",
    "sector": "Sector-level pattern didn't hold today.",
}
NO_PRECEDENT_REASON = "No similar historical headline pattern found (max TF-IDF similarity < 0.15) — a genuinely novel event, not a known signal misfiring."


def fetch_recent_nifty(existing_prices):
    """Lightweight direct fetch for just NIFTY's most recent closes —
    the spec asks for a targeted yfinance pull here, not the full
    56-ticker phase1_nifty50.py refresh (that script is untouched,
    and re-running its 1-2 minute chunked download just to check one
    number would be wasteful)."""
    try:
        raw = yf.download("^NSEI", period="10d", progress=False, auto_adjust=True)
        fresh = raw["Close"]
        if hasattr(fresh, "iloc") and fresh.ndim > 1:
            fresh = fresh.iloc[:, 0]
        fresh.name = "NIFTY 50 Index"
    except Exception as e:
        print(f"  Warning: fresh NIFTY fetch failed ({e}), using existing price history only.")
        return existing_prices
    merged = existing_prices.copy()
    merged["NIFTY 50 Index"] = merged["NIFTY 50 Index"].combine_first(fresh) if "NIFTY 50 Index" in merged.columns else fresh
    for date, val in fresh.items():
        merged.loc[date, "NIFTY 50 Index"] = val
    return merged.sort_index()


def resolve_outcomes(conn, prices):
    print("Step 1: recording actual outcomes for unresolved predictions")
    unresolved = pd.read_sql(
        "SELECT date, prediction, confidence, top_signals, similar_past_events, news_date "
        "FROM prediction_log WHERE date NOT IN (SELECT date FROM prediction_outcomes)", conn,
    )
    if unresolved.empty:
        print("  Nothing new to resolve.")
        return []

    resolved = []
    for row in unresolved.itertuples(index=False):
        if not row.news_date:
            continue
        news_date = pd.Timestamp(row.news_date)
        actual_return = fe.next_session_return(prices, "NIFTY 50 Index", news_date)
        if actual_return is None:
            print(f"  {row.date}: next session close not available yet — will retry next run.")
            continue

        actual_label = fe.label_from_return(actual_return)
        correct = int(actual_label == row.prediction)
        error_magnitude = 0.0 if correct else abs(actual_return)

        top_signals = json.loads(row.top_signals) if row.top_signals else []
        similar_events = json.loads(row.similar_past_events) if row.similar_past_events else []
        main_error_signal = None
        if not correct:
            if not similar_events or max((e.get("similarity") or 0) for e in similar_events) < 0.15:
                main_error_signal = "no_precedent"
            elif top_signals:
                main_error_signal = top_signals[0]["signal"]

        conn.execute(
            "INSERT INTO prediction_outcomes (date, predicted, actual_direction, actual_pct_change, "
            "correct, error_magnitude, main_error_signal) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO NOTHING",
            (row.date, row.prediction, actual_label, actual_return, correct, error_magnitude, main_error_signal),
        )
        conn.commit()
        print(f"  {row.date}: predicted {row.prediction}, actual {actual_label} ({actual_return:+.2f}%) "
              f"-> {'CORRECT' if correct else 'WRONG (' + str(main_error_signal) + ')'}")
        resolved.append({
            "date": row.date, "news_date": row.news_date, "correct": correct,
            "main_error_signal": main_error_signal, "top_signals": top_signals,
        })
    return resolved


def adjust_signal_weights(conn, resolved):
    print("\nStep 2-4: signal weight adjustment + confidence decay")
    for outcome in resolved:
        if not outcome["top_signals"]:
            continue
        # The signal actually being judged this round: what actually
        # drove today's prediction (top_signals[0]) — win or lose,
        # that's the signal whose track record this outcome informs.
        signal_name = outcome["top_signals"][0]["signal"]
        if signal_name not in signals.SIGNAL_FEATURE_MAP:
            continue  # "sector" isn't a weight-adjustable signal

        row = conn.execute(
            "SELECT current_weight, consecutive_wrong, consecutive_correct, "
            "wrong_since_adjustment, correct_since_adjustment, pre_decay_weight FROM signal_weights WHERE signal_name = ?",
            (signal_name,),
        ).fetchone()
        weight, cons_wrong, cons_correct, wrong_adj, correct_adj, pre_decay = row
        now = datetime.now().isoformat()

        if outcome["correct"]:
            cons_wrong, cons_correct = 0, cons_correct + 1
            wrong_adj, correct_adj = wrong_adj, correct_adj + 1
        else:
            cons_wrong, cons_correct = cons_wrong + 1, 0
            wrong_adj, correct_adj = wrong_adj + 1, correct_adj

        conn.execute(
            "UPDATE signal_weights SET consecutive_wrong=?, consecutive_correct=?, "
            "wrong_since_adjustment=?, correct_since_adjustment=? WHERE signal_name=?",
            (cons_wrong, cons_correct, wrong_adj, correct_adj, signal_name),
        )
        conn.commit()

        # Step 4: confidence decay (checked first — it's the more urgent signal)
        if cons_wrong >= DECAY_STREAK and pre_decay is None:
            new_weight = signals.set_signal_weight(
                conn, signal_name, weight * 0.5, f"confidence decay: {DECAY_STREAK} wrong in a row",
                pre_decay_weight=weight, reset_streaks=None,
            )
            conn.execute("UPDATE signal_weights SET consecutive_wrong=0 WHERE signal_name=?", (signal_name,))
            conn.commit()
            conn.execute(
                "INSERT INTO prediction_errors (date, signal_type, error_reason, weight_before, weight_after, decay_triggered) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (outcome["date"], signal_name, f"{DECAY_STREAK} consecutive wrong predictions", weight, new_weight),
            )
            conn.commit()
            signals.record_weight_drift(new_weight - weight)
            print(f"  DECAY: {signal_name} {weight:.2f} -> {new_weight:.2f} (3 wrong in a row)")
            weight = new_weight
        elif cons_correct >= RESTORE_STREAK and pre_decay is not None:
            new_weight = signals.set_signal_weight(
                conn, signal_name, pre_decay, "restored after 2 consecutive correct", pre_decay_weight=None,
            )
            conn.execute("UPDATE signal_weights SET pre_decay_weight=NULL, consecutive_correct=0 WHERE signal_name=?", (signal_name,))
            conn.commit()
            conn.execute(
                "INSERT INTO prediction_errors (date, signal_type, error_reason, weight_before, weight_after, decay_triggered) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (outcome["date"], signal_name, "restored after 2 consecutive correct", weight, new_weight),
            )
            conn.commit()
            signals.record_weight_drift(new_weight - weight)
            print(f"  RESTORE: {signal_name} {weight:.2f} -> {new_weight:.2f} (2 correct in a row)")
            weight = new_weight

        # Step 3: the "every 5" rolling adjustment (independent of decay/restore above)
        if wrong_adj >= ADJUSTMENT_EVERY:
            new_weight = signals.set_signal_weight(
                conn, signal_name, weight * (1 - ADJUSTMENT_STEP), f"{ADJUSTMENT_EVERY} wrong attributions reached",
            )
            conn.execute("UPDATE signal_weights SET wrong_since_adjustment=0 WHERE signal_name=?", (signal_name,))
            conn.commit()
            signals.record_weight_drift(new_weight - weight)
            print(f"  -10%: {signal_name} {weight:.2f} -> {new_weight:.2f} ({ADJUSTMENT_EVERY} wrong attributions)")
        elif correct_adj >= ADJUSTMENT_EVERY:
            new_weight = signals.set_signal_weight(
                conn, signal_name, weight * (1 + ADJUSTMENT_STEP), f"{ADJUSTMENT_EVERY} correct attributions reached",
            )
            conn.execute("UPDATE signal_weights SET correct_since_adjustment=0 WHERE signal_name=?", (signal_name,))
            conn.commit()
            signals.record_weight_drift(new_weight - weight)
            print(f"  +10%: {signal_name} {weight:.2f} -> {new_weight:.2f} ({ADJUSTMENT_EVERY} correct attributions)")


def update_sources(conn, resolved, routed):
    print("\nStep 5: source credibility")
    for outcome in resolved:
        news_date = pd.Timestamp(outcome["news_date"])
        day_routed = routed[routed["ticker"] == signals.NIFTY_TICKER].copy()
        day_routed["published_at"] = pd.to_datetime(day_routed["published_at"], utc=True).dt.tz_localize(None)
        day_sources = day_routed[day_routed["published_at"].dt.normalize() == news_date]["source"].dropna().tolist()
        if not day_sources:
            continue
        changes = signals.update_source_credibility(conn, day_sources, bool(outcome["correct"]))
        for source, old_w, new_w, acc in changes:
            print(f"  {source}: weight {old_w:.2f} -> {new_w:.2f} (accuracy {acc:.0%})")


def check_and_retrain(conn):
    print("\nStep 6: retrain trigger check")
    should_retrain, reason = signals.check_retrain_trigger(conn)
    if not should_retrain:
        print("  No trigger fired.")
        return
    print(f"  Trigger fired: {reason}")

    # check_retrain_trigger compares against news_articles' total
    # labeled-headline count — mark_retrained must be given that same
    # count (not the smaller derived ticker-day row count below), or
    # the trigger would fire on every single run forever.
    total_headlines = conn.execute(
        "SELECT COUNT(*) FROM news_articles WHERE sentiment_label IS NOT NULL"
    ).fetchone()[0]

    routed = pd.read_csv(REPO_ROOT / "phase2_routing" / "phase2_routed_events.csv")
    prices = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_close_prices_all.csv", index_col="Date", parse_dates=True).sort_index()
    macro_csv = REPO_ROOT / "phase1_expansion" / "phase1_macro_data.csv"
    if macro_csv.exists():
        macro = pd.read_csv(macro_csv, index_col="Date", parse_dates=True).sort_index()
        prices = prices.join(macro, how="outer")
    sector_map = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_sector_map.csv")
    full_dataset = fe.build_dataset(routed, prices, sector_map)
    labeled = full_dataset[full_dataset["label"].notna()]

    old_accuracy = signals.load_learning_state().get("last_retrain_accuracy")

    unique_dates = sorted(labeled["date"].unique())
    split_idx = max(1, int(len(unique_dates) * 0.8))
    split_date = unique_dates[split_idx] if len(unique_dates) > split_idx else unique_dates[-1]
    train_slice = labeled[labeled["date"] < split_date]
    test_slice = labeled[labeled["date"] >= split_date]

    model, sector_encoder, label_encoder, weights = signals.train_weighted_model(labeled)
    signals.save_model_state(model, sector_encoder, label_encoder, fe.FEATURE_COLS,
                              {"trained_at": datetime.now().isoformat(), "n_rows": len(labeled), "trigger": reason})

    new_accuracy = None
    if len(test_slice) > 0 and train_slice["label"].nunique() >= 2:
        eval_model, eval_sector_enc, eval_label_enc, _ = signals.train_weighted_model(train_slice)
        X_test = fe.model_matrix(test_slice, eval_sector_enc)
        pred = eval_label_enc.inverse_transform(eval_model.predict(X_test))
        new_accuracy = float((pred == test_slice["label"].values).mean())

    signals.mark_retrained(total_headlines, new_accuracy)
    state = signals.load_learning_state()
    state.setdefault("retrain_history", []).append({
        "at": datetime.now().isoformat(), "trigger": reason,
        "old_accuracy": old_accuracy, "new_accuracy": new_accuracy,
        "signal_weights": weights, "n_rows": len(labeled),
    })
    signals.save_learning_state(state)
    print(f"  Retrained on {len(labeled)} rows. Old accuracy: {old_accuracy}. New accuracy: {new_accuracy}.")


def generate_forward_signals(conn, prices, routed):
    print("\nStep 7: forward-chain watchlist for tomorrow")
    today = prices.index.max()
    tomorrow = str((pd.Timestamp.now().normalize() + timedelta(days=1)).date())
    sector_map = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_sector_map.csv")

    def pct_change(col):
        series = prices[col].dropna() if col in prices.columns else pd.Series(dtype=float)
        if len(series) < 2:
            return None
        return (series.iloc[-1] / series.iloc[-2] - 1) * 100

    try:
        gold_raw = yf.download("GC=F", period="10d", progress=False, auto_adjust=True)["Close"]
        if hasattr(gold_raw, "iloc") and gold_raw.ndim > 1:
            gold_raw = gold_raw.iloc[:, 0]
        gold_change = (gold_raw.iloc[-1] / gold_raw.iloc[-2] - 1) * 100 if len(gold_raw) >= 2 else None
    except Exception:
        gold_change = None

    crude_change = pct_change("Crude Oil")
    sp500_change = pct_change("S&P 500")
    usdinr_change = pct_change("USD/INR")

    rbi_today = False
    if not routed.empty:
        r = routed.copy()
        r["published_at"] = pd.to_datetime(r["published_at"], utc=True).dt.tz_localize(None)
        rbi_rows = r[(r["topic"] == "RBI") & (r["published_at"].dt.normalize() == today)]
        rbi_today = len(rbi_rows) > 0
        rbi_headline = rbi_rows["headline"].iloc[0] if rbi_today else None

    signals_out = []
    if crude_change is not None and crude_change > 2:
        signals_out.append(("Aviation", None, "crude_oil_spike",
                             "Crude oil up >2% today — aviation stocks typically move on fuel-cost sensitivity. "
                             "NOTE: no aviation/airline tickers are tracked in phase1_sector_map.csv for this "
                             "project (NIFTY50 constituents don't include an airline) — logged as a sector-level "
                             "signal for future coverage, not attached to any specific stock.",
                             None, min(abs(crude_change) / 5, 1.0)))
    if rbi_today:
        banking = sector_map[sector_map["sector"] == "Banking"]
        for _, r in banking.iterrows():
            signals_out.append(("Banking", r["ticker"], "rbi_news",
                                 "RBI-related news today.", rbi_headline, 0.6))
    if sp500_change is not None and sp500_change < -1:
        it_stocks = sector_map[sector_map["sector"] == "IT"]
        for _, r in it_stocks.iterrows():
            signals_out.append(("IT", r["ticker"], "us_markets_fell",
                                 f"S&P 500 fell {sp500_change:.2f}% overnight — IT stocks have high US-revenue exposure.",
                                 None, min(abs(sp500_change) / 3, 1.0)))
    if gold_change is not None and gold_change > 1:
        for sector in ("FMCG", "Pharma"):
            for _, r in sector_map[sector_map["sector"] == sector].iterrows():
                signals_out.append((sector, r["ticker"], "risk_off_gold",
                                     f"Gold up {gold_change:.2f}% — classic risk-off signal, flagging defensive sectors.",
                                     None, min(abs(gold_change) / 3, 1.0)))
    if usdinr_change is not None and abs(usdinr_change) > 0.5:
        it_stocks = sector_map[sector_map["sector"] == "IT"]
        for _, r in it_stocks.iterrows():
            signals_out.append(("IT", r["ticker"], "usdinr_move",
                                 f"USD/INR moved {usdinr_change:+.2f}% — IT exporters are directly exposed.",
                                 None, min(abs(usdinr_change) / 1.5, 1.0)))

    for sector, stock, sig_type, reasoning, headline, conf in signals_out:
        conn.execute(
            "INSERT INTO forward_signals (date, sector, stock, signal_type, reasoning, source_headline, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tomorrow, sector, stock, sig_type, reasoning, headline, conf),
        )
    conn.commit()
    print(f"  Logged {len(signals_out)} forward signals for {tomorrow}")


def main(retrain_check_only=False):
    print("=" * 60)
    print("  PHASE 4: Afternoon learning cycle" if not retrain_check_only else "  PHASE 4: Retrain-trigger check (hourly)")
    print("=" * 60)

    schema.create_tables()
    conn = signals.get_conn()

    if retrain_check_only:
        # scheduler.py calls this every hour after routing runs — "every
        # 50 new headlines" from the spec's scheduler section is the
        # same trigger Step 6 checks, so it's one implementation, just
        # invoked on two cadences instead of duplicated.
        check_and_retrain(conn)
        conn.close()
        print("\nDONE")
        return

    prices = pd.read_csv(REPO_ROOT / "phase1_expansion" / "phase1_close_prices_all.csv", index_col="Date", parse_dates=True).sort_index()
    macro_csv = REPO_ROOT / "phase1_expansion" / "phase1_macro_data.csv"
    if macro_csv.exists():
        macro = pd.read_csv(macro_csv, index_col="Date", parse_dates=True).sort_index()
        prices = prices.join(macro, how="outer")
    prices = fetch_recent_nifty(prices)
    routed = pd.read_csv(REPO_ROOT / "phase2_routing" / "phase2_routed_events.csv")

    signals.get_signal_weights(conn)  # ensure default rows exist

    resolved = resolve_outcomes(conn, prices)
    if resolved:
        adjust_signal_weights(conn, resolved)
        update_sources(conn, resolved, routed)
    check_and_retrain(conn)
    generate_forward_signals(conn, prices, routed)

    conn.close()
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main(retrain_check_only="--retrain-check-only" in sys.argv)
