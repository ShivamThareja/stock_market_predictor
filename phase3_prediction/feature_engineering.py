"""
Shared feature-engineering for Phase 3.
Used by both `phase3_xgboost_model.py` (training/evaluation) and the
final dashboard's "live prediction" panel — pulled out into its own
module so the two never compute features a different way.

Everything here is pure data transformation (no printing, no file
I/O beyond the functions' own return values) so it's safe to import
from a Streamlit app without side effects firing on import.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FLAT_THRESHOLD = 0.3   # % move — below this counts as FLAT, not UP/DOWN

# Market-wide series attached to EVERY ticker's row (not just that
# ticker's own price history) — the "overnight signal" / cross-
# timezone features. Column names match phase1_macro_data.py's
# output plus "NIFTY 50 Index" from phase1_close_prices_all.csv,
# which is already in `prices` without any extra merge needed.
MARKET_SERIES = {
    "NIFTY 50 Index": "nifty_market_prior_1d_return",
    "S&P 500": "sp500_prior_1d_return",
    "Crude Oil": "crude_oil_prior_1d_return",
    "USD/INR": "usdinr_prior_1d_return",
    "India VIX": "india_vix_prior_1d_return",
}

FEATURE_COLS = [
    "n_headlines", "pct_positive", "pct_negative", "pct_neutral",
    "avg_confidence", "net_sentiment",
    "prior_1d_return", "prior_3d_return", "market_net_sentiment",
    "day_of_week",
] + list(MARKET_SERIES.values())
SENTIMENT_SIGN = {"positive": 1, "negative": -1, "neutral": 0}


def next_session_return(prices, name, news_date):
    """
    Baseline = most recent close at/before news_date.
    Target   = first close strictly after news_date.
    Returns None if either side is missing (e.g. news too recent for
    a next close to exist yet, or ticker has no price history that
    far back) — those rows stay "in-flight": a real prediction whose
    outcome isn't known yet, not a training example.
    """
    if name not in prices.columns:
        return None
    series = prices[name].dropna()
    before = series[series.index <= news_date]
    after = series[series.index > news_date]
    if len(before) == 0 or len(after) == 0:
        return None
    baseline = before.iloc[-1]
    target = after.iloc[0]
    if baseline == 0 or pd.isna(baseline) or pd.isna(target):
        return None
    return (target / baseline - 1) * 100


def momentum_features(prices, name, news_date):
    """Prior 1-day / 3-day returns going INTO news_date — always
    computable from history, unlike next_session_return."""
    if name not in prices.columns:
        return pd.Series({"prior_1d_return": np.nan, "prior_3d_return": np.nan})
    series = prices[name].dropna()
    before = series[series.index <= news_date]
    if len(before) < 4:
        return pd.Series({"prior_1d_return": np.nan, "prior_3d_return": np.nan})
    latest = before.iloc[-1]
    return pd.Series({
        "prior_1d_return": (latest / before.iloc[-2] - 1) * 100,
        "prior_3d_return": (latest / before.iloc[-4] - 1) * 100,
    })


def label_from_return(r):
    if pd.isna(r):
        return None
    if r > FLAT_THRESHOLD:
        return "UP"
    if r < -FLAT_THRESHOLD:
        return "DOWN"
    return "FLAT"


def add_market_features(dataset, prices):
    """Prior-1-day return for each market-wide series in MARKET_SERIES
    (NIFTY itself, S&P 500, crude oil, USD/INR, India VIX), attached
    to every row regardless of which ticker that row is about — plus
    day-of-week. A series missing from `prices` (e.g. macro data was
    never fetched, or yfinance failed on one ticker) just contributes
    an all-zero column rather than breaking the whole build."""
    for series_name, col in MARKET_SERIES.items():
        if series_name in prices.columns:
            mom = dataset["date"].apply(lambda d: momentum_features(prices, series_name, d))
            dataset[col] = mom["prior_1d_return"].fillna(0.0)
        else:
            dataset[col] = 0.0
    dataset["day_of_week"] = dataset["date"].dt.dayofweek
    return dataset


def build_dataset(routed, prices, sector_map):
    """
    Turn Phase 2's headline-level routed events into one row per
    (ticker, date): sentiment aggregates + price momentum + market-
    wide sentiment/macro signals + (if resolvable) the real next-
    session return and UP/DOWN/FLAT label.

    Rows where the next session hasn't closed yet keep label=None /
    next_session_return=NaN rather than being dropped — the training
    script filters those out (dataset["label"].notna()), the
    dashboard's live-prediction panel uses exactly those rows.
    """
    routed = routed.copy()
    routed["date"] = pd.to_datetime(routed["published_at"], utc=True).dt.tz_localize(None).dt.normalize()
    routed["sentiment_sign"] = routed["sentiment_label"].map(SENTIMENT_SIGN)
    routed["weighted_sign"] = routed["sentiment_sign"] * routed["sentiment_confidence"]

    grouped = routed.groupby(["ticker", "date"])
    features = grouped.agg(
        n_headlines=("headline", "count"),
        n_positive=("sentiment_label", lambda s: (s == "positive").sum()),
        n_negative=("sentiment_label", lambda s: (s == "negative").sum()),
        n_neutral=("sentiment_label", lambda s: (s == "neutral").sum()),
        avg_confidence=("sentiment_confidence", "mean"),
        net_sentiment=("weighted_sign", "mean"),
    ).reset_index()
    features["pct_positive"] = features["n_positive"] / features["n_headlines"]
    features["pct_negative"] = features["n_negative"] / features["n_headlines"]
    features["pct_neutral"] = features["n_neutral"] / features["n_headlines"]

    sector_lookup = sector_map[["ticker", "sector"]].drop_duplicates()
    features = features.merge(sector_lookup, on="ticker", how="left")

    unique_pairs = routed[["ticker", "company_name", "date"]].drop_duplicates()
    unique_pairs["next_session_return"] = [
        next_session_return(prices, row.company_name, row.date) for row in unique_pairs.itertuples(index=False)
    ]
    unique_pairs["label"] = unique_pairs["next_session_return"].apply(label_from_return)

    dataset = features.merge(
        unique_pairs[["ticker", "date", "next_session_return", "label"]],
        on=["ticker", "date"], how="left",
    )
    dataset = dataset.sort_values("date").reset_index(drop=True)

    name_lookup = sector_map[["ticker", "name"]].drop_duplicates().set_index("ticker")["name"]
    momentum = dataset.apply(
        lambda row: momentum_features(prices, name_lookup.get(row["ticker"], row["ticker"]), row["date"]), axis=1
    )
    dataset = pd.concat([dataset, momentum], axis=1)

    index_sentiment = features[features["ticker"] == "^NSEI"][["date", "net_sentiment"]].rename(
        columns={"net_sentiment": "market_net_sentiment"}
    )
    dataset = dataset.merge(index_sentiment, on="date", how="left")
    dataset["market_net_sentiment"] = dataset["market_net_sentiment"].fillna(0.0)
    dataset["prior_1d_return"] = dataset["prior_1d_return"].fillna(0.0)
    dataset["prior_3d_return"] = dataset["prior_3d_return"].fillna(0.0)

    dataset = add_market_features(dataset, prices)

    return dataset


def make_model():
    """Same hyperparameters everywhere a model gets trained — the
    time-split evaluation in phase3_xgboost_model.py and the
    all-data live model in the dashboard should differ only in what
    rows they're fit on, not in architecture."""
    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )


def fit_encoders(dataset):
    """Fit sector/label encoders across the full dataset (train+test,
    or train+live) so a category unseen in the fit slice never
    breaks .transform() on the other slice."""
    sector_encoder = LabelEncoder()
    sector_encoder.fit(dataset["sector"].fillna("Unknown"))
    label_encoder = LabelEncoder()
    label_encoder.fit(["DOWN", "FLAT", "UP"])
    return sector_encoder, label_encoder


def model_matrix(dataset, sector_encoder):
    dataset = dataset.copy()
    dataset["sector_enc"] = sector_encoder.transform(dataset["sector"].fillna("Unknown"))
    return dataset[FEATURE_COLS + ["sector_enc"]]
