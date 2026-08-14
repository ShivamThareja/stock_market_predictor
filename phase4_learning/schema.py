"""
PHASE 4: Self-learning engine — database schema
Project: Global Financial News → Indian Stock Market Predictor

Adds 6 new tables to the EXISTING database — every other script in
this folder imports create_tables() before touching the DB, so it's
always safe to run standalone or as a side effect of import.

IMPORTANT CORRECTION FROM THE ORIGINAL SPEC: this project's real,
populated database is phase0_week3/news_pipeline.db (386 headlines,
everything Phase 0-3 read/write). There is a separate, empty (0-byte),
untracked phase0_week3/news.db sitting in the folder — a stray file,
not the project's database (nothing reads or writes it; it was
gitignored last session precisely because it isn't real project
output). Wiring new tables into news.db would silently build this
whole system on top of nothing. Every script in phase4_learning/
points at news_pipeline.db.

Only CREATE TABLE IF NOT EXISTS here — nothing in this file ever
ALTERs or writes to news_articles, the one existing table. Every
script in this folder only SELECTs from news_articles.

Run standalone to (re)create the tables without running anything else:
  python3 schema.py
"""

import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "phase0_week3" / "news_pipeline.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    date                    TEXT UNIQUE NOT NULL,
    timestamp               TEXT NOT NULL,
    prediction              TEXT NOT NULL,
    confidence              REAL,
    top_signals             TEXT,   -- JSON: [{signal, contribution}, ...]
    influential_headlines   TEXT,   -- JSON: [{headline, source, sentiment}, ...]
    similar_past_events     TEXT,   -- JSON: [{date, headline, similarity, outcome}, ...]
    signal_weights_snapshot TEXT,   -- JSON: {signal_name: weight, ...} at prediction time
    source_weights_snapshot TEXT,   -- JSON: {source_name: weight, ...} at prediction time
    news_date               TEXT    -- date of the underlying feature row (which session this predicts) — extra column beyond the original spec's list, needed so learning_engine.py can resolve the right session; this table didn't exist before this session so adding a column here touches nothing pre-existing
);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT UNIQUE NOT NULL,
    predicted           TEXT NOT NULL,
    actual_direction    TEXT NOT NULL,
    actual_pct_change   REAL NOT NULL,
    correct             INTEGER NOT NULL,   -- 0/1
    error_magnitude     REAL,               -- 0 if correct; |predicted class boundary - actual %| if wrong
    main_error_signal   TEXT                -- signal name that most drove the wrong call, NULL if correct
);

CREATE TABLE IF NOT EXISTS prediction_errors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    error_reason    TEXT NOT NULL,
    weight_before   REAL NOT NULL,
    weight_after    REAL NOT NULL,
    decay_triggered INTEGER NOT NULL DEFAULT 0  -- 0/1
);

CREATE TABLE IF NOT EXISTS signal_weights (
    signal_name             TEXT PRIMARY KEY,
    current_weight          REAL NOT NULL DEFAULT 1.0,
    consecutive_wrong       INTEGER NOT NULL DEFAULT 0,   -- TRUE streak (resets on any opposite outcome) — drives Step 4 decay/restore
    consecutive_correct     INTEGER NOT NULL DEFAULT 0,   -- TRUE streak — drives Step 4 restore
    wrong_since_adjustment   INTEGER NOT NULL DEFAULT 0,  -- rolling tally toward the next -10% (Step 3), independent of the streak above
    correct_since_adjustment INTEGER NOT NULL DEFAULT 0,  -- rolling tally toward the next +10% (Step 3)
    pre_decay_weight        REAL,          -- set when confidence decay fires, so we know what to restore to
    last_updated            TEXT,
    weight_history_json     TEXT           -- JSON list of {date, weight, reason}
);

CREATE TABLE IF NOT EXISTS source_credibility (
    source_name         TEXT PRIMARY KEY,
    total_predictions    INTEGER NOT NULL DEFAULT 0,
    correct_predictions  INTEGER NOT NULL DEFAULT 0,
    accuracy_rate        REAL,
    current_weight       REAL NOT NULL DEFAULT 1.0,
    last_updated         TEXT
);

CREATE TABLE IF NOT EXISTS forward_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    sector          TEXT,
    stock           TEXT,
    signal_type     TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    source_headline TEXT,
    confidence      REAL
);
"""


# CREATE TABLE IF NOT EXISTS won't add columns to a table that
# already exists from an earlier run — these ALTERs (same
# try/except pattern week4_finbert.py already uses for its own
# schema evolution) cover that, purely additive either way.
EXTRA_COLUMNS = [
    ("prediction_log", "news_date", "TEXT"),
    ("signal_weights", "wrong_since_adjustment", "INTEGER NOT NULL DEFAULT 0"),
    ("signal_weights", "correct_since_adjustment", "INTEGER NOT NULL DEFAULT 0"),
]


def create_tables(db_file=DB_FILE):
    conn = sqlite3.connect(db_file)
    conn.executescript(SCHEMA)
    for table, column, coltype in EXTRA_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        except sqlite3.OperationalError:
            pass  # already added by a previous run
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_tables()
    print(f"Tables ready in {DB_FILE}")
