"""
PHASE 4: Self-learning engine — scheduler
Project: Global Financial News → Indian Stock Market Predictor

Single long-running process, orchestrating everything on a schedule.
Every job is a SUBPROCESS call to an existing or new script — this
process never imports week3_pipeline.py / week4_finbert.py /
phase2_news_routing.py (the protected files), it just invokes them
exactly as a human running them by hand would, on a timer instead of
manually. No cron job involved — APScheduler holds the schedule
in memory for as long as this process runs (see run_system.sh for
how it's kept running unattended).

Schedule (all times IST):
  :07 past every hour  -> week3_pipeline.py --once      (fetch news)
  :15 past every hour  -> week4_finbert.py               (label sentiment)
  :18 past every hour  -> week4_label_rules.py            (rule corrections — ADDED, see note below)
  :20 past every hour  -> phase2_news_routing.py          (route to stocks)
  :25 past every hour  -> learning_engine.py --retrain-check-only  (the "every 50 new headlines" trigger)
  7:00 AM              -> prediction_logger.py            (morning prediction)
  4:00 PM              -> learning_engine.py               (full afternoon learning cycle)

Addition beyond the original spec: week4_label_rules.py wasn't in the
requested hourly list, but skipping it would mean every hour's new
headlines go out with raw FinBERT labels forever, silently regressing
the label-quality work from the previous round for anything collected
after this system starts. Runs between week4_finbert.py and the
routing step, same order prediction_logger.py's own pre-prediction
refresh already uses.

Run: python3 scheduler.py
(stays in the foreground — see run_system.sh to run it detached)
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PYTHON = sys.executable
IST = ZoneInfo("Asia/Kolkata")


def run(label, cwd, args):
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"\n[{ts}] {label}")
    try:
        result = subprocess.run([PYTHON] + args, cwd=cwd, capture_output=True, text=True, timeout=1800)
        tail = (result.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")
        if result.returncode != 0:
            print(f"    WARNING: exited {result.returncode}: {(result.stderr or '')[-300:]}")
    except subprocess.TimeoutExpired:
        print("    WARNING: timed out")
    except Exception as e:
        print(f"    WARNING: {e}")


def fetch_news():
    run("week3_pipeline.py --once", REPO_ROOT / "phase0_week3", ["week3_pipeline.py", "--once"])


def label_sentiment():
    run("week4_finbert.py", REPO_ROOT / "phase0_week4", ["week4_finbert.py"])


def apply_label_rules():
    run("week4_label_rules.py", REPO_ROOT / "phase0_week4", ["week4_label_rules.py"])


def route_news():
    run("phase2_news_routing.py", REPO_ROOT / "phase2_routing", ["phase2_news_routing.py"])


def retrain_check():
    run("learning_engine.py --retrain-check-only", HERE, ["learning_engine.py", "--retrain-check-only"])


def morning_prediction():
    run("prediction_logger.py", HERE, ["prediction_logger.py"])


def afternoon_learning():
    run("learning_engine.py", HERE, ["learning_engine.py"])


def main():
    print("=" * 60)
    print("  PHASE 4: Scheduler starting")
    print("=" * 60)
    print(f"  Timezone: Asia/Kolkata")
    print(f"  Hourly: fetch :07, label :15, rules :18, route :20, retrain-check :25")
    print(f"  Daily:  predict 7:00 AM, learn 4:00 PM")
    print("  Press Ctrl+C to stop (or see run_system.sh to run detached).\n")

    scheduler = BlockingScheduler(timezone=IST)
    scheduler.add_job(fetch_news, "cron", minute=7, id="fetch_news")
    scheduler.add_job(label_sentiment, "cron", minute=15, id="label_sentiment")
    scheduler.add_job(apply_label_rules, "cron", minute=18, id="apply_label_rules")
    scheduler.add_job(route_news, "cron", minute=20, id="route_news")
    scheduler.add_job(retrain_check, "cron", minute=25, id="retrain_check")
    scheduler.add_job(morning_prediction, "cron", hour=7, minute=0, id="morning_prediction")
    scheduler.add_job(afternoon_learning, "cron", hour=16, minute=0, id="afternoon_learning")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\nScheduler stopped.")


if __name__ == "__main__":
    main()
