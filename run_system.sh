#!/bin/bash
# One command starts everything — the self-learning engine's scheduler,
# running unattended in the background.
#
# Deliberately NOT a bare "python3 scheduler.py &": that form ties the
# process to this shell session and kills it the moment the terminal
# closes, which defeats "runs automatically on its own." nohup + disown
# detaches it properly so it keeps running after you close the terminal.
set -e
cd "$(dirname "$0")"

export DYLD_LIBRARY_PATH="$(pwd)/venv/lib/python3.9/site-packages/torch/lib"  # macOS XGBoost/libomp workaround — see phase3_prediction/README.md
source venv/bin/activate

mkdir -p phase4_learning/logs
# -u: unbuffered stdout — without it, Python block-buffers when writing
# to a file instead of a terminal, so nohup's log stays empty for long
# stretches even while jobs are actually running.
nohup python3 -u phase4_learning/scheduler.py >> phase4_learning/logs/scheduler.log 2>&1 &
SCHEDULER_PID=$!
disown

echo $SCHEDULER_PID > phase4_learning/scheduler.pid

echo "System running (PID $SCHEDULER_PID). Pipeline collecting. Learning active."
echo "Logs: phase4_learning/logs/scheduler.log"
echo "To stop: kill \$(cat phase4_learning/scheduler.pid)"
echo "To see dashboard: streamlit run final_dashboard/dashboard.py"
