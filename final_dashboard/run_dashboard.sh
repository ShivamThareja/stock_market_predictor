#!/bin/bash
# Launches the dashboard with the venv activated and (on macOS
# without Homebrew's libomp) XGBoost's OpenMP dependency resolved
# via the copy already bundled inside torch. See README.md.
set -e
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH="$(pwd)/../venv/lib/python3.9/site-packages/torch/lib"
source ../venv/bin/activate
streamlit run dashboard.py
