#!/usr/bin/env bash

# Wrapper script to run the 2TB daily workflow:
#   1. Generate today's predictions
#   2. Grade yesterday's predictions (if any)
#
# This script logs its output to logs/prediction_runs/<date>.log and
# logs grading output to logs/grading_runs/<date>.log.  It is safe to run
# repeatedly – it will skip already‑generated files to avoid overwriting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE_TODAY=$(date +%Y-%m-%d)

# 1. Run predictions
echo "=== Running daily predictions for $DATE_TODAY ==="
python3 "$SCRIPT_DIR/run_daily_predictions.py" --date "$DATE_TODAY"

# 2. Grade yesterday's predictions (if any)
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d || date -v-1d +%Y-%m-%d)
echo "=== Grading predictions for $YESTERDAY ==="
python3 "$SCRIPT_DIR/grade_predictions.py" --date "$YESTERDAY" || echo "No predictions to grade for $YESTERDAY"

echo "=== 2TB daily workflow completed ==="
