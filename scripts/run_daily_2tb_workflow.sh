#!/usr/bin/env bash
# ==============================================================================
# scripts/run_daily_2tb_workflow.sh
# ==============================================================================
# 2TB daily automation wrapper
#
#   1. Generate today's predictions (America/Denver date)
#   2. Grade all ungraded prior dates
#
# Safe to re-run — prediction step skips if file already exists.
# All stdout+stderr is logged to logs/prediction_runs/ and logs/grading_runs/.
#
# Suggested cron schedule (run from repo root with the project venv):
#
#   # 11:00 AM Denver — generate today's predictions + grade ungraded prior dates
#   0 11 * * *  cd /root/2tb-model && bash scripts/run_daily_2tb_workflow.sh
#
#   # 02:00 AM Denver — safety-net grader for yesterday
#   0 2 * * *   cd /root/2tb-model && \
#                 /root/2tb-model/.venv/bin/python3 scripts/grade_predictions.py \
#                 --date "$(TZ=America/Denver date -d 'yesterday' +\%Y-\%m-\%d)" \
#                 >> logs/grading_runs/$(TZ=America/Denver date +\%Y\%m\%d).log 2>&1
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Use America/Denver for date — consistent with the dashboard
DATE_TODAY="$(TZ=America/Denver date +%Y-%m-%d)"

PREDICTION_LOG_DIR="$REPO_ROOT/logs/prediction_runs"
GRADING_LOG_DIR="$REPO_ROOT/logs/grading_runs"
mkdir -p "$PREDICTION_LOG_DIR" "$GRADING_LOG_DIR"

PYTHON="$REPO_ROOT/.venv/bin/python3"
PREDICTION_SCRIPT="$REPO_ROOT/scripts/run_daily_predictions.py"
GRADING_SCRIPT="$REPO_ROOT/scripts/grade_predictions.py"

FAILED=0

# ── 1. Generate predictions ───────────────────────────────────────────────────
echo "=== $(TZ=America/Denver date '+%Y-%m-%d %H:%M:%S %Z') — Running daily predictions for $DATE_TODAY ==="

if "$PYTHON" "$PREDICTION_SCRIPT" --date "$DATE_TODAY" \
    >> "$PREDICTION_LOG_DIR/${DATE_TODAY//-/}.log" 2>&1; then
    echo "=== Predictions completed successfully ==="
else
    EXIT_CODE=$?
    echo "=== FAIL: Predictions exited with code $EXIT_CODE ===" >&2
    echo "  Log: $PREDICTION_LOG_DIR/${DATE_TODAY//-/}.log"
    FAILED=1
fi

# ── 2. Grade all ungraded prior dates ────────────────────────────────────────
echo "=== Grading all ungraded prior dates ==="

if "$PYTHON" "$GRADING_SCRIPT" --all-ungraded \
    >> "$GRADING_LOG_DIR/${DATE_TODAY//-/}.log" 2>&1; then
    echo "=== Grading completed successfully ==="
else
    EXIT_CODE=$?
    echo "=== FAIL: Grading exited with code $EXIT_CODE ===" >&2
    echo "  Log: $GRADING_LOG_DIR/${DATE_TODAY//-/}.log"
    FAILED=1
fi

echo "=== $(TZ=America/Denver date '+%Y-%m-%d %H:%M:%S %Z') — 2TB daily workflow finished ==="

if [ "$FAILED" -ne 0 ]; then
    echo "⚠️  One or more steps failed. Check logs above." >&2
    exit 1
fi
