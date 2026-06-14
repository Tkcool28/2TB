#!/usr/bin/env python3
"""
grade_predictions.py
====================

Grade archived 2TB daily predictions against actual game outcomes.

Reads:
  - predictions/<YYYY>/<YYYYMMDD>_predictions.csv
  - data/raw/full_game_logs_<YYYY>.json  (primary actuals source)

Writes:
  - results/<YYYY>/<YYYYMMDD>_results.csv
  - results/<YYYY>/<YYYYMMDD>_results_summary.json
  - logs/grading_runs/<YYYYMMDD>.log

Usage:
    python scripts/grade_predictions.py --date 2025-06-01
    python scripts/grade_predictions.py --all-ungraded

Grading rules:
  - Match predictions to actuals by player_id only (normalized to int).
  - If actual data is missing for a player, that row is marked
    grading_status="ungraded" and excluded from hit-rate metrics.
  - Total bases = hits + doubles + 2*triples + 3*home_runs
    (equivalently: singles + 2*doubles + 3*triples + 4*home_runs).
  - hit_2tb = True when actual_total_bases >= 2.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = Path(os.environ.get("_GP_PREDICTIONS_DIR", str(REPO_ROOT / "predictions")))
RESULTS_DIR = Path(os.environ.get("_GP_RESULTS_DIR", str(REPO_ROOT / "results")))
LOGS_DIR = Path(os.environ.get("_GP_LOGS_DIR", str(REPO_ROOT / "logs" / "grading_runs")))
RAW_DIR = Path(os.environ.get("_GP_RAW_DIR", str(REPO_ROOT / "data" / "raw")))

PROBA_FIELD = "predicted_proba_2tb"
FALLBACK_PROBA_FIELD = "ensemble_probability"

REQUIRED_PRED_COLS = {"date", "player_id"}  # predicted_proba_2tb or ensemble_probability checked at runtime

# Extra columns written to the results CSV (appended to original fields)
RESULT_EXTRA_COLS = [
    "actual_total_bases",
    "actual_hits",
    "actual_doubles",
    "actual_triples",
    "actual_home_runs",
    "hit_2tb",
    "grading_status",
    "unmatched_reason",
    "graded_timestamp",
]

SUMMARY_KEYS = [
    "game_date",
    "total_predictions",
    "graded_predictions",
    "ungraded_predictions",
    "total_hits_2tb",
    "hit_rate",
    "top_10_hit_rate",
    "top_20_hit_rate",
    "top_prediction_hit",
    "unmatched_examples",
    "grading_runtime_seconds",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_player_id(raw) -> int | None:
    """Normalize various player-id forms to int.

    Handles: 592450, "592450", "ID592450", "592450.0"
    Returns None if the value cannot be parsed.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # Strip common prefixes
    if s.upper().startswith("ID"):
        s = s[2:]
    # Strip trailing .0 from float-like strings
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def compute_total_bases(hits: int, doubles: int, triples: int, home_runs: int) -> int:
    """Standard total-bases formula: 1B + 2*2B + 3*3B + 4*HR.

    Note: hits already includes doubles, triples, and HR, so
    TB = (hits - doubles - triples - HR) + 2*doubles + 3*triples + 4*HR
       = hits + doubles + 2*triples + 3*home_runs
    """
    singles = hits - doubles - triples - home_runs
    return singles + 2 * doubles + 3 * triples + 4 * home_runs


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Reset handlers so repeated calls don't duplicate
    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_predictions(date_str: str) -> tuple[list[dict], list[str]]:
    """Load prediction CSV. Returns (rows, fieldnames)."""
    date_compact = date_str.replace("-", "")
    year = date_str[:4]
    csv_path = PREDICTIONS_DIR / year / f"{date_compact}_predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames, csv_path


def load_actuals(date_str: str) -> dict[int, dict]:
    """Build player_id -> actual-stats dict from full_game_logs_YYYY.json.

    Returns empty dict if the source file is missing for that year.
    """
    year = date_str[:4]
    source_path = RAW_DIR / f"full_game_logs_{year}.json"
    if not source_path.is_file():
        logging.warning("Game-log source not found: %s", source_path)
        return {}

    with source_path.open() as f:
        games = json.load(f)

    actuals: dict[int, dict] = {}
    for row in games:
        if row.get("date") != date_str:
            continue
        pid = normalize_player_id(row.get("player_id"))
        if pid is None:
            continue
        hits = int(row.get("hits", 0))
        doubles = int(row.get("doubles", 0))
        triples = int(row.get("triples", 0))
        hr = int(row.get("home_runs", 0))
        tb = compute_total_bases(hits, doubles, triples, hr)
        # If a player has multiple entries (e.g. pinch-hit), sum them
        if pid in actuals:
            existing = actuals[pid]
            existing["actual_hits"] += hits
            existing["actual_doubles"] += doubles
            existing["actual_triples"] += triples
            existing["actual_home_runs"] += hr
            existing["actual_total_bases"] = compute_total_bases(
                existing["actual_hits"],
                existing["actual_doubles"],
                existing["actual_triples"],
                existing["actual_home_runs"],
            )
        else:
            actuals[pid] = {
                "actual_total_bases": tb,
                "actual_hits": hits,
                "actual_doubles": doubles,
                "actual_triples": triples,
                "actual_home_runs": hr,
            }

    logging.info("Loaded actuals for %d players on %s", len(actuals), date_str)
    return actuals


def resolve_proba_field(row: dict) -> str:
    """Return the probability field name present in the row."""
    if PROBA_FIELD in row:
        return PROBA_FIELD
    if FALLBACK_PROBA_FIELD in row:
        return FALLBACK_PROBA_FIELD
    return PROBA_FIELD  # default; will raise KeyError if missing


def grade_date(date_str: str) -> None:
    """Grade predictions for a single date."""
    start = time.monotonic()
    date_compact = date_str.replace("-", "")
    year = date_str[:4]

    out_csv = RESULTS_DIR / year / f"{date_compact}_results.csv"
    out_json = RESULTS_DIR / year / f"{date_compact}_results_summary.json"
    log_path = LOGS_DIR / f"{date_compact}.log"

    setup_logging(log_path)
    logging.info("=== grade_predictions start for %s ===", date_str)

    # --- Load predictions ---
    try:
        pred_rows, pred_fields, pred_path = load_predictions(date_str)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        sys.exit(1)

    logging.info("Loaded %d prediction rows from %s", len(pred_rows), pred_path)

    # Validate required columns
    missing_cols = REQUIRED_PRED_COLS - set(pred_fields)
    if missing_cols:
        logging.error("Prediction CSV missing required columns: %s", missing_cols)
        sys.exit(1)

    # Validate that at least one probability field exists
    if PROBA_FIELD not in pred_fields and FALLBACK_PROBA_FIELD not in pred_fields:
        logging.error(
            "Prediction CSV missing probability field: need '%s' or '%s'",
            PROBA_FIELD, FALLBACK_PROBA_FIELD,
        )
        sys.exit(1)

    # --- Load actuals ---
    actuals = load_actuals(date_str)

    if not actuals:
        logging.error(
            "No actual game-log data available for %s. "
            "Aborting — no results will be written so this date remains "
            "eligible for a future --all-ungraded retry. "
            "(Source file may be missing or contain no rows for this date.)",
            date_str,
        )
        sys.exit(1)

    # --- Grade each row ---
    graded_rows: list[dict] = []
    unmatched_examples: list[dict] = []
    graded_count = 0
    ungraded_count = 0
    hit_count = 0

    for row in pred_rows:
        pid_raw = row.get("player_id")
        pid = normalize_player_id(pid_raw)

        if pid is None:
            # Cannot even normalize the ID — mark ungraded
            extra = {
                "actual_total_bases": None,
                "actual_hits": None,
                "actual_doubles": None,
                "actual_triples": None,
                "actual_home_runs": None,
                "hit_2tb": None,
                "grading_status": "ungraded",
                "unmatched_reason": f"unparseable player_id: {pid_raw!r}",
                "graded_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            graded_rows.append({**row, **extra})
            ungraded_count += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append({"player_id": pid_raw, "reason": "unparseable player_id"})
            continue

        actual = actuals.get(pid)
        if actual is None:
            extra = {
                "actual_total_bases": None,
                "actual_hits": None,
                "actual_doubles": None,
                "actual_triples": None,
                "actual_home_runs": None,
                "hit_2tb": None,
                "grading_status": "ungraded",
                "unmatched_reason": "no game-log data for player on this date",
                "graded_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            graded_rows.append({**row, **extra})
            ungraded_count += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append({"player_id": pid, "reason": "no game-log data for player on this date"})
            continue

        hit_2tb = actual["actual_total_bases"] >= 2
        if hit_2tb:
            hit_count += 1

        extra = {
            **{k: actual[k] for k in ["actual_total_bases", "actual_hits", "actual_doubles", "actual_triples", "actual_home_runs"]},
            "hit_2tb": hit_2tb,
            "grading_status": "graded",
            "unmatched_reason": None,
            "graded_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        graded_rows.append({**row, **extra})
        graded_count += 1

    runtime = time.monotonic() - start

    # --- Write results CSV ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Build fieldnames: original fields + extras (avoiding duplicates)
    seen = set()
    fieldnames = []
    for f in pred_fields + RESULT_EXTRA_COLS:
        if f not in seen:
            seen.add(f)
            fieldnames.append(f)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(graded_rows)
    logging.info("Wrote results CSV: %s (%d rows)", out_csv, len(graded_rows))

    # --- Compute summary metrics (graded rows only) ---
    graded_only = [r for r in graded_rows if r["grading_status"] == "graded"]

    # Sort by probability descending for top-N metrics
    def _proba_key(r):
        field = resolve_proba_field(r)
        try:
            return float(r.get(field, 0))
        except (ValueError, TypeError):
            return 0.0

    sorted_graded = sorted(graded_only, key=_proba_key, reverse=True)

    def top_n_rate(n: int) -> float | None:
        if not sorted_graded:
            return None
        slice_rows = sorted_graded[:n]
        if not slice_rows:
            return None
        hits = sum(1 for r in slice_rows if r.get("hit_2tb") is True)
        return round(hits / len(slice_rows), 4)

    hit_rate = round(hit_count / graded_count, 4) if graded_count else None
    top_prediction_hit = sorted_graded[0].get("hit_2tb") if sorted_graded else None

    summary = {
        "game_date": date_str,
        "total_predictions": len(pred_rows),
        "graded_predictions": graded_count,
        "ungraded_predictions": ungraded_count,
        "total_hits_2tb": hit_count,
        "hit_rate": hit_rate,
        "top_10_hit_rate": top_n_rate(10),
        "top_20_hit_rate": top_n_rate(20),
        "top_prediction_hit": top_prediction_hit,
        "unmatched_examples": unmatched_examples[:5],
        "grading_runtime_seconds": round(runtime, 2),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info("Wrote summary JSON: %s", out_json)
    logging.info(
        "Done: %d graded, %d ungraded, hit_rate=%s, runtime=%.2fs",
        graded_count, ungraded_count, hit_rate, runtime,
    )


def find_ungraded_dates(through_date: str | None = None) -> list[str]:
    """Return sorted list of dates that have predictions but no results CSV.

    If through_date is given (YYYY-MM-DD), only include dates <= through_date.
    If through_date is None, defaults to yesterday in America/Denver (never today).
    """
    if through_date is None:
        tz = ZoneInfo("America/Denver")
        from datetime import timedelta
        through_date = (datetime.now(tz).date() - timedelta(days=1)).isoformat()

    dates: list[str] = []
    for csv_path in PREDICTIONS_DIR.rglob("*_predictions.csv"):
        # Extract YYYYMMDD from filename
        stem = csv_path.stem  # e.g. "20250601_predictions"
        date_compact = stem.replace("_predictions", "")
        if len(date_compact) != 8 or not date_compact.isdigit():
            continue
        d = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        if d > through_date:
            continue
        year = date_compact[:4]
        result_csv = RESULTS_DIR / year / f"{date_compact}_results.csv"
        if not result_csv.is_file():
            dates.append(d)
    dates.sort()
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade 2TB model daily predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date to grade (YYYY-MM-DD)")
    group.add_argument(
        "--all-ungraded",
        action="store_true",
        help="Grade all dates that have predictions but no results yet",
    )
    parser.add_argument(
        "--through-date",
        default=None,
        help=(
            "Only grade dates <= this date (YYYY-MM-DD). "
            "Defaults to yesterday in America/Denver when used with --all-ungraded."
        ),
    )
    args = parser.parse_args()

    if args.date:
        grade_date(args.date)
    else:
        dates = find_ungraded_dates(through_date=args.through_date)
        if not dates:
            print("No ungraded dates found.")
            return
        print(f"Found {len(dates)} ungraded date(s): {', '.join(dates)}")
        for date_str in dates:
            print(f"\n--- Grading {date_str} ---")
            grade_date(date_str)


if __name__ == "__main__":
    main()
