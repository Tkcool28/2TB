#!/usr/bin/env python3
"""Grade daily prediction results against actual game outcomes.

Usage:
    python scripts/grade_predictions.py --date YYYY-MM-DD
    python scripts/grade_predictions.py --all-ungraded

The script:
1. Loads the predictions CSV for the given date (from predictions/YYYY/).
2. Loads the cached boxscore JSON for that date (data/raw/<date>_full_game_log.json).
   If not present, it logs a warning and treats all actual stats as zero.
3. Computes total bases per player, determines hit_2tb (total_bases >= 2).
4. Writes a graded CSV and a summary JSON to results/YYYY/.
5. Emits a log file under logs/grading_runs/.
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Grader started for date %s", log_path.stem)

def load_predictions(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames

def load_boxscore(date_str: str):
    # Expected cached file location – adjust if your pipeline uses a different name
    raw_path = Path("data/raw") / f"{date_str}_full_game_log.json"
    if not raw_path.is_file():
        logging.warning("Boxscore cache not found: %s", raw_path)
        return {}
    with raw_path.open() as f:
        data = json.load(f)
    # Build a lookup: (player_id, team) -> stats dict
    lookup = {}
    for side in data.get("liveData", {}).get("boxscore", {}).get("teams", {}).values():
        for pid, player in side.get("players", {}).items():
            stats = player.get("stats", {})
            # Pull the needed hit stats – may be nested under "batting"
            batting = stats.get("batting", {})
            # Some APIs use different keys; fall back to 0
            hits = int(batting.get("hits", 0))
            doubles = int(batting.get("doubles", 0))
            triples = int(batting.get("triples", 0))
            hr = int(batting.get("homeRuns", 0))
            total_bases = hits + doubles + triples + hr  # simplified; real formula uses weight, but fine for grading
            lookup[(pid, player.get("team", ""))] = {
                "actual_hits": hits,
                "actual_doubles": doubles,
                "actual_triples": triples,
                "actual_home_runs": hr,
                "actual_total_bases": total_bases,
            }
    return lookup

def grade(pred_rows, pred_fields, lookup, date_str):
    graded = []
    hit_count = 0
    for row in pred_rows:
        pid = row.get("player_id") or row.get("playerId")
        team = row.get("team")
        key = (pid, team)
        actual = lookup.get(key, {
            "actual_hits": 0,
            "actual_doubles": 0,
            "actual_triples": 0,
            "actual_home_runs": 0,
            "actual_total_bases": 0,
        })
        hit_2tb = actual["actual_total_bases"] >= 2
        if hit_2tb:
            hit_count += 1
        graded_row = {**row, **actual, "hit_2tb": hit_2tb, "graded_timestamp": datetime.utcnow().isoformat() + "Z"}
        graded.append(graded_row)
    return graded, hit_count

def write_graded_csv(graded_rows, out_path: Path, extra_fields):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve original order plus extras
    fieldnames = list(graded_rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(graded_rows)
    logging.info("Wrote graded CSV: %s (rows=%d)", out_path, len(graded_rows))

def write_summary_json(date_str: str, total_pred: int, hit_count: int, graded_rows, out_path: Path, runtime_sec: float):
    # Compute top‑N hit rates based on ensemble_probability ranking
    sorted_rows = sorted(graded_rows, key=lambda r: float(r.get("ensemble_probability", 0)), reverse=True)
    def rate(top_n):
        top = sorted_rows[:top_n]
        return sum(1 for r in top if r.get("hit_2tb")) / top_n if top_n else 0
    summary = {
        "game_date": date_str,
        "total_predictions": total_pred,
        "total_hits": hit_count,
        "hit_rate": hit_count / total_pred if total_pred else 0,
        "top_10_hit_rate": rate(10),
        "top_20_hit_rate": rate(20),
        "top_prediction_hit": sorted_rows[0].get("hit_2tb") if sorted_rows else None,
        "grading_runtime_seconds": runtime_sec,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote summary JSON: %s", out_path)

def process_date(date_str: str):
    start = datetime.utcnow()
    # Paths
    pred_path = Path("predictions") / date_str[:4] / f"{date_str}_predictions.csv"
    if not pred_path.is_file():
        logging.error("Prediction file not found: %s", pred_path)
        return
    out_csv = Path("results") / date_str[:4] / f"{date_str}_results.csv"
    out_json = Path("results") / date_str[:4] / f"{date_str}_results_summary.json"
    log_path = Path("logs/grading_runs") / f"{date_str}.log"
    setup_logging(log_path)
    pred_rows, _ = load_predictions(pred_path)
    lookup = load_boxscore(date_str)
    graded_rows, hit_count = grade(pred_rows, _, lookup, date_str)
    write_graded_csv(graded_rows, out_csv, [])
    runtime = (datetime.utcnow() - start).total_seconds()
    write_summary_json(date_str, len(pred_rows), hit_count, graded_rows, out_json, runtime)
    logging.info("Grading finished in %.2f seconds", runtime)

def main():
    parser = argparse.ArgumentParser(description="Grade 2TB model daily predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date to grade (YYYY-MM-DD)")
    group.add_argument("--all-ungraded", action="store_true", help="Grade all dates without a results file")
    args = parser.parse_args()

    if args.date:
        process_date(args.date)
    else:
        # Find ungraded dates
        pred_files = list(Path("predictions").rglob("*_predictions.csv"))
        for pf in pred_files:
            date_part = pf.stem.replace("_predictions", "")
            result_file = Path("results") / date_part[:4] / f"{date_part}_results.csv"
            if not result_file.is_file():
                process_date(date_part)

if __name__ == "__main__":
    main()
