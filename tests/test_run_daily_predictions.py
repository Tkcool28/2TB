import os
import sys
import subprocess
import json
import csv
from pathlib import Path

def test_run_daily_predictions_creates_outputs(tmp_path):
    # Run from repo root so relative paths work
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    # Execute the script for a known offline‑compatible date
    result = subprocess.run(
        [sys.executable,
         str(repo_root / "scripts" / "run_daily_predictions.py"),
         "--date", "2025-06-01", "--offline"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Script exited with {result.returncode}: {result.stderr}"

    # Expected output locations
    predictions_csv = repo_root / "predictions" / "2025" / "20250601_predictions.csv"
    summary_json   = repo_root / "predictions" / "2025" / "20250601_summary.json"

    # Verify files were created
    assert predictions_csv.is_file(), f"CSV not found: {predictions_csv}"
    assert summary_json.is_file(), f"JSON not found: {summary_json}"

    # ------------------------------------------------------------------
    # CSV sanity checks
    # ------------------------------------------------------------------
    with open(predictions_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Must contain at least one prediction row
        assert rows, "CSV is empty"

        # Columns that the script actually writes
        expected_cols = {
            "date",
            "game_pk",
            "team",
            "opponent",
            "player_id",
            "lineup_position",
            "is_home",
            "predicted_proba_2tb",
            "model_count",
        }
        missing = expected_cols - set(reader.fieldnames)
        assert not missing, f"Missing columns in CSV: {missing}"

    # ------------------------------------------------------------------
    # JSON sanity checks
    # ------------------------------------------------------------------
    with open(summary_json) as f:
        data = json.load(f)

    for key in ["total_games", "total_players", "top_probability", "model_version"]:
        assert key in data, f"Missing key {key} in summary JSON"

    assert data["total_games"] > 0, "total_games should be > 0"
    assert data["total_players"] > 0, "total_players should be > 0"

    # ------------------------------------------------------------------
    # Cleanup – remove the files this test created
    # ------------------------------------------------------------------
    predictions_csv.unlink()
    summary_json.unlink()
    # Remove the year folder if it became empty
    try:
        (predictions_csv.parent).rmdir()
    except OSError:
        pass
