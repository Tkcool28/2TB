#!/usr/bin/env python3
"""
tests/test_dashboard.py
=======================
Unit tests for the dashboard data-loader and aggregation helpers.

Covers:
  1. Newest prediction date selected by default
  2. Missing prediction folder handled gracefully
  3. Malformed summary JSON produces warnings, no crash
  4. Weighted overall hit-rate calculation
  5. Empty result history
  6. Partial / ungraded days don't crash aggregation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add repo root so `import dashboard.app` works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dashboard import app as dash


# ---------------------------------------------------------------------------
# Fixtures — temporary directory tree
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dirs(tmp_path):
    """Create a temporary predictions/ and results/ tree, patch the module."""
    pred_dir = tmp_path / "predictions"
    res_dir = tmp_path / "results"
    pred_dir.mkdir()
    res_dir.mkdir()

    with patch.object(dash, "PRED_DIR", pred_dir), \
         patch.object(dash, "RESULT_DIR", res_dir):
        yield {
            "pred_dir": pred_dir,
            "res_dir": res_dir,
            "repo_root": tmp_path,
        }


def _write_pred_csv(directory: Path, date_str: str, rows: list[dict]):
    """Helper: write a predictions CSV for a given date."""
    year = date_str[:4]
    compact = date_str.replace("-", "")
    d = directory / year
    d.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(d / f"{compact}_predictions.csv", index=False)


def _write_pred_summary(directory: Path, date_str: str, data: dict):
    """Helper: write a prediction summary JSON for a given date."""
    year = date_str[:4]
    compact = date_str.replace("-", "")
    d = directory / year
    d.mkdir(exist_ok=True)
    with open(d / f"{compact}_summary.json", "w") as f:
        json.dump(data, f)


def _write_result_summary(directory: Path, date_str: str, data: dict):
    """Helper: write a results summary JSON for a given date."""
    year = date_str[:4]
    compact = date_str.replace("-", "")
    d = directory / year
    d.mkdir(exist_ok=True)
    with open(d / f"{compact}_results_summary.json", "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Test 1: newest prediction date selected by default
# ---------------------------------------------------------------------------

def test_newest_date_selected_by_default(tmp_dirs):
    """When multiple dates exist, discover_prediction_dates returns sorted list.
    The dashboard sidebar defaults to the last (newest) one."""
    pred_dir = tmp_dirs["pred_dir"]
    _write_pred_csv(pred_dir, "2025-06-01", [
        {"player_id": 1, "team": "BOS", "opponent": "NYY",
         "lineup_position": 1, "is_home": 1,
         "predicted_proba_2tb": 0.40, "model_count": 3},
    ])
    _write_pred_csv(pred_dir, "2025-06-03", [
        {"player_id": 2, "team": "LAD", "opponent": "SFG",
         "lineup_position": 2, "is_home": 0,
         "predicted_proba_2tb": 0.55, "model_count": 3},
    ])
    _write_pred_csv(pred_dir, "2025-06-10", [
        {"player_id": 3, "team": "CHC", "opponent": "STL",
         "lineup_position": 3, "is_home": 1,
         "predicted_proba_2tb": 0.60, "model_count": 3},
    ])

    # Clear the st.cache_data so discovery re-runs
    dash.load_todays_predictions.clear()

    dates = dash.discover_prediction_dates()
    assert dates == ["2025-06-01", "2025-06-03", "2025-06-10"]

    # The dashboard sidebar index should be len-1 (newest)
    # We simulate sidebar selection logic:
    selected = dates[-1] if dates else dash._today_denver()
    assert selected == "2025-06-10"


# ---------------------------------------------------------------------------
# Test 2: missing prediction folder handled gracefully
# ---------------------------------------------------------------------------

def test_missing_prediction_folder(tmp_dirs):
    """If predictions/ does not exist (or is empty), loaders return empty + warn."""
    pred_dir = tmp_dirs["pred_dir"]
    # Remove the predictions directory entirely
    import shutil
    shutil.rmtree(pred_dir)

    dash.load_todays_predictions.clear()
    dash.load_todays_summary.clear()

    # Recreate as empty to avoid Path.exists errors on the patch target
    # Actually the module references the patched path — let's just make a new empty dir
    pred_dir.mkdir()

    df, warns = dash.load_todays_predictions("2025-06-01")
    assert df.empty
    assert isinstance(warns, list)

    summary, warns2 = dash.load_todays_summary("2025-06-01")
    assert summary == {}
    assert isinstance(warns2, list)


# ---------------------------------------------------------------------------
# Test 3: malformed summary JSON produces warnings, no crash
# ---------------------------------------------------------------------------

def test_malformed_json_returns_warning(tmp_dirs):
    """A corrupt _summary.json should not crash — should return empty + warning."""
    pred_dir = tmp_dirs["pred_dir"]
    year = "2025"
    compact = "20250601"
    d = pred_dir / year
    d.mkdir(parents=True, exist_ok=True)
    # Write corrupt JSON
    with open(d / f"{compact}_summary.json", "w") as f:
        f.write("{not valid json!!!")

    dash.load_todays_summary.clear()

    summary, warns = dash.load_todays_summary("2025-06-01")
    assert summary == {}
    assert len(warns) >= 1
    # Warning message mentions the file path
    assert any("20250601" in str(w) for w in warns)


def test_malformed_result_summary_json(tmp_dirs):
    """A corrupt _results_summary.json should be skipped with a warning."""
    res_dir = tmp_dirs["res_dir"]
    year = "2025"
    compact = "20250601"
    d = res_dir / year
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{compact}_results_summary.json", "w") as f:
        f.write("{bad json")

    dash.load_all_result_summaries.clear()

    df, warns = dash.load_all_result_summaries()
    assert df.empty
    assert len(warns) >= 1
    assert any("20250601" in str(w) for w in warns)


# ---------------------------------------------------------------------------
# Test 4: weighted overall hit-rate calculation
# ---------------------------------------------------------------------------

def test_weighted_hit_rate(tmp_dirs):
    """Overall hit rate must be sum(hits)/sum(graded), not mean(daily_rate)."""
    res_dir = tmp_dirs["res_dir"]

    # Day 1: 10 graded, 5 hits → 50%
    _write_result_summary(res_dir, "2025-06-01", {
        "game_date": "2025-06-01", "total_predictions": 10,
        "graded_predictions": 10, "total_hits_2tb": 5,
        "hit_rate": 0.5, "top_10_hit_rate": 0.3, "top_20_hit_rate": 0.35,
    })
    # Day 2: 100 graded, 10 hits → 10%
    _write_result_summary(res_dir, "2025-06-02", {
        "game_date": "2025-06-02", "total_predictions": 100,
        "graded_predictions": 100, "total_hits_2tb": 10,
        "hit_rate": 0.1, "top_10_hit_rate": 0.15, "top_20_hit_rate": 0.20,
    })

    dash.load_all_result_summaries.clear()
    results_df, warns = dash.load_all_result_summaries()

    assert warns == []
    assert not results_df.empty

    row = dash._weighted_agg(results_df)
    # Weighted: (5+10)/(10+100) = 15/110 ≈ 0.1364
    expected = 15.0 / 110.0
    assert abs(row["Hit Rate"] - expected) < 1e-9

    # NOT the simple average of 0.5 and 0.1 (= 0.30)
    assert abs(row["Hit Rate"] - 0.30) > 0.01


# ---------------------------------------------------------------------------
# Test 5: empty result history
# ---------------------------------------------------------------------------

def test_empty_result_history(tmp_dirs):
    """If results/ is empty, loaders return empty DataFrame and no warnings."""
    dash.load_all_result_summaries.clear()

    df, warns = dash.load_all_result_summaries()
    assert df.empty
    assert warns == []

    row = dash._weighted_agg(df)
    assert row == {}


# ---------------------------------------------------------------------------
# Test 6: partial / ungraded days
# ---------------------------------------------------------------------------

def test_partial_ungraded_days(tmp_dirs):
    """Days with 0 graded predictions should not crash aggregation.
    Hit rate should only count days with graded > 0."""
    res_dir = tmp_dirs["res_dir"]

    # Normal day
    _write_result_summary(res_dir, "2025-06-01", {
        "game_date": "2025-06-01", "total_predictions": 50,
        "graded_predictions": 48, "ungraded_predictions": 2,
        "total_hits_2tb": 12, "hit_rate": 0.25,
        "top_10_hit_rate": 0.2, "top_20_hit_rate": 0.3,
    })
    # All ungraded (no game-log data)
    _write_result_summary(res_dir, "2025-06-02", {
        "game_date": "2025-06-02", "total_predictions": 40,
        "graded_predictions": 0, "ungraded_predictions": 40,
        "total_hits_2tb": 0, "hit_rate": 0.0,
        "top_10_hit_rate": None, "top_20_hit_rate": None,
    })
    # Another normal day
    _write_result_summary(res_dir, "2025-06-03", {
        "game_date": "2025-06-03", "total_predictions": 60,
        "graded_predictions": 58, "ungraded_predictions": 2,
        "total_hits_2tb": 20, "hit_rate": 0.3448,
        "top_10_hit_rate": 0.25, "top_20_hit_rate": 0.35,
    })

    dash.load_all_result_summaries.clear()
    results_df, warns = dash.load_all_result_summaries()

    assert warns == []
    assert len(results_df) == 3

    row = dash._weighted_agg(results_df)
    # Weighted: (12+0+20)/(48+0+58) = 32/106 ≈ 0.3019
    expected = 32.0 / 106.0
    assert abs(row["Hit Rate"] - expected) < 1e-9
    assert row["Days Tracked"] == 3


# ---------------------------------------------------------------------------
# Test 7: player_name column handled when present
# ---------------------------------------------------------------------------

def test_player_name_displayed_when_present(tmp_dirs):
    """If the CSV has a player_name column, it should be loaded correctly."""
    pred_dir = tmp_dirs["pred_dir"]
    _write_pred_csv(pred_dir, "2025-06-10", [
        {"player_id": 123, "player_name": "Mike Trout", "team": "LAA",
         "opponent": "HOU", "lineup_position": 3, "is_home": 1,
         "predicted_proba_2tb": 0.55, "model_count": 3},
    ])
    _write_pred_summary(pred_dir, "2025-06-10", {
        "date": "2025-06-10", "total_games": 5, "total_players": 50,
        "top_prediction": {"player_id": 123, "player_name": "Mike Trout",
                           "team": "LAA", "opponent": "HOU",
                           "lineup_position": 3, "predicted_proba_2tb": 0.55},
    })

    dash.load_todays_predictions.clear()
    dash.load_todays_summary.clear()

    df, _ = dash.load_todays_predictions("2025-06-10")
    assert "player_name" in df.columns
    assert df.iloc[0]["player_name"] == "Mike Trout"

    summary, _ = dash.load_todays_summary("2025-06-10")
    assert summary["top_prediction"]["player_name"] == "Mike Trout"


def test_player_name_absent_documented(tmp_dirs):
    """When player_name is absent, the info note should be shown (no crash)."""
    pred_dir = tmp_dirs["pred_dir"]
    _write_pred_csv(pred_dir, "2025-06-01", [
        {"player_id": 456, "team": "BOS", "opponent": "NYY",
         "lineup_position": 1, "is_home": 1,
         "predicted_proba_2tb": 0.40, "model_count": 3},
    ])

    dash.load_todays_predictions.clear()
    df, _ = dash.load_todays_predictions("2025-06-01")
    assert "player_name" not in df.columns
    # Dashboard should handle this gracefully (no KeyError)


# ---------------------------------------------------------------------------
# Test 8: _today_denver returns America/Denver date
# ---------------------------------------------------------------------------

def test_today_denver_uses_correct_timezone():
    """_today_denver should return the date in America/Denver, not UTC/server."""
    from zoneinfo import ZoneInfo
    denver_today = dash._today_denver()
    # It should be a valid YYYY-MM-DD string
    import datetime
    datetime.datetime.strptime(denver_today, "%Y-%m-%d")

    # Use America/Denver zoneinfo to verify it matches
    tz = ZoneInfo("America/Denver")
    expected = datetime.datetime.now(tz).strftime("%Y-%m-%d")
    assert denver_today == expected


# ---------------------------------------------------------------------------
# Test 9: import smoke test (dashboard module loads without streamlit runtime)
# ---------------------------------------------------------------------------

def test_dashboard_imports():
    """The dashboard module should be importable outside a Streamlit context."""
    # This is a basic import smoke test — if we got here, imports work.
    assert hasattr(dash, "main")
    assert hasattr(dash, "load_todays_predictions")
    assert hasattr(dash, "load_todays_summary")
    assert hasattr(dash, "load_all_result_summaries")
    assert hasattr(dash, "_weighted_agg")
    assert hasattr(dash, "_today_denver")
    assert hasattr(dash, "discover_prediction_dates")
