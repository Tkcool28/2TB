#!/usr/bin/env python3
"""
tests/test_grade_predictions.py
================================

Deterministic unit tests for scripts/grade_predictions.py.

No live API calls. All fixtures are created in tmp_path.
"""

import csv
import json
import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a minimal repo layout, monkey-patch module paths via env vars,
    reload the module, and return (tmp_path, gp_module)."""
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"
    logs = tmp_path / "logs" / "grading_runs"
    raw = tmp_path / "data" / "raw"
    predictions.mkdir(parents=True)
    results.mkdir(parents=True)
    logs.mkdir(parents=True)
    raw.mkdir(parents=True)

    monkeypatch.setenv("_GP_PREDICTIONS_DIR", str(predictions))
    monkeypatch.setenv("_GP_RESULTS_DIR", str(results))
    monkeypatch.setenv("_GP_LOGS_DIR", str(logs))
    monkeypatch.setenv("_GP_RAW_DIR", str(raw))

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib
    import grade_predictions as gp
    importlib.reload(gp)

    return tmp_path, gp


def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    """Write a predictions CSV with standard columns."""
    fieldnames = [
        "date", "game_pk", "team", "opponent", "player_id",
        "lineup_position", "is_home", "predicted_proba_2tb", "model_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_game_log(path: Path, rows: list[dict]) -> None:
    """Write a full_game_logs_YYYY.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f)


# ---------------------------------------------------------------------------
# Total-bases math
# ---------------------------------------------------------------------------

class TestTotalBases:
    def test_single(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(1, 0, 0, 0) == 1

    def test_double(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(1, 1, 0, 0) == 2

    def test_triple(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(1, 0, 1, 0) == 3

    def test_home_run(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(1, 0, 0, 1) == 4

    def test_single_plus_double(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(2, 1, 0, 0) == 3

    def test_all_zeros(self, repo):
        _, gp = repo
        assert gp.compute_total_bases(0, 0, 0, 0) == 0

    def test_complex_line(self, repo):
        _, gp = repo
        # 4 hits: 1 single, 1 double, 1 triple, 1 HR → TB = 10
        assert gp.compute_total_bases(4, 1, 1, 1) == 10


# ---------------------------------------------------------------------------
# Player-ID normalization
# ---------------------------------------------------------------------------

class TestNormalizePlayerId:
    def test_integer(self, repo):
        _, gp = repo
        assert gp.normalize_player_id(592450) == 592450

    def test_numeric_string(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("592450") == 592450

    def test_id_prefix(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("ID592450") == 592450

    def test_id_prefix_lowercase(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("id592450") == 592450

    def test_float_string(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("592450.0") == 592450

    def test_none(self, repo):
        _, gp = repo
        assert gp.normalize_player_id(None) is None

    def test_empty_string(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("") is None

    def test_garbage(self, repo):
        _, gp = repo
        assert gp.normalize_player_id("abc") is None


# ---------------------------------------------------------------------------
# Missing source data → never becomes losses
# ---------------------------------------------------------------------------

class TestMissingSourceData:
    def test_missing_source_aborts_no_result_files(self, repo):
        """When actuals source is completely missing, grader aborts with
        SystemExit and creates NO result files — date stays retryable."""
        tmp_path, gp = repo
        pred_rows = [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ]
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, pred_rows)
        # No game-log file at all

        with pytest.raises(SystemExit):
            gp.grade_date("2025-06-01")

        # No results files should exist
        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        summary_json = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        assert not result_csv.is_file(), "results CSV should NOT be created"
        assert not summary_json.is_file(), "summary JSON should NOT be created"

        # Date should still appear as ungraded (eligible for retry)
        assert "2025-06-01" in gp.find_ungraded_dates()

    def test_empty_actuals_source_aborts_no_result_files(self, repo):
        """When game-log file exists but has zero rows for the date,
        grader aborts with SystemExit and creates NO result files."""
        tmp_path, gp = repo
        pred_rows = [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ]
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, pred_rows)

        # Game log exists but has no rows for 2025-06-01
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-07-15", "game_pk": 200, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 2, "doubles": 1, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        with pytest.raises(SystemExit):
            gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        summary_json = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        assert not result_csv.is_file(), "results CSV should NOT be created"
        assert not summary_json.is_file(), "summary JSON should NOT be created"

        # Date should still be retryable
        assert "2025-06-01" in gp.find_ungraded_dates()


# ---------------------------------------------------------------------------
# Unmatched player handling
# ---------------------------------------------------------------------------

class TestUnmatchedPlayer:
    def test_unmatched_marked_ungraded(self, repo):
        tmp_path, gp = repo
        pred_rows = [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ]
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, pred_rows)

        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 999999,
                "player_name": "Other Player", "team": "NYY", "opponent": "BOS",
                "home_away": "away", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 1, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        with result_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["grading_status"] == "ungraded"
        assert rows[0]["unmatched_reason"] != ""

    def test_unmatched_excluded_from_hit_rate(self, repo):
        tmp_path, gp = repo
        pred_rows = [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "999999", "lineup_position": "2",
                "is_home": "1", "predicted_proba_2tb": "0.60", "model_count": "3",
            },
        ]
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, pred_rows)

        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Real Player", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 1, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        summary_path = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        with summary_path.open() as f:
            summary = json.load(f)

        assert summary["total_predictions"] == 2
        assert summary["graded_predictions"] == 1
        assert summary["ungraded_predictions"] == 1
        assert summary["total_hits_2tb"] == 0
        assert summary["hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# Output creation
# ---------------------------------------------------------------------------

class TestOutputCreation:
    def test_results_csv_created(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 2, "doubles": 1, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        assert result_csv.is_file()

        with result_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for col in [
            "actual_total_bases", "actual_hits", "actual_doubles",
            "actual_triples", "actual_home_runs", "hit_2tb",
            "grading_status", "unmatched_reason", "graded_timestamp",
        ]:
            assert col in reader.fieldnames, f"Missing column: {col}"

        assert "predicted_proba_2tb" in reader.fieldnames
        assert "player_id" in reader.fieldnames

    def test_summary_json_created(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 2, "doubles": 1, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        summary_path = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        assert summary_path.is_file()

        with summary_path.open() as f:
            summary = json.load(f)

        for key in [
            "game_date", "total_predictions", "graded_predictions",
            "ungraded_predictions", "total_hits_2tb", "hit_rate",
            "top_10_hit_rate", "top_20_hit_rate", "top_prediction_hit",
            "unmatched_examples", "grading_runtime_seconds",
        ]:
            assert key in summary, f"Missing summary key: {key}"

    def test_log_file_created(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        log_path = tmp_path / "logs" / "grading_runs" / "20250601.log"
        assert log_path.is_file()


# ---------------------------------------------------------------------------
# Correct grading with matched data
# ---------------------------------------------------------------------------

class TestGradingCorrectness:
    def test_hit_2tb_true(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        # 1 double = 2 TB → hit
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 1, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        with result_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["hit_2tb"] == "True"
        assert rows[0]["grading_status"] == "graded"
        assert int(rows[0]["actual_total_bases"]) == 2

    def test_hit_2tb_false(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 1, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        with result_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["hit_2tb"] == "False"
        assert rows[0]["grading_status"] == "graded"

    def test_home_run_is_4_tb(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
                "home_runs": 1, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        with result_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        assert int(rows[0]["actual_total_bases"]) == 4
        assert rows[0]["hit_2tb"] == "True"


# ---------------------------------------------------------------------------
# Ranking / top-N metrics
# ---------------------------------------------------------------------------

class TestRankingMetrics:
    def test_uses_predicted_proba_2tb_for_ranking(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        rows = []
        for pid, proba in [("111", "0.3"), ("222", "0.9"), ("333", "0.6")]:
            rows.append({
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": pid, "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": proba, "model_count": "3",
            })
        write_predictions_csv(pred_csv, rows)

        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        game_rows = []
        for pid in [111, 222, 333]:
            game_rows.append({
                "date": "2025-06-01", "game_pk": 100, "player_id": pid,
                "player_name": f"P{pid}", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 0, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            })
        write_game_log(game_log, game_rows)

        gp.grade_date("2025-06-01")

        summary_path = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        with summary_path.open() as f:
            summary = json.load(f)

        assert summary["graded_predictions"] == 3
        assert summary["total_hits_2tb"] == 0
        assert summary["hit_rate"] == 0.0
        assert summary["top_prediction_hit"] is False

    def test_top_n_denominator_uses_actual_graded_count(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        rows = []
        for i in range(5):
            pid = str(100 + i)
            rows.append({
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": pid, "lineup_position": str(i + 1),
                "is_home": "1", "predicted_proba_2tb": str(0.9 - i * 0.1), "model_count": "3",
            })
        write_predictions_csv(pred_csv, rows)

        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        game_rows = []
        for i in range(5):
            pid = 100 + i
            game_rows.append({
                "date": "2025-06-01", "game_pk": 100, "player_id": pid,
                "player_name": f"P{pid}", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4,
                "hits": 1 if i < 2 else 0,
                "doubles": 1 if i < 2 else 0,
                "triples": 0, "home_runs": 0,
                "strikeouts": 0, "walks": 0, "year": 2025,
            })
        write_game_log(game_log, game_rows)

        gp.grade_date("2025-06-01")

        summary_path = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        with summary_path.open() as f:
            summary = json.load(f)

        assert summary["graded_predictions"] == 5
        assert summary["total_hits_2tb"] == 2
        assert summary["hit_rate"] == 0.4
        # Only 5 graded, so top-10 and top-20 both use denominator 5
        assert summary["top_10_hit_rate"] == 0.4
        assert summary["top_20_hit_rate"] == 0.4


# ---------------------------------------------------------------------------
# --all-ungraded
# ---------------------------------------------------------------------------

class TestAllUngraded:
    def test_only_grades_dates_with_predictions_and_no_results(self, repo):
        tmp_path, gp = repo
        for date_str, compact in [("2025-06-01", "20250601"), ("2025-06-02", "20250602")]:
            pred_csv = tmp_path / "predictions" / "2025" / f"{compact}_predictions.csv"
            write_predictions_csv(pred_csv, [
                {
                    "date": date_str, "game_pk": "100", "team": "BOS",
                    "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                    "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
                },
            ])

        existing_results = tmp_path / "results" / "2025" / "20250601_results.csv"
        existing_results.parent.mkdir(parents=True, exist_ok=True)
        existing_results.touch()

        ungraded = gp.find_ungraded_dates()
        assert ungraded == ["2025-06-02"]

    def test_no_false_completed_results_when_source_missing(self, repo):
        """When source data is missing, grade_date aborts — no result files
        created, date stays in ungraded list for future retry."""
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        write_predictions_csv(pred_csv, [
            {
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "predicted_proba_2tb": "0.75", "model_count": "3",
            },
        ])
        # No game log file

        ungraded = gp.find_ungraded_dates()
        assert "2025-06-01" in ungraded

        # grade_date should abort — no result files created
        with pytest.raises(SystemExit):
            gp.grade_date("2025-06-01")

        result_csv = tmp_path / "results" / "2025" / "20250601_results.csv"
        summary_json = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        assert not result_csv.is_file()
        assert not summary_json.is_file()

        # Date must still be eligible for retry
        ungraded_after = gp.find_ungraded_dates()
        assert "2025-06-01" in ungraded_after


# ---------------------------------------------------------------------------
# Fallback probability field
# ---------------------------------------------------------------------------

class TestFallbackProba:
    def test_ensemble_probability_fallback(self, repo):
        tmp_path, gp = repo
        pred_csv = tmp_path / "predictions" / "2025" / "20250601_predictions.csv"
        fieldnames = [
            "date", "game_pk", "team", "opponent", "player_id",
            "lineup_position", "is_home", "ensemble_probability", "model_count",
        ]
        pred_csv.parent.mkdir(parents=True, exist_ok=True)
        with pred_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "date": "2025-06-01", "game_pk": "100", "team": "BOS",
                "opponent": "NYY", "player_id": "592450", "lineup_position": "1",
                "is_home": "1", "ensemble_probability": "0.75", "model_count": "3",
            })

        game_log = tmp_path / "data" / "raw" / "full_game_logs_2025.json"
        write_game_log(game_log, [
            {
                "date": "2025-06-01", "game_pk": 100, "player_id": 592450,
                "player_name": "Test", "team": "BOS", "opponent": "NYY",
                "home_away": "home", "bats": "R", "batting_order": 1,
                "at_bats": 4, "hits": 0, "doubles": 0, "triples": 0,
                "home_runs": 0, "strikeouts": 0, "walks": 0, "year": 2025,
            },
        ])

        gp.grade_date("2025-06-01")

        summary_path = tmp_path / "results" / "2025" / "20250601_results_summary.json"
        with summary_path.open() as f:
            summary = json.load(f)

        assert summary["graded_predictions"] == 1
