"""Focused tests for load_player_names() fallback behavior."""
import json
import os
import tempfile
import logging
from unittest.mock import patch

import pytest


def load_player_names_for_test(date_str=None, mock_raw_dir=None):
    """Import and call load_player_names with mocked RAW_DIR."""
    # Import the function with a patched RAW_DIR
    import scripts.tb_predict_live as tb_predict_live

    original_raw_dir = tb_predict_live.RAW_DIR

    if mock_raw_dir:
        tb_predict_live.RAW_DIR = mock_raw_dir

    try:
        result = tb_predict_live.load_player_names(date_str)
    finally:
        tb_predict_live.RAW_DIR = original_raw_dir

    return result


class TestLoadPlayerNamesFallback:
    """Tests for year fallback and precedence logic."""

    def test_2026_falls_back_to_2025(self, tmp_path, caplog):
        """A 2026 run falls back to 2025 lookup when 2026 file is missing.

        2026 file doesn't exist (we have 2022-2025), so should find
        players in 2025 data.
        """
        # Create mock 2025 game logs with specific players
        games_2025 = [
            {"player_id": 100, "player_name": "Player 2025 Original"},
            {"player_id": 101, "player_name": "Player 2025 Another"},
        ]
        (tmp_path / "full_game_logs_2025.json").write_text(json.dumps(games_2025))

        # No 2026 file created - testing fallback

        # Capture logs
        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2026-06-14", mock_raw_dir=str(tmp_path))

        # Should have loaded from 2025 as fallback
        assert 100 in result
        assert result[100] == "Player 2025 Original"
        assert 101 in result
        assert result[101] == "Player 2025 Another"

    def test_target_year_overrides_older_year(self, tmp_path, caplog):
        """Target-year data overrides older-year data for same player ID.

        If player_id 100 exists in both 2025 and 2024 files,
        the 2025 (target) name should take precedence.
        """
        # Create mock 2024 game logs
        games_2024 = [
            {"player_id": 100, "player_name": "Old Name 2024"},
        ]
        (tmp_path / "full_game_logs_2024.json").write_text(json.dumps(games_2024))

        # Create mock 2025 game logs with SAME player_id but different name
        games_2025 = [
            {"player_id": 100, "player_name": "New Name 2025"},
        ]
        (tmp_path / "full_game_logs_2025.json").write_text(json.dumps(games_2025))

        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2025-06-14", mock_raw_dir=str(tmp_path))

        # Target year (2025) name should be used, not 2024
        assert 100 in result
        assert result[100] == "New Name 2025", "Target year should override older year"

    def test_unknown_players_get_unknown(self, tmp_path, caplog):
        """Unknown players safely receive 'Unknown' when player_id not in lookup."""
        # Create empty game logs - no players
        (tmp_path / "full_game_logs_2025.json").write_text(json.dumps([]))

        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2025-06-14", mock_raw_dir=str(tmp_path))

        # Result should be empty dict
        assert result == {}

        # The calling code should use .get(pid, "Unknown") to handle this
        # This test verifies the function returns empty dict for unknown players
        # (the "Unknown" handling is in the caller)
        assert result.get(99999, "Unknown") == "Unknown"

    def test_missing_lookup_file_warns(self, tmp_path, caplog):
        """Log warning when lookup file is missing."""
        # Create no files at all
        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2026-06-14", mock_raw_dir=str(tmp_path))

        # Should have warnings for missing files
        assert any("Missing lookup file" in r.message for r in caplog.records)

    def test_json_load_failure_warns(self, tmp_path, caplog):
        """Log warning on JSON parse failure."""
        # Create a malformed JSON file
        (tmp_path / "full_game_logs_2025.json").write_text("{ not valid json }")

        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2025-06-14", mock_raw_dir=str(tmp_path))

        assert any("JSON load failure" in r.message for r in caplog.records)

    def test_empty_final_lookup_warns(self, tmp_path, caplog):
        """Log warning when no player names were loaded."""
        # Create file with no valid player data
        (tmp_path / "full_game_logs_2025.json").write_text(json.dumps([
            {"player_id": None, "player_name": "No ID"},
            {"player_id": 100, "player_name": None},
            {"player_id": 101, "player_name": ""},
        ]))

        caplog.set_level(logging.WARNING)

        result = load_player_names_for_test("2025-06-14", mock_raw_dir=str(tmp_path))

        assert any("Empty final lookup" in r.message for r in caplog.records)

    def test_no_date_aggregates_all_years(self, tmp_path, caplog):
        """When date_str is None, aggregates all available years (2022-2025)."""
        # Create files for multiple years with different players
        games_2022 = [
            {"player_id": 10, "player_name": "Player 2022"},
        ]
        games_2023 = [
            {"player_id": 11, "player_name": "Player 2023"},
        ]
        games_2024 = [
            {"player_id": 12, "player_name": "Player 2024"},
        ]
        games_2025 = [
            {"player_id": 13, "player_name": "Player 2025"},
        ]

        (tmp_path / "full_game_logs_2022.json").write_text(json.dumps(games_2022))
        (tmp_path / "full_game_logs_2023.json").write_text(json.dumps(games_2023))
        (tmp_path / "full_game_logs_2024.json").write_text(json.dumps(games_2024))
        (tmp_path / "full_game_logs_2025.json").write_text(json.dumps(games_2025))

        result = load_player_names_for_test(None, mock_raw_dir=str(tmp_path))

        # Should have all players from all years
        assert len(result) == 4
        assert 10 in result
        assert 11 in result
        assert 12 in result
        assert 13 in result