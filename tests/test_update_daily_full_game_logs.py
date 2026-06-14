#!/usr/bin/env python3
"""
tests/test_update_daily_full_game_logs.py
==========================================

Deterministic unit tests for scripts/update_daily_full_game_logs.py.

No live API calls. All HTTP interactions are monkey-patched.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers to build fake API responses
# ---------------------------------------------------------------------------

def make_schedule_response(games: list[dict]) -> dict:
    """Build a fake schedule API response."""
    dates = []
    for g in games:
        dates.append({
            "date": g["date"],
            "games": [{
                "gamePk": g["game_pk"],
                "status": {"statusCode": g.get("status", "F")},
                "teams": {
                    "home": {"team": {"name": g["home_team"]}},
                    "away": {"team": {"name": g["away_team"]}},
                },
            }],
        })
    # Flatten — in real API, all games for a date go in one dates entry
    # but for simplicity we merge by date
    return {"dates": dates}


def make_live_feed(game_pk: int, players: list[dict]) -> dict:
    """Build a fake liveData feed response.

    players: list of dicts with keys:
        player_id, player_name, side (home/away), team, opponent,
        bats, batting_order, at_bats, hits, doubles, triples, home_runs, strikeouts, walks
    """
    boxscore_players_home = {}
    boxscore_players_away = {}
    batters_home = []
    batters_away = []

    for p in players:
        key = f"ID{p['player_id']}"
        side = p.get("side", "home")
        boxscore_player = {
            "person": {"fullName": p["player_name"]},
            "stats": {
                "batting": {
                    "atBats": str(p["at_bats"]),
                    "hits": str(p["hits"]),
                    "doubles": str(p["doubles"]),
                    "triples": str(p["triples"]),
                    "homeRuns": str(p["home_runs"]),
                    "strikeOuts": str(p["strikeouts"]),
                    "baseOnBalls": str(p["walks"]),
                }
            },
            "battingOrder": str(p.get("batting_order", 1) * 100),
        }
        gd_player = {"batSide": {"code": p.get("bats", "R")}}

        if side == "home":
            boxscore_players_home[key] = boxscore_player
            batters_home.append(p["player_id"])
        else:
            boxscore_players_away[key] = boxscore_player
            batters_away.append(p["player_id"])

    return {
        "gameData": {
            "players": {
                **{f"ID{p['player_id']}": {"batSide": {"code": p.get("bats", "R")}}
                   for p in players}
            }
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "home": {
                        "players": boxscore_players_home,
                        "batters": batters_home,
                    },
                    "away": {
                        "players": boxscore_players_away,
                        "batters": batters_away,
                    },
                }
            }
        },
    }


class FakeResponse:
    """Mimics a urllib.response object."""
    def __init__(self, data: dict):
        self._bytes = json.dumps(data).encode()

    def read(self):
        return self._bytes


def fake_urlopen_factory(schedule_data: dict, live_feeds: dict):
    """
    Return a fake urlopen that dispatches based on the request URL.

    live_feeds: {game_pk: live_feed_dict}
    """
    def _fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "schedule" in url:
            return FakeResponse(schedule_data)
        for pk, feed_data in live_feeds.items():
            if f"/game/{pk}/" in url:
                return FakeResponse(feed_data)
        raise ConnectionError(f"Unexpected URL: {url}")

    return _fake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_setup(tmp_path, monkeypatch):
    """Create repo layout, monkey-patch module paths, reload module."""
    raw = tmp_path / "data" / "raw"
    predictions = tmp_path / "predictions"
    results = tmp_path / "results"
    logs = tmp_path / "logs" / "actuals_runs"
    raw.mkdir(parents=True)
    predictions.mkdir(parents=True)
    results.mkdir(parents=True)
    logs.mkdir(parents=True)

    # Prevent sleeps during tests
    monkeypatch.setattr("time.sleep", lambda s: None)

    import importlib
    import update_daily_full_game_logs as m
    importlib.reload(m)

    # Now monkey-patch the module's globals after import
    monkeypatch.setattr(m, "RAW_DIR", raw)
    monkeypatch.setattr(m, "PREDICTIONS_DIR", predictions)
    monkeypatch.setattr(m, "RESULTS_DIR", results)
    monkeypatch.setattr(m, "LOGS_DIR", logs)

    return {
        "raw": raw,
        "predictions": predictions,
        "results": results,
        "logs": logs,
        "mod": m,
    }


# ---------------------------------------------------------------------------
# Tests: missing JSON creates new file
# ---------------------------------------------------------------------------

class TestCreatesNewFile:
    def test_missing_json_creates_new(self, env_setup, monkeypatch):
        """When full_game_logs_<year>.json doesn't exist, create it."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 592450, "player_name": "Test Player",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 2,
             "doubles": 1, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 1},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["games_found"] == 1
        assert stats["games_processed"] == 1
        assert stats["rows_added"] == 1
        assert stats["rows_updated"] == 0

        out_path = env_setup["raw"] / "full_game_logs_2026.json"
        assert out_path.is_file()

        with out_path.open() as f:
            rows = json.load(f)
        assert len(rows) == 1
        assert rows[0]["player_id"] == 592450
        assert rows[0]["hits"] == 2

    def test_at_bats_zero_skipped(self, env_setup, monkeypatch):
        """Players with at_bats == 0 must be skipped like pull_full_game_logs."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 111, "player_name": "No AB",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 0, "hits": 0,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 0},
            {"player_id": 222, "player_name": "Has AB",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "L", "batting_order": 2, "at_bats": 3, "hits": 1,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 1, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["rows_added"] == 1

        out_path = env_setup["raw"] / "full_game_logs_2026.json"
        with out_path.open() as f:
            rows = json.load(f)
        assert len(rows) == 1
        assert rows[0]["player_id"] == 222


# ---------------------------------------------------------------------------
# Tests: existing JSON updates correctly
# ---------------------------------------------------------------------------

class TestUpdatesExisting:
    def test_upsert_adds_new_rows(self, env_setup, monkeypatch):
        """New player rows are appended when they don't exist."""
        mod = env_setup["mod"]

        # Pre-populate with an existing row for a different player
        existing = [{
            "date": "2026-06-13", "game_pk": 100, "player_id": 999,
            "player_name": "Old", "team": "Yankees", "opponent": "Red Sox",
            "home_away": "home", "bats": "R", "batting_order": 1,
            "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
            "home_runs": 0, "strikeouts": 1, "walks": 0, "year": 2026,
        }]
        env_setup["raw"].mkdir(parents=True, exist_ok=True)
        with (env_setup["raw"] / "full_game_logs_2026.json").open("w") as f:
            json.dump(existing, f)

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 592450, "player_name": "New",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 2,
             "doubles": 1, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["rows_added"] == 1
        assert stats["rows_updated"] == 0

        with (env_setup["raw"] / "full_game_logs_2026.json").open() as f:
            rows = json.load(f)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Tests: duplicate / upsert prevention
# ---------------------------------------------------------------------------

class TestDuplicatePrevention:
    def test_same_key_updates_not_duplicates(self, env_setup, monkeypatch):
        """Re-fetching the same (date, game_pk, player_id) updates in place."""
        mod = env_setup["mod"]

        # Same player already exists
        existing = [{
            "date": "2026-06-13", "game_pk": 100, "player_id": 592450,
            "player_name": "Old Name", "team": "Yankees", "opponent": "Red Sox",
            "home_away": "home", "bats": "R", "batting_order": 1,
            "at_bats": 4, "hits": 1, "doubles": 0, "triples": 0,
            "home_runs": 0, "strikeouts": 2, "walks": 0, "year": 2026,
        }]
        with (env_setup["raw"] / "full_game_logs_2026.json").open("w") as f:
            json.dump(existing, f)

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 592450, "player_name": "Updated Name",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "L", "batting_order": 3, "at_bats": 5, "hits": 3,
             "doubles": 2, "triples": 0, "home_runs": 1,
             "strikeouts": 0, "walks": 1},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["rows_added"] == 0
        assert stats["rows_updated"] == 1

        with (env_setup["raw"] / "full_game_logs_2026.json").open() as f:
            rows = json.load(f)
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Updated Name"
        assert rows[0]["hits"] == 3

    def test_idempotent_run_no_duplicates(self, env_setup, monkeypatch):
        """Running the same date twice must not create duplicates."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 592450, "player_name": "Test",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 1,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 1, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            mod.process_date("2026-06-13")
            stats2 = mod.process_date("2026-06-13")

        assert stats2["rows_added"] == 0
        assert stats2["rows_updated"] == 1

        with (env_setup["raw"] / "full_game_logs_2026.json").open() as f:
            rows = json.load(f)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Tests: non-final games skipped
# ---------------------------------------------------------------------------

class TestNonFinalGamesSkipped:
    def test_in_progress_games_skipped(self, env_setup, monkeypatch):
        """Games with statusCode != F/FR/O are skipped entirely."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([
            {
                "date": "2026-06-13", "game_pk": 100,
                "home_team": "Yankees", "away_team": "Red Sox",
                "status": "P",  # Preview / in-progress
            },
            {
                "date": "2026-06-13", "game_pk": 200,
                "home_team": "Dodgers", "away_team": "Giants",
                "status": "F",  # Final
            },
        ])
        # When status is P, exclude from our schedule; only game 200 is final
        # But our make_schedule_response puts them in separate date entries.
        # Override with a custom response.
        custom_schedule = {
            "dates": [{
                "date": "2026-06-13",
                "games": [
                    {
                        "gamePk": 100,
                        "status": {"statusCode": "P"},
                        "teams": {
                            "home": {"team": {"name": "Yankees"}},
                            "away": {"team": {"name": "Red Sox"}},
                        },
                    },
                    {
                        "gamePk": 200,
                        "status": {"statusCode": "F"},
                        "teams": {
                            "home": {"team": {"name": "Dodgers"}},
                            "away": {"team": {"name": "Giants"}},
                        },
                    },
                ],
            }],
        }

        live_200 = make_live_feed(200, [
            {"player_id": 111, "player_name": "Player A",
             "side": "home", "team": "Dodgers", "opponent": "Giants",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 1,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(custom_schedule, {200: live_200})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["games_found"] == 1  # only final game
        assert stats["games_processed"] == 1
        assert stats["rows_added"] == 1

    def test_no_final_games_returns_early(self, env_setup, monkeypatch):
        """If all games are not-final, nothing is written."""
        mod = env_setup["mod"]

        custom_schedule = {
            "dates": [{
                "date": "2026-06-13",
                "games": [
                    {
                        "gamePk": 100,
                        "status": {"statusCode": "P"},
                        "teams": {
                            "home": {"team": {"name": "Yankees"}},
                            "away": {"team": {"name": "Red Sox"}},
                        },
                    },
                ],
            }],
        }

        fake_open = fake_urlopen_factory(custom_schedule, {})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = mod.process_date("2026-06-13")

        assert stats["games_processed"] == 0
        assert not (env_setup["raw"] / "full_game_logs_2026.json").exists()


# ---------------------------------------------------------------------------
# Tests: atomic write behavior
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_full_game_log_file_created_atomically(self, env_setup, monkeypatch):
        """The output file should be valid JSON — not a partial write."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 1, "player_name": "P1",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 1,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 0},
            {"player_id": 2, "player_name": "P2",
             "side": "away", "team": "Red Sox", "opponent": "Yankees",
             "bats": "L", "batting_order": 2, "at_bats": 3, "hits": 2,
             "doubles": 1, "triples": 0, "home_runs": 0,
             "strikeouts": 1, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            mod.process_date("2026-06-13")

        out_path = env_setup["raw"] / "full_game_logs_2026.json"
        assert out_path.is_file()

        # Must be parseable JSON (atomicity check)
        with out_path.open() as f:
            rows = json.load(f)
        assert len(rows) == 2

    def test_no_temp_file_left_behind(self, env_setup, monkeypatch):
        """After a successful write, no temp files should remain."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 100,
            "home_team": "Yankees", "away_team": "Red Sox",
            "status": "F",
        }])
        live = make_live_feed(100, [
            {"player_id": 1, "player_name": "P1",
             "side": "home", "team": "Yankees", "opponent": "Red Sox",
             "bats": "R", "batting_order": 1, "at_bats": 4, "hits": 1,
             "doubles": 0, "triples": 0, "home_runs": 0,
             "strikeouts": 0, "walks": 0},
        ])

        fake_open = fake_urlopen_factory(schedule, {100: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            mod.process_date("2026-06-13")

        tmp_files = list(env_setup["raw"].glob(".tmp_full_game_logs_*"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Tests: --all-ungraded date discovery
# ---------------------------------------------------------------------------

class TestAllUngraded:
    def test_finds_predictions_without_results(self, env_setup, monkeypatch):
        """Dates with predictions but no results are discovered."""
        mod = env_setup["mod"]

        # Create predictions for 2026-06-13
        pred_dir = env_setup["predictions"] / "2026"
        pred_dir.mkdir(parents=True)
        (pred_dir / "20260613_predictions.csv").write_text(
            "date,player_id,predicted_proba_2tb\n2026-06-13,1,0.5\n"
        )

        # Must pass through_date explicitly; default caps at yesterday
        dates = mod.find_ungraded_dates(through_date="2026-06-13")
        assert "2026-06-13" in dates

    def test_skip_dates_with_completed_results(self, env_setup, monkeypatch):
        """Dates with predictions and fully graded results are not returned."""
        mod = env_setup["mod"]

        pred_dir = env_setup["predictions"] / "2026"
        pred_dir.mkdir(parents=True)
        (pred_dir / "20260613_predictions.csv").write_text(
            "date,player_id,predicted_proba_2tb\n2026-06-13,1,0.5\n"
        )
        res_dir = env_setup["results"] / "2026"
        res_dir.mkdir(parents=True)
        (res_dir / "20260613_results.csv").write_text(
            "date,player_id,grading_status\n2026-06-13,1,graded\n"
        )

        dates = mod.find_ungraded_dates(through_date="2026-06-13")
        assert "2026-06-13" not in dates

    def test_partial_results_with_ungraded_rows_are_retryable(self, env_setup, monkeypatch):
        """Partial results stay eligible so later-final games can be recovered."""
        mod = env_setup["mod"]

        pred_dir = env_setup["predictions"] / "2026"
        pred_dir.mkdir(parents=True)
        (pred_dir / "20260613_predictions.csv").write_text(
            "date,player_id,predicted_proba_2tb\n2026-06-13,1,0.5\n2026-06-13,2,0.4\n"
        )
        res_dir = env_setup["results"] / "2026"
        res_dir.mkdir(parents=True)
        (res_dir / "20260613_results.csv").write_text(
            "date,player_id,grading_status\n2026-06-13,1,graded\n2026-06-13,2,ungraded\n"
        )

        dates = mod.find_ungraded_dates(through_date="2026-06-13")
        assert "2026-06-13" in dates

    def test_non_prediction_files_ignored(self, env_setup, monkeypatch):
        """Files not ending in _predictions.csv are ignored."""
        mod = env_setup["mod"]

        pred_dir = env_setup["predictions"] / "2026"
        pred_dir.mkdir(parents=True)
        (pred_dir / "notes.txt").write_text("not a prediction")

        dates = mod.find_ungraded_dates(through_date="2026-12-31")
        assert dates == []


# ---------------------------------------------------------------------------
# Tests: --through-date filtering (same-day grading blocker fix)
# ---------------------------------------------------------------------------

class TestThroughDateFilter:
    @staticmethod
    def _make_pred(pred_dir: Path, date_compact: str) -> None:
        """Write a minimal predictions CSV for a given YYYYMMDD."""
        d = pred_dir / date_compact[:4]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{date_compact}_predictions.csv").write_text(
            f"date,player_id,predicted_proba_2tb\n{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]},1,0.5\n"
        )

    def test_today_excluded_by_default(self, env_setup, monkeypatch):
        """Today's prediction file (no results) is excluded by default
        (through_date defaults to yesterday in America/Denver)."""
        mod = env_setup["mod"]
        pred_dir = env_setup["predictions"]

        # Create a prediction file for "today" (2026-06-13 in test env)
        # Yesterday in Denver today = 2026-06-12, so 2026-06-13 > yesterday → excluded
        self._make_pred(pred_dir, "20260613")

        dates = mod.find_ungraded_dates(through_date="2026-06-12")
        assert "2026-06-13" not in dates

    def test_future_prediction_excluded(self, env_setup, monkeypatch):
        """Future prediction files are excluded by through-date cap."""
        mod = env_setup["mod"]
        pred_dir = env_setup["predictions"]

        self._make_pred(pred_dir, "20260614")
        self._make_pred(pred_dir, "20260615")

        dates = mod.find_ungraded_dates(through_date="2026-06-13")
        assert "2026-06-14" not in dates
        assert "2026-06-15" not in dates

    def test_past_ungraded_included(self, env_setup, monkeypatch):
        """Past ungraded prediction files are still included when within cap."""
        mod = env_setup["mod"]
        pred_dir = env_setup["predictions"]

        self._make_pred(pred_dir, "20260610")
        self._make_pred(pred_dir, "20260611")
        self._make_pred(pred_dir, "20260612")

        dates = mod.find_ungraded_dates(through_date="2026-06-13")
        assert "2026-06-10" in dates
        assert "2026-06-11" in dates
        assert "2026-06-12" in dates

    def test_through_date_inclusive(self, env_setup, monkeypatch):
        """through_date is inclusive: dates <= through_date are included."""
        mod = env_setup["mod"]
        pred_dir = env_setup["predictions"]

        self._make_pred(pred_dir, "20260611")
        self._make_pred(pred_dir, "20260612")
        self._make_pred(pred_dir, "20260613")

        dates = mod.find_ungraded_dates(through_date="2026-06-12")
        assert "2026-06-11" in dates
        assert "2026-06-12" in dates
        assert "2026-06-13" not in dates

class TestRowSchema:
    def test_output_fields_match_pull_full_game_logs(self, env_setup, monkeypatch):
        """Output row fields must match the schema from pull_full_game_logs."""
        mod = env_setup["mod"]

        schedule = make_schedule_response([{
            "date": "2026-06-13", "game_pk": 777,
            "home_team": "Dodgers", "away_team": "Giants",
            "status": "F",
        }])
        live = make_live_feed(777, [
            {"player_id": 545361, "player_name": "Mike Trout",
             "side": "away", "team": "Giants", "opponent": "Dodgers",
             "bats": "R", "batting_order": 3, "at_bats": 5, "hits": 3,
             "doubles": 1, "triples": 0, "home_runs": 1,
             "strikeouts": 0, "walks": 1},
        ])

        fake_open = fake_urlopen_factory(schedule, {777: live})
        with patch("urllib.request.urlopen", side_effect=fake_open):
            mod.process_date("2026-06-13")

        out_path = env_setup["raw"] / "full_game_logs_2026.json"
        with out_path.open() as f:
            rows = json.load(f)

        expected_keys = {
            "date", "game_pk", "player_id", "player_name", "team",
            "opponent", "home_away", "bats", "batting_order",
            "at_bats", "hits", "doubles", "triples", "home_runs",
            "strikeouts", "walks", "year",
        }
        assert set(rows[0].keys()) == expected_keys

        row = rows[0]
        assert row["date"] == "2026-06-13"
        assert row["game_pk"] == 777
        assert row["player_id"] == 545361
        assert row["player_name"] == "Mike Trout"
        assert row["team"] == "Giants"
        assert row["opponent"] == "Dodgers"
        assert row["home_away"] == "away"
        assert row["bats"] == "R"
        assert row["batting_order"] == 3
        assert row["at_bats"] == 5
        assert row["hits"] == 3
        assert row["home_runs"] == 1
        assert row["year"] == 2026
