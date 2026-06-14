#!/usr/bin/env python3
"""
update_daily_full_game_logs.py
===============================

Incrementally update data/raw/full_game_logs_<YYYY>.json with actual
boxscore data for a single date, closing the live-season grading loop.

Usage:
    python scripts/update_daily_full_game_logs.py --date YYYY-MM-DD
    python scripts/update_daily_full_game_logs.py --all-ungraded

API usage per --date call:
    1 × schedule API
    N × live feed API (one per completed game, N ≈ 15 max for a full slate)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs" / "actuals_runs"

REQUEST_DELAY = 1.2  # seconds between MLB API calls

# Keys used for deduplication / upsert
ROW_KEY_FIELDS = ("date", "game_pk", "player_id")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _url_fetch(url: str) -> dict:
    """Fetch a URL and return parsed JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    return json.loads(resp.read())


def fetch_schedule(date_str: str) -> list[dict]:
    """
    Fetch the MLB schedule for a given date.

    Returns a list of game dicts with:
        game_pk, status, home_team, away_team, date, year
    Only games with statusCode 'F' (Final) or 'FR' (Final, rain)
    are included — games still in progress are skipped.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    data = _url_fetch(url)
    year = int(date_str[:4])
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("statusCode", "")
            # 'F' = Final, 'FR' = Final (rain shortened), 'O' = Official
            if status not in ("F", "FR", "O"):
                continue
            games.append({
                "game_pk": game["gamePk"],
                "status": status,
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
                "date": date_entry.get("date", date_str),
                "year": year,
            })
    return games


def fetch_game_players(game_pk: int, date_str: str, home_team: str, away_team: str, year: int) -> list[dict]:
    """
    Fetch the live feed for a game and extract batting rows.

    Returns a list of player dicts matching the schema used by
    pull_full_game_logs.py. Players with at_bats == 0 are skipped.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    data = _url_fetch(url)

    boxscore = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    game_data = data.get("gameData", {}).get("players", {})

    players = []
    for side, team_key in [("home", home_team), ("away", away_team)]:
        team_data = boxscore.get(side, {})
        batters = team_data.get("batters", [])
        boxscore_players = team_data.get("players", {})

        for pid in batters:
            key = f"ID{pid}"
            pdata = boxscore_players.get(key, {})
            stats = pdata.get("stats", {}).get("batting", {})
            person = pdata.get("person", {})

            # Batter hand from gameData
            gd_player = game_data.get(key, {})
            bat_side = gd_player.get("batSide", {}).get("code", "R")

            ab = int(stats.get("atBats", 0) or 0)
            if ab == 0:
                continue

            batting_order = 0
            order_raw = pdata.get("battingOrder", "0")
            try:
                batting_order = int(order_raw) // 100 if order_raw else 0
            except (ValueError, TypeError):
                batting_order = 0

            players.append({
                "date": date_str,
                "game_pk": game_pk,
                "player_id": pid,
                "player_name": person.get("fullName", ""),
                "team": team_key,
                "opponent": away_team if side == "home" else home_team,
                "home_away": "home" if side == "home" else "away",
                "bats": bat_side,
                "batting_order": batting_order,
                "at_bats": ab,
                "hits": int(stats.get("hits", 0) or 0),
                "doubles": int(stats.get("doubles", 0) or 0),
                "triples": int(stats.get("triples", 0) or 0),
                "home_runs": int(stats.get("homeRuns", 0) or 0),
                "strikeouts": int(stats.get("strikeOuts", 0) or 0),
                "walks": int(stats.get("baseOnBalls", 0) or 0),
                "year": year,
            })

    return players


# ---------------------------------------------------------------------------
# JSON file helpers (load / atomic save)
# ---------------------------------------------------------------------------

def load_game_log(year: int) -> list[dict]:
    """Load full_game_logs_<year>.json, returning [] if missing."""
    path = RAW_DIR / f"full_game_logs_{year}.json"
    if not path.is_file():
        return []
    with path.open() as f:
        return json.load(f)


def save_game_log(year: int, rows: list[dict]) -> None:
    """
    Atomically write full_game_logs_<year>.json.

    Uses a temporary file + os.replace for crash safety.
    """
    target = RAW_DIR / f"full_game_logs_{year}.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same dir so os.replace is atomic on same fs
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".tmp_full_game_logs_{year}_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f)
        os.replace(tmp_path, target)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def make_key(row: dict) -> tuple:
    """Return the upsert key for a row."""
    d, g, p = [row.get(f) for f in ROW_KEY_FIELDS]
    return (str(d), str(g), str(p))


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_rows(existing: list[dict], new_rows: list[dict]) -> tuple[list[dict], int, int]:
    """
    Insert or update rows keyed by (date, game_pk, player_id).

    Returns:
        (updated_list, rows_added, rows_updated)
    """
    index: dict[tuple, int] = {}
    for i, row in enumerate(existing):
        index[make_key(row)] = i

    added = 0
    updated = 0
    for row in new_rows:
        key = make_key(row)
        if key in index:
            existing[index[key]] = row
            updated += 1
        else:
            existing.append(row)
            index[key] = len(existing) - 1
            added += 1

    return existing, added, updated


# ---------------------------------------------------------------------------
# Date discovery (--all-ungraded)
# ---------------------------------------------------------------------------

def find_ungraded_dates() -> list[str]:
    """
    Return sorted list of dates that have predictions but no results CSV.

    Mirrors the logic in grade_predictions.find_ungraded_dates() to avoid
    coupling the two modules.
    """
    dates: list[str] = []
    for csv_path in PREDICTIONS_DIR.rglob("*_predictions.csv"):
        stem = csv_path.stem
        date_compact = stem.replace("_predictions", "")
        if len(date_compact) != 8 or not date_compact.isdigit():
            continue
        year = date_compact[:4]
        result_csv = RESULTS_DIR / year / f"{date_compact}_results.csv"
        if not result_csv.is_file():
            dates.append(f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}")
    dates.sort()
    return dates


# ---------------------------------------------------------------------------
# Core: process a single date
# ---------------------------------------------------------------------------

def process_date(date_str: str) -> dict:
    """
    Fetch actuals for a single date and upsert into the game-log file.

    Returns a stats dict:
        {
            "date": str,
            "games_found": int,
            "games_processed": int,
            "games_skipped": int,
            "rows_added": int,
            "rows_updated": int,
            "api_failures": int,
        }
    """
    year = int(date_str[:4])
    stats = {
        "date": date_str,
        "games_found": 0,
        "games_processed": 0,
        "games_skipped": 0,
        "rows_added": 0,
        "rows_updated": 0,
        "api_failures": 0,
    }

    # Step 1: schedule
    logging.info("Fetching schedule for %s", date_str)
    games = fetch_schedule(date_str)
    stats["games_found"] = len(games)
    logging.info("Found %d final game(s)", len(games))

    if not games:
        logging.info("No final games — nothing to do.")
        return stats

    # Step 2: load existing game log
    existing = load_game_log(year)
    existing_keys_before = set(make_key(r) for r in existing)
    logging.info("Loaded %d existing rows for %d", len(existing), year)

    # Step 3: fetch each game
    all_new_rows: list[dict] = []
    api_failures = 0
    games_processed = 0
    games_skipped = 0

    for game in games:
        pk = game["game_pk"]
        try:
            players = fetch_game_players(
                game_pk=pk,
                date_str=date_str,
                home_team=game["home_team"],
                away_team=game["away_team"],
                year=game["year"],
            )
            all_new_rows.extend(players)
            games_processed += 1
            logging.info("  Game %s: %d player rows", pk, len(players))
            time.sleep(REQUEST_DELAY)
        except Exception as exc:
            api_failures += 1
            logging.error("  Error fetching game %s: %s", pk, exc)
            time.sleep(2)

    stats["games_processed"] = games_processed
    stats["games_skipped"] = len(games) - games_processed
    stats["api_failures"] = api_failures

    if not all_new_rows:
        logging.info("No player rows extracted — nothing to write.")
        return stats

    # Step 4: upsert
    updated, added, upd = upsert_rows(existing, all_new_rows)
    stats["rows_added"] = added
    stats["rows_updated"] = upd

    # Step 5: atomic save
    save_game_log(year, updated)
    logging.info(
        "Saved %s: %d rows total (%d added, %d updated)",
        RAW_DIR / f"full_game_logs_{year}.json",
        len(updated),
        added,
        upd,
    )

    return stats


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(date_str: str) -> None:
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{date_str.replace('-', '')}.log"

    logging.root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update daily full game logs with live boxscore data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date to process (YYYY-MM-DD)")
    group.add_argument(
        "--all-ungraded",
        action="store_true",
        help="Process all dates that have predictions but no results yet",
    )
    args = parser.parse_args()

    if args.date:
        date_str = args.date
        setup_logging(date_str)
        logging.info("=== update_daily_full_game_logs: date=%s ===", date_str)
        stats = process_date(date_str)
        logging.info("Stats: %s", stats)
    else:
        # --all-ungraded: find dates and process each
        dates = find_ungraded_dates()
        if not dates:
            print("No ungraded dates found.")
            return

        print(f"Found {len(dates)} ungraded date(s): {', '.join(dates)}")
        for date_str in dates:
            setup_logging(date_str)
            logging.info("=== update_daily_full_game_logs: date=%s (--all-ungraded) ===", date_str)
            stats = process_date(date_str)
            logging.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
