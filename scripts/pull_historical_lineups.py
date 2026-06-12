#!/usr/bin/env python3
"""
pull_historical_lineups.py
===========================
Pull historical batting order (lineup position) for every game
in the schedule using MLB Stats API v1.1 feed/live endpoint.

The battingOrder field encodes position as:
  100 = 1st, 200 = 2nd, ..., 900 = 9th (starters)
  150, 250, etc. = substitutes

Output: data/raw/historical_lineups.json
  Key: "{date}_{game_pk}" -> {date, game_pk, home_team, away_team, home_lineup, away_lineup}
  Where home_lineup/away_lineup = {player_id_str: position_int}

Usage:
    python3 pull_historical_lineups.py
"""

import json
import os
import time
import zipfile
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "historical_lineups.json")
PROGRESS_PATH = os.path.join(RAW_DIR, "lineup_pull_progress.json")

SCHEDULE_YEARS = [2022, 2023, 2024, 2025]
REQUEST_DELAY = 1.2  # seconds between API calls


def load_schedule():
    """Load all schedule game_pks from zip files."""
    games = []
    for year in SCHEDULE_YEARS:
        zip_path = os.path.join(RAW_DIR, f"schedule_{year}.zip")
        if not os.path.exists(zip_path):
            print(f"  Missing schedule_{year}.zip, skipping")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read(zf.namelist()[0]))
        for g in data:
            games.append({
                "game_pk": g["game_pk"],
                "date": g["date"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
            })
        print(f"  Loaded {len(data)} games from {year}")
    print(f"  Total games to process: {len(games)}")
    return games


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return set(json.load(f).get("completed_pks", []))
    return set()


def save_progress(completed_pks):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"completed_pks": list(completed_pks)}, f)


def fetch_lineup(game_pk):
    """Fetch batting order for a single game. Returns (home_lineup, away_lineup) dicts."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    data = json.loads(resp.read())

    boxscore = data.get("liveData", {}).get("boxscore", {}).get("teams", {})

    result = {}
    for side in ["home", "away"]:
        batters = boxscore.get(side, {}).get("batters", [])
        players = boxscore.get(side, {}).get("players", {})

        lineup = {}
        for pid in batters:
            key = f"ID{pid}"
            pdata = players.get(key, {})
            order_raw = pdata.get("battingOrder", "0")
            try:
                order = int(order_raw)
            except (ValueError, TypeError):
                continue
            # Keep starters (100, 200, ..., 900) and subs (150, 250, etc.)
            if 100 <= order <= 999:
                pos = order // 100  # 1-9
                # Only keep first occurrence (starter) per position
                if pos not in lineup.values():
                    lineup[str(pid)] = pos

        result[side] = lineup

    return result["home"], result["away"]


def main():
    print("=" * 60)
    print("Pull Historical Lineups")
    print("=" * 60)

    games = load_schedule()
    completed = load_progress()
    print(f"  Already completed: {len(completed)} games")

    all_lineups = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            all_lineups = json.load(f)

    errors = 0
    for i, game in enumerate(games):
        pk = game["game_pk"]
        if pk in completed:
            continue

        try:
            home_lineup, away_lineup = fetch_lineup(pk)
            key = f"{game['date']}_{pk}"
            all_lineups[key] = {
                "date": game["date"],
                "game_pk": pk,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
            }
            completed.add(pk)

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(games)} games ({len(completed)} completed)")
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(all_lineups, f)
                save_progress(completed)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on game {pk}: {e}")
            time.sleep(2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_lineups, f)
    save_progress(completed)

    print(f"\nDone! {len(completed)} games processed, {errors} errors")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
