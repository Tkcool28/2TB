#!/usr/bin/env python3
"""
pull_full_game_logs.py
=======================
Pull full game logs (boxscores) for every game in 2022-2024.

For each game, extracts for every player who played:
  - player_id, player_name, team, opponent, home_away
  - at_bats, hits, doubles, triples, home_runs, strikeouts, walks
  - batting_order (lineup position from the boxscore)

Output: data/raw/full_game_logs_2022.json, full_game_logs_2023.json, full_game_logs_2024.json

Usage:
    python3 pull_full_game_logs.py
"""

import json
import os
import time
import zipfile
import urllib.request
from collections import defaultdict

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROGRESS_PATH = os.path.join(RAW_DIR, "full_game_log_progress.json")

REQUEST_DELAY = 1.2  # seconds between API calls


def load_schedule():
    """Load all game_pks from schedule zips."""
    games = []
    for year in [2022, 2023, 2024, 2025]:
        zip_path = os.path.join(RAW_DIR, f"schedule_{year}.zip")
        if not os.path.exists(zip_path):
            print(f"  Missing schedule_{year}.zip")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            schedule = json.loads(zf.read(zf.namelist()[0]))
        for g in schedule:
            games.append({
                "game_pk": g["game_pk"],
                "date": g["date"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "year": year,
            })
        print(f"  Loaded {len(schedule)} games from {year}")
    print(f"  Total: {len(games)} games")
    return games


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            d = json.load(f)
        return set(d.get("completed_pks", [])), d.get("results", {})
    return set(), {}


def save_progress(completed, results):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"completed_pks": list(completed), "results": results}, f)


def fetch_boxscore(game_pk, date, home_team, away_team, year):
    """Fetch boxscore for a single game. Returns list of player stat dicts."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    data = json.loads(resp.read())

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

            # Get batter hand from gameData players
            gd_player = game_data.get(key, {})
            bat_side = gd_player.get("batSide", {}).get("code", "R")

            ab = int(stats.get("atBats", 0) or 0)
            if ab == 0:
                continue

            # Get batting order
            order_raw = pdata.get("battingOrder", "0")
            try:
                batting_order = int(order_raw) // 100 if order_raw else 0
            except (ValueError, TypeError):
                batting_order = 0

            players.append({
                "date": date,
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


def main():
    print("=" * 60)
    print("Pull Full Game Logs (Boxscores)")
    print("=" * 60)

    games = load_schedule()
    completed, all_results = load_progress()
    print(f"  Already completed: {len(completed)} games")

    # Organize results by year
    by_year = defaultdict(list)
    for year in [2022, 2023, 2024, 2025]:
        by_year[year] = all_results.get(str(year), [])

    errors = 0
    for i, game in enumerate(games):
        pk = game["game_pk"]
        if pk in completed:
            continue

        try:
            players = fetch_boxscore(pk, game["date"], game["home_team"], game["away_team"], game["year"])
            by_year[game["year"]].extend(players)
            completed.add(pk)

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(games)} games ({len(completed)} completed)")
                # Save intermediate
                all_results = {str(y): by_year[y] for y in [2022, 2023, 2024, 2025]}
                save_progress(completed, all_results)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error on game {pk}: {e}")
            time.sleep(2)

    # Final save — one file per year
    for year in [2022, 2023, 2024, 2025]:
        path = os.path.join(RAW_DIR, f"full_game_logs_{year}.json")
        with open(path, "w") as f:
            json.dump(by_year[year], f)
        print(f"  Saved {path}: {len(by_year[year])} rows")

    # Save progress
    all_results = {str(y): by_year[y] for y in [2022, 2023, 2024, 2025]}
    save_progress(completed, all_results)

    total_players = sum(len(by_year[y]) for y in [2022, 2023, 2024, 2025])
    print(f"\nDone! {len(completed)} games, {total_players} player-games, {errors} errors")


if __name__ == "__main__":
    main()
