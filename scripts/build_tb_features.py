#!/usr/bin/env python3
"""
build_tb_features.py
====================
Build feature datasets for the 2+ Total Bases model.

Target: player gets 2+ total bases in a game
  TB = singles + 2*doubles + 3*triples + 4*HR
  singles = hits - doubles - triples - HR

Features include:
  - Rolling form (7d/15d/30d): TB rate, SLG, XBH rate, HR rate, hit rate
  - Season stats: SLG, TB/AB, HR rate, XBH rate, BB%, K%
  - Batter Statcast (prior year): xBA, xSLG, xwOBA, barrel%, hard_hit%, EV, sweet_spot%
  - Pitcher Statcast (prior year): xBA_against, xSLG_against, xwOBA_against, barrel%_allowed, etc.
  - Pitcher basics (prior year): ERA, WHIP, OPS_against, vs_LHB/RHB splits
  - Matchup: platoon advantage, park factor, home/away
  - Lineup position: from historical lineups (1-9, default 5)

Leakage controls:
  - Statcast uses PRIOR YEAR (season - 1)
  - Rolling/season stats computed from games BEFORE the target game
  - Pitcher stats use PRIOR YEAR
  - Lineups are known pre-game (no leakage)

Output: data/processed/{train,validate,holdout}_tb.json

Usage:
    python3 build_tb_features.py
"""

import json
import os
import zipfile
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Ensure output directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

FEATURES = [
    # Rolling form (7d/15d/30d)
    "roll_7d_tb_rate", "roll_7d_slg", "roll_7d_xbh_rate", "roll_7d_hr_rate",
    "roll_7d_hit_rate", "roll_7d_k_rate", "roll_7d_bb_rate",
    "roll_15d_tb_rate", "roll_15d_slg", "roll_15d_xbh_rate", "roll_15d_hr_rate",
    "roll_30d_tb_rate", "roll_30d_slg",
    # Season stats
    "season_slg", "season_tb_per_ab", "season_hr_rate", "season_xbh_rate",
    "season_bb_pct", "season_k_pct", "season_avg",
    # Context
    "is_home", "platoon_advantage", "park_factor", "lineup_position",
    "hit_streak", "days_rest",
    # Batter Statcast (prior year)
    "xba", "xslg", "xwoba",
    "barrel_pct", "hard_hit_pct", "avg_exit_velo", "sweet_spot_pct",
    # Pitcher Statcast (prior year)
    "pitcher_xba_against", "pitcher_xslg_against", "pitcher_xwoba_against",
    "pitcher_barrel_pct_allowed", "pitcher_hard_hit_pct_allowed",
    "pitcher_avg_ev_allowed",
    # Pitcher basics (prior year)
    "pitcher_era", "pitcher_whip", "pitcher_ops_against",
    "pitcher_vs_batter_hand_slg", "pitcher_vs_batter_hand_ops",
]

# League averages for defaults
LEAGUE_AVG = {
    "roll_7d_tb_rate": 0.35, "roll_7d_slg": 0.400, "roll_7d_xbh_rate": 0.050,
    "roll_7d_hr_rate": 0.020, "roll_7d_hit_rate": 0.250, "roll_7d_k_rate": 0.220,
    "roll_7d_bb_rate": 0.080,
    "roll_15d_tb_rate": 0.35, "roll_15d_slg": 0.400, "roll_15d_xbh_rate": 0.050,
    "roll_15d_hr_rate": 0.020,
    "roll_30d_tb_rate": 0.35, "roll_30d_slg": 0.400,
    "season_slg": 0.400, "season_tb_per_ab": 0.35, "season_hr_rate": 0.020,
    "season_xbh_rate": 0.050, "season_bb_pct": 0.080, "season_k_pct": 0.220,
    "season_avg": 0.250,
    "is_home": 0, "platoon_advantage": 0, "park_factor": 1.0, "lineup_position": 5,
    "hit_streak": 0, "days_rest": 1,
    "xba": 0.245, "xslg": 0.420, "xwoba": 0.310,
    "barrel_pct": 6.0, "hard_hit_pct": 35.0, "avg_exit_velo": 88.0, "sweet_spot_pct": 33.0,
    "pitcher_xba_against": 0.245, "pitcher_xslg_against": 0.420, "pitcher_xwoba_against": 0.310,
    "pitcher_barrel_pct_allowed": 6.0, "pitcher_hard_hit_pct_allowed": 35.0,
    "pitcher_avg_ev_allowed": 88.0,
    "pitcher_era": 4.00, "pitcher_whip": 1.30, "pitcher_ops_against": 0.720,
    "pitcher_vs_batter_hand_slg": 0.400, "pitcher_vs_batter_hand_ops": 0.720,
}

# Park factors for total bases (approximate, source: FanGraphs)
PARK_FACTORS = {
    "COL": 1.18,  # Coors Field
    "BOS": 1.10,  # Fenway Park
    "TEX": 1.08,  # Globe Life Field
    "ARI": 1.07,  # Chase Field
    "MIN": 1.06,  # Target Field
    "CIN": 1.06,  # Great American Ball Park
    "DET": 1.05,  # Comerica Park
    "LAA": 1.04,  # Angel Stadium
    "MIL": 1.04,  # American Family Field
    "CHW": 1.04,  # Guaranteed Rate Field
    "KCR": 1.03,  # Kauffman Stadium
    "STL": 1.02,  # Busch Stadium
    "CHC": 1.02,  # Wrigley Field
    "NYY": 1.02,  # Yankee Stadium
    "SDP": 1.01,  # Petco Park
    "PIT": 1.01,  # PNC Park
    "CLE": 1.00,  # Progressive Field
    "HOU": 1.00,  # Minute Maid Park
    "TOR": 1.00,  # Rogers Centre
    "SEA": 0.99,  # T-Mobile Park
    "BAL": 0.99,  # Camden Yards
    "NYM": 0.98,  # Citi Field
    "PHI": 0.98,  # Citizens Bank Park
    "ATL": 0.97,  # Truist Park
    "SFG": 0.96,  # Oracle Park
    "LAD": 0.96,  # Dodger Stadium
    "TBR": 0.95,  # Tropicana Field
    "MIA": 0.94,  # loanDepot park
    "WSN": 0.97,  # Nationals Park
    "OAK": 0.98,  # Oakland Coliseum
}


def load_game_logs():
    """Load game logs — combines old limited logs with new full boxscore logs."""
    all_logs = []

    # Load new full game logs (boxscores) — these have ALL players
    for year in [2022, 2023, 2024, 2025]:
        path = os.path.join(RAW_DIR, f"full_game_logs_{year}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for row in data:
                row["_year"] = year
                row["total_bases"] = (
                    (row["hits"] - row["doubles"] - row["triples"] - row["home_runs"])
                    + 2 * row["doubles"]
                    + 3 * row["triples"]
                    + 4 * row["home_runs"]
                )
                row["target_2tb"] = 1 if row["total_bases"] >= 2 else 0
            all_logs.extend(data)
            print(f"  Loaded full_game_logs_{year}.json: {len(data)} rows")

    # If no full game logs exist yet, fall back to old limited logs
    if not all_logs:
        print("  WARNING: No full game logs found — falling back to limited game logs")
        for year in [2022, 2023, 2024, 2025]:
            zip_path = os.path.join(RAW_DIR, f"game_logs_{year}.zip")
            if not os.path.exists(zip_path):
                continue
            with zipfile.ZipFile(zip_path) as zf:
                data = json.loads(zf.read(zf.namelist()[0]))
            for row in data:
                row["_year"] = year
                singles = row["hits"] - row["doubles"] - row["triples"] - row["home_runs"]
                row["total_bases"] = singles + 2 * row["doubles"] + 3 * row["triples"] + 4 * row["home_runs"]
                row["target_2tb"] = 1 if row["total_bases"] >= 2 else 0
            all_logs.extend(data)
        print(f"  Loaded {len(all_logs)} game log rows (limited)")

    print(f"  Total game log rows: {len(all_logs)}")
    return all_logs


def load_statcast():
    """Load batter and pitcher Statcast lookups keyed by (year, mlbam_id)."""
    batter_sc = {}
    path = os.path.join(RAW_DIR, "statcast_batters.csv")
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            key = (int(row["year"]), int(row["mlbam_id"]))
            batter_sc[key] = {
                "xba": float(row.get("xba", 0) or 0),
                "xslg": float(row.get("xslg", 0) or 0),
                "xwoba": float(row.get("xwoba", 0) or 0),
                "barrel_pct": float(row.get("barrel_pct", 0) or 0),
                "hard_hit_pct": float(row.get("hard_hit_pct", 0) or 0),
                "avg_exit_velo": float(row.get("avg_exit_velo", 0) or 0),
                "sweet_spot_pct": float(row.get("sweet_spot_pct", 0) or 0),
            }
    print(f"  Batter Statcast: {len(batter_sc)} entries")

    pitcher_sc = {}
    path = os.path.join(RAW_DIR, "statcast_pitchers.csv")
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            key = (int(row["year"]), int(row["mlbam_id"]))
            pitcher_sc[key] = {
                "pitcher_xba_against": float(row.get("pitcher_xba_against", 0) or 0),
                "pitcher_xslg_against": float(row.get("pitcher_xslg_against", 0) or 0),
                "pitcher_xwoba_against": float(row.get("pitcher_xwoba_against", 0) or 0),
                "pitcher_barrel_pct_allowed": float(row.get("pitcher_barrel_pct_allowed", 0) or 0),
                "pitcher_hard_hit_pct_allowed": float(row.get("pitcher_hard_hit_pct_allowed", 0) or 0),
                "pitcher_avg_ev_allowed": float(row.get("pitcher_avg_ev_allowed", 0) or 0),
            }
    print(f"  Pitcher Statcast: {len(pitcher_sc)} entries")
    return batter_sc, pitcher_sc


def load_pitcher_basics():
    """Load pitcher basic stats keyed by (year, pitcher_id)."""
    pitcher_basic = {}
    for year in [2022, 2023, 2024, 2025]:
        zip_path = os.path.join(RAW_DIR, f"pitchers_{year}.zip")
        if not os.path.exists(zip_path):
            continue
        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read(zf.namelist()[0]))
        for row in data:
            pid = row.get("pitcher_id") or row.get("mlbam_id") or row.get("id")
            if pid:
                pitcher_basic[(year, int(pid))] = row
    print(f"  Pitcher basics: {len(pitcher_basic)} entries")
    return pitcher_basic


def load_opp_pitcher_lookup(year):
    """Build (date, team) -> pitcher_id from schedule."""
    path = os.path.join(RAW_DIR, f"schedule_{year}.zip")
    if not os.path.exists(path):
        return {}
    with zipfile.ZipFile(path) as zf:
        schedule = json.loads(zf.read(zf.namelist()[0]))
    lookup = {}
    for game in schedule:
        date = game["date"]
        if game.get("home_pitcher_id"):
            lookup[(date, game["away_team"])] = game["home_pitcher_id"]
        if game.get("away_pitcher_id"):
            lookup[(date, game["home_team"])] = game["away_pitcher_id"]
    return lookup


def compute_rolling_stats(player_games, target_date):
    """Compute rolling stats from prior games only (no leakage)."""
    defaults = {k: LEAGUE_AVG[k] for k in [
        "roll_7d_tb_rate", "roll_7d_slg", "roll_7d_xbh_rate", "roll_7d_hr_rate",
        "roll_7d_hit_rate", "roll_7d_k_rate", "roll_7d_bb_rate",
        "roll_15d_tb_rate", "roll_15d_slg", "roll_15d_xbh_rate", "roll_15d_hr_rate",
        "roll_30d_tb_rate", "roll_30d_slg",
        "season_slg", "season_tb_per_ab", "season_hr_rate", "season_xbh_rate",
        "season_bb_pct", "season_k_pct", "season_avg",
        "hit_streak", "days_rest",
    ]}

    prior = [g for g in player_games if g["date"] < target_date]
    if not prior:
        return defaults

    # Season stats (all prior games this year)
    season_ab = sum(g["at_bats"] for g in prior)
    if season_ab == 0:
        return defaults

    season_hits = sum(g["hits"] for g in prior)
    season_2b = sum(g["doubles"] for g in prior)
    season_3b = sum(g["triples"] for g in prior)
    season_hr = sum(g["home_runs"] for g in prior)
    season_k = sum(g["strikeouts"] for g in prior)
    season_bb = sum(g["walks"] for g in prior)
    season_tb = sum(g["total_bases"] for g in prior)
    season_xbh = season_2b + season_3b + season_hr

    def rolling(window_days):
        window = prior[-window_days:] if len(prior) >= window_days else prior
        ab = sum(g["at_bats"] for g in window)
        if ab == 0:
            return None
        h = sum(g["hits"] for g in window)
        k = sum(g["strikeouts"] for g in window)
        bb = sum(g["walks"] for g in window)
        tb = sum(g["total_bases"] for g in window)
        xbh = sum(g["doubles"] + g["triples"] + g["home_runs"] for g in window)
        hr = sum(g["home_runs"] for g in window)
        return {
            "tb_rate": tb / ab,
            "slg": tb / ab,
            "xbh_rate": xbh / ab,
            "hr_rate": hr / ab,
            "hit_rate": h / ab,
            "k_rate": k / ab,
            "bb_rate": bb / ab,
        }

    r7 = rolling(7)
    r15 = rolling(15)
    r30 = rolling(30)

    # Hit streak
    streak = 0
    for g in reversed(prior):
        if g["hits"] > 0:
            streak += 1
        else:
            break

    # Days rest
    if len(prior) >= 2:
        try:
            last_date = pd.to_datetime(prior[-1]["date"])
            curr_date = pd.to_datetime(target_date)
            days_rest = max(1, (curr_date - last_date).days)
        except Exception:
            days_rest = 1
    else:
        days_rest = 1

    return {
        "roll_7d_tb_rate": r7["tb_rate"] if r7 else defaults["roll_7d_tb_rate"],
        "roll_7d_slg": r7["slg"] if r7 else defaults["roll_7d_slg"],
        "roll_7d_xbh_rate": r7["xbh_rate"] if r7 else defaults["roll_7d_xbh_rate"],
        "roll_7d_hr_rate": r7["hr_rate"] if r7 else defaults["roll_7d_hr_rate"],
        "roll_7d_hit_rate": r7["hit_rate"] if r7 else defaults["roll_7d_hit_rate"],
        "roll_7d_k_rate": r7["k_rate"] if r7 else defaults["roll_7d_k_rate"],
        "roll_7d_bb_rate": r7["bb_rate"] if r7 else defaults["roll_7d_bb_rate"],
        "roll_15d_tb_rate": r15["tb_rate"] if r15 else defaults["roll_15d_tb_rate"],
        "roll_15d_slg": r15["slg"] if r15 else defaults["roll_15d_slg"],
        "roll_15d_xbh_rate": r15["xbh_rate"] if r15 else defaults["roll_15d_xbh_rate"],
        "roll_15d_hr_rate": r15["hr_rate"] if r15 else defaults["roll_15d_hr_rate"],
        "roll_30d_tb_rate": r30["tb_rate"] if r30 else defaults["roll_30d_tb_rate"],
        "roll_30d_slg": r30["slg"] if r30 else defaults["roll_30d_slg"],
        "season_slg": season_tb / season_ab,
        "season_tb_per_ab": season_tb / season_ab,
        "season_hr_rate": season_hr / season_ab,
        "season_xbh_rate": season_xbh / season_ab,
        "season_bb_pct": season_bb / season_ab,
        "season_k_pct": season_k / season_ab,
        "season_avg": season_hits / season_ab,
        "hit_streak": streak,
        "days_rest": days_rest,
    }



def get_pitcher_hand(pitcher_basic, year, pitcher_id):
    """Get pitcher throwing hand."""
    key = (year, pitcher_id)
    if key in pitcher_basic:
        return pitcher_basic[key].get("hand", "R")
    return "R"


def get_pitcher_splits(pitcher_basic, year, pitcher_id, batter_hand):
    """Get pitcher stats vs batter hand."""
    key = (year, pitcher_id)
    if key not in pitcher_basic:
        return LEAGUE_AVG["pitcher_vs_batter_hand_slg"], LEAGUE_AVG["pitcher_vs_batter_hand_ops"]
    row = pitcher_basic[key]
    hand_key = "vs_lhb" if batter_hand == "L" else "vs_rhb"
    splits = row.get(hand_key, {})
    try:
        slg = float(splits.get("slg", 0.400) or 0.400)
        ops = float(splits.get("ops", 0.720) or 0.720)
    except (ValueError, TypeError):
        slg, ops = 0.400, 0.720
    return slg, ops


def main():
    print("=" * 60)
    print("Build TB Features")
    print("=" * 60)

    # Load data
    game_logs = load_game_logs()
    batter_sc, pitcher_sc = load_statcast()
    pitcher_basic = load_pitcher_basics()

    # Build opponent pitcher lookups
    opp_pitcher = {}
    for year in [2022, 2023, 2024, 2025]:
        opp_pitcher.update(load_opp_pitcher_lookup(year))
    print(f"  Opponent pitcher lookup: {len(opp_pitcher)} entries")

    # Group game logs by player
    player_games = defaultdict(list)
    for row in game_logs:
        player_games[int(row["player_id"])].append(row)

    # Sort each player's games by date
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["date"])

    # Build feature rows
    print("\nBuilding features...")
    all_rows = []
    skipped = 0

    for row in game_logs:
        pid = int(row["player_id"])
        date = row["date"]
        year = row["_year"]
        team = row["team"]
        opp = row["opponent"]
        is_home = 1 if row.get("home_away") == "home" else 0

        # Rolling stats (from prior games only)
        rolling = compute_rolling_stats(player_games[pid], date)

        # Batter Statcast (prior year)
        sc_year = year - 1
        bsc = batter_sc.get((sc_year, pid), {})

        # Opponent pitcher
        opp_pid = opp_pitcher.get((date, team))
        psc = pitcher_sc.get((sc_year, opp_pid), {}) if opp_pid else {}
        pb = pitcher_basic.get((sc_year, opp_pid), {}) if opp_pid else {}

        # Batter hand
        batter_hand = row.get("bats", "R")
        pitcher_hand = get_pitcher_hand(pitcher_basic, sc_year, opp_pid) if opp_pid else "R"
        platoon = 1 if (batter_hand == "L" and pitcher_hand == "R") or \
                       (batter_hand == "R" and pitcher_hand == "L") else 0

        # Pitcher vs batter hand splits
        vs_slg, vs_ops = get_pitcher_splits(pitcher_basic, sc_year, opp_pid, batter_hand) if opp_pid else (0.400, 0.720)

        # Park factor
        park = PARK_FACTORS.get(opp if is_home else team, 1.0)

        # Lineup position — from boxscore batting_order (1-9)
        lineup_pos = int(row.get("batting_order", 5) or 5)
        if lineup_pos < 1 or lineup_pos > 9:
            lineup_pos = 5

        # Build feature vector
        feature_row = {
            "date": date,
            "player_id": pid,
            "player_name": row["player_name"],
            "team": team,
            "opponent": opp,
            "year": year,
            "target_2tb": row["target_2tb"],
            "total_bases": row["total_bases"],
            "at_bats": row["at_bats"],
            # Rolling
            **{k: rolling.get(k, LEAGUE_AVG[k]) for k in [
                "roll_7d_tb_rate", "roll_7d_slg", "roll_7d_xbh_rate", "roll_7d_hr_rate",
                "roll_7d_hit_rate", "roll_7d_k_rate", "roll_7d_bb_rate",
                "roll_15d_tb_rate", "roll_15d_slg", "roll_15d_xbh_rate", "roll_15d_hr_rate",
                "roll_30d_tb_rate", "roll_30d_slg",
                "season_slg", "season_tb_per_ab", "season_hr_rate", "season_xbh_rate",
                "season_bb_pct", "season_k_pct", "season_avg",
                "hit_streak", "days_rest",
            ]},
            # Context
            "is_home": is_home,
            "platoon_advantage": platoon,
            "park_factor": park,
            "lineup_position": lineup_pos,
            # Batter Statcast
            "xba": bsc.get("xba", LEAGUE_AVG["xba"]),
            "xslg": bsc.get("xslg", LEAGUE_AVG["xslg"]),
            "xwoba": bsc.get("xwoba", LEAGUE_AVG["xwoba"]),
            "barrel_pct": bsc.get("barrel_pct", LEAGUE_AVG["barrel_pct"]),
            "hard_hit_pct": bsc.get("hard_hit_pct", LEAGUE_AVG["hard_hit_pct"]),
            "avg_exit_velo": bsc.get("avg_exit_velo", LEAGUE_AVG["avg_exit_velo"]),
            "sweet_spot_pct": bsc.get("sweet_spot_pct", LEAGUE_AVG["sweet_spot_pct"]),
            # Pitcher Statcast
            "pitcher_xba_against": psc.get("pitcher_xba_against", LEAGUE_AVG["pitcher_xba_against"]),
            "pitcher_xslg_against": psc.get("pitcher_xslg_against", LEAGUE_AVG["pitcher_xslg_against"]),
            "pitcher_xwoba_against": psc.get("pitcher_xwoba_against", LEAGUE_AVG["pitcher_xwoba_against"]),
            "pitcher_barrel_pct_allowed": psc.get("pitcher_barrel_pct_allowed", LEAGUE_AVG["pitcher_barrel_pct_allowed"]),
            "pitcher_hard_hit_pct_allowed": psc.get("pitcher_hard_hit_pct_allowed", LEAGUE_AVG["pitcher_hard_hit_pct_allowed"]),
            "pitcher_avg_ev_allowed": psc.get("pitcher_avg_ev_allowed", LEAGUE_AVG["pitcher_avg_ev_allowed"]),
            # Pitcher basics
            "pitcher_era": float(pb.get("era", 4.0) or 4.0) if pb else 4.0,
            "pitcher_whip": float(pb.get("whip", 1.3) or 1.3) if pb else 1.3,
            "pitcher_ops_against": float(pb.get("ops_against", 0.720) or 0.720) if pb else 0.720,
            "pitcher_vs_batter_hand_slg": vs_slg,
            "pitcher_vs_batter_hand_ops": vs_ops,
        }

        all_rows.append(feature_row)

    print(f"  Built {len(all_rows)} feature rows")

    # Split by year: 2022-23 train, 2024 val, 2025 holdout
    # For now, use 2022-23 as train, 2024 as val (no 2025 full data yet)
    train = [r for r in all_rows if r["year"] <= 2023]
    val = [r for r in all_rows if r["year"] == 2024]
    holdout = [r for r in all_rows if r["year"] == 2025]  # empty until we pull 2025

    print(f"  Train: {len(train)} rows (2022-23)")
    print(f"  Validate: {len(val)} rows (2024)")
    print(f"  Holdout: {len(holdout)} rows (2025)")

    # Target base rate
    for name, split in [("Train", train), ("Val", val), ("Holdout", holdout)]:
        if split:
            rate = sum(r["target_2tb"] for r in split) / len(split)
            print(f"  {name} 2+ TB rate: {rate:.3f}")

    # Save
    for name, split in [("train", train), ("validate", val), ("holdout", holdout)]:
        path = os.path.join(PROCESSED_DIR, f"{name}_tb.json")
        with open(path, "w") as f:
            json.dump(split, f)
        print(f"  Saved {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
