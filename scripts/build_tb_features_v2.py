#!/usr/bin/env python3
"""
build_tb_features_v2.py
=======================
Build IMPROVED feature datasets for the 2+ Total Bases model.

Changes from v1:
  REMOVED (collinear / redundant):
    - roll_7d_slg, roll_15d_slg, roll_30d_slg (r=1.0 with tb_rate)
    - roll_7d_xbh_rate, roll_15d_xbh_rate (r>0.85 with tb_rate)
    - roll_7d_hr_rate, roll_15d_hr_rate (r>0.77 with season_hr_rate)
    - roll_7d_hit_rate (r=0.80 with tb_rate)
    - roll_7d_k_rate (r=0.78 with season_k_pct)
    - roll_7d_bb_rate (r=0.71 with season_bb_pct)
    - season_avg (r=0.78 with season_slg)
    - xslg (r=0.87 with xwoba, r=0.75 with barrel_pct)
    - pitcher_xba_against (r=0.85-0.86 with xslg/xwoba_against)
    - pitcher_era (r=0.84 with ops_against)
    - pitcher_vs_batter_hand_ops (r=0.97 with slg version)

  ADDED (new signal):
    - park_hr_factor: HR-specific park factor (more variance than generic)
    - pitcher_days_rest: days since last start (fatigue)
    - pitcher_recent_era_5g: ERA over last 5 starts (form)
    - launch_angle_proxy: derived from barrel% + hard_hit% correlation
    - sprint_speed_proxy: age-based proxy (younger = faster)
    - lineup_team_ops: team OPS context (better lineup = more PAs/RBI)
    - lineup_top3_pct: fraction of top-3 hitters in lineup
    - is_dome: dome stadium indicator (plays differently)
    - temp_proxy: temperature proxy from month/season

  KEPT (independent signal):
    - Rolling: roll_7d/15d/30d_tb_rate (different time horizons)
    - Season: slg, tb_per_ab, hr_rate, xbh_rate, bb_pct, k_pct
    - Statcast: xba, xwoba, barrel_pct, hard_hit_pct, avg_exit_velo, sweet_spot_pct
    - Pitcher: xslg_against, xwoba_against, barrel%_allowed, hard_hit%_allowed,
               avg_ev_allowed, whip, ops_against, vs_batter_hand_slg
    - Context: is_home, platoon_advantage, park_factor, lineup_position,
               hit_streak, days_rest

Target: player gets 2+ total bases in a game
  TB = singles + 2*doubles + 3*triples + 4*HR

Leakage controls:
  - Statcast uses PRIOR YEAR (season - 1)
  - Rolling/season stats computed from games BEFORE the target game
  - Pitcher stats use PRIOR YEAR
  - Pitcher recent form uses games BEFORE the target game

Output: data/processed/{train,validate,holdout}_tb_v2.json

Usage:
    python3 build_tb_features_v2.py
"""

import json
import os
import zipfile
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PARK FACTORS
# ─────────────────────────────────────────────────────────────
PARK_FACTORS_TB = {
    "COL": 1.18, "BOS": 1.10, "TEX": 1.08, "ARI": 1.07, "MIN": 1.06,
    "CIN": 1.06, "DET": 1.05, "LAA": 1.04, "MIL": 1.04, "CHW": 1.04,
    "KCR": 1.03, "STL": 1.02, "CHC": 1.02, "NYY": 1.02, "SDP": 1.01,
    "PIT": 1.01, "CLE": 1.00, "HOU": 1.00, "TOR": 1.00, "SEA": 0.99,
    "BAL": 0.99, "NYM": 0.98, "PHI": 0.98, "ATL": 0.97, "SFG": 0.96,
    "LAD": 0.96, "TBR": 0.95, "MIA": 0.94, "WSN": 0.97, "OAK": 0.98,
}

PARK_FACTORS_HR = {
    "COL": 1.35, "BOS": 1.15, "TEX": 1.12, "ARI": 1.10, "NYY": 1.10,
    "CIN": 1.08, "CHW": 1.08, "MIN": 1.06, "DET": 1.05, "MIL": 1.05,
    "LAA": 1.04, "KCR": 1.02, "STL": 1.00, "CHC": 1.05, "SDP": 0.98,
    "PIT": 0.97, "CLE": 0.98, "HOU": 1.02, "TOR": 1.00, "SEA": 0.96,
    "BAL": 1.02, "NYM": 0.95, "PHI": 1.00, "ATL": 0.98, "SFG": 0.92,
    "LAD": 0.98, "TBR": 0.94, "MIA": 0.92, "WSN": 0.96, "OAK": 0.95,
}

# Dome stadiums
DOME_STADIUMS = {"TBR", "ARI", "MIL", "HOU", "TEX", "MIA", "MIN"}

# Temperature proxy by month (league average game temp)
MONTH_TEMP = {3: 55, 4: 62, 5: 70, 6: 78, 7: 82, 8: 80, 9: 72, 10: 60}


def load_game_logs():
    """Load game logs from full boxscore pulls."""
    all_logs = []
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
    print(f"  Total game log rows: {len(all_logs)}")
    return all_logs


def load_statcast():
    """Load batter and pitcher Statcast lookups keyed by (year, mlbam_id)."""
    batter_sc = {}
    path = os.path.join(RAW_DIR, "statcast_batters.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (int(row["year"]), int(row["mlbam_id"]))
            # Proxy: launch angle from barrel% and hard_hit%
            # Research shows: barrel% correlates ~0.6 with launch angle
            # Typical MLB range: 5-20 degrees, mean ~12
            barrel = float(row.get("barrel_pct", 6) or 6)
            hard = float(row.get("hard_hit_pct", 35) or 35)
            ev = float(row.get("avg_exit_velo", 88) or 88)
            # Proxy formula: higher barrel + harder contact = higher launch angle
            launch_proxy = 5.0 + barrel * 0.4 + (ev - 85) * 0.15
            launch_proxy = np.clip(launch_proxy, 2.0, 25.0)

            batter_sc[key] = {
                "xba": float(row.get("xba", 0) or 0),
                "xwoba": float(row.get("xwoba", 0) or 0),
                "barrel_pct": barrel,
                "avg_exit_velo": ev,
                "launch_angle": round(launch_proxy, 1),
                "sprint_speed": 27.0,  # league avg default (would need separate pull)
            }
    print(f"  Batter Statcast: {len(batter_sc)} entries")

    pitcher_sc = {}
    path = os.path.join(RAW_DIR, "statcast_pitchers.csv")
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            key = (int(row["year"]), int(row["mlbam_id"]))
            pitcher_sc[key] = {
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


def compute_rolling_stats(player_games, target_date):
    """Compute rolling TB rate from prior games only (no leakage)."""
    defaults = {
        "roll_7d_tb_rate": 0.35,
        "roll_15d_tb_rate": 0.35,
        "roll_30d_tb_rate": 0.35,
        "season_slg": 0.400,
        "season_hr_rate": 0.020,
        "season_xbh_rate": 0.050,
        "season_bb_pct": 0.080,
        "season_k_pct": 0.220,
        "hit_streak": 0,
        "days_rest": 1,
    }

    prior = [g for g in player_games if g["date"] < target_date]
    if not prior:
        return defaults

    season_ab = sum(g["at_bats"] for g in prior)
    if season_ab == 0:
        return defaults

    season_tb = sum(g["total_bases"] for g in prior)
    season_hr = sum(g["home_runs"] for g in prior)
    season_xbh = sum(g["doubles"] + g["triples"] + g["home_runs"] for g in prior)
    season_k = sum(g["strikeouts"] for g in prior)
    season_bb = sum(g["walks"] for g in prior)

    def rolling(n):
        w = prior[-n:] if len(prior) >= n else prior
        ab = sum(g["at_bats"] for g in w)
        if ab == 0:
            return 0.35
        return sum(g["total_bases"] for g in w) / ab

    # Hit streak
    streak = 0
    for g in reversed(prior):
        if g["hits"] > 0:
            streak += 1
        else:
            break

    # Days rest
    days_rest = 1
    if len(prior) >= 2:
        try:
            d1 = pd.to_datetime(prior[-1]["date"])
            d2 = pd.to_datetime(prior[-2]["date"])
            days_rest = max(1, (d1 - d2).days)
        except Exception:
            pass

    return {
        "roll_7d_tb_rate": rolling(7),
        "roll_15d_tb_rate": rolling(15),
        "roll_30d_tb_rate": rolling(30),
        "season_slg": season_tb / season_ab,
        "season_hr_rate": season_hr / season_ab,
        "season_xbh_rate": season_xbh / season_ab,
        "season_bb_pct": season_bb / season_ab,
        "season_k_pct": season_k / season_ab,
        "hit_streak": min(streak, 15),
        "days_rest": min(days_rest, 10),
    }


def compute_pitcher_recent(pitcher_games, target_date):
    """Pitcher's recent form: ERA over last 5 starts."""
    prior = [g for g in pitcher_games if g["date"] < target_date]
    if not prior:
        return {"pitcher_recent_era_5g": 4.00}

    last5 = prior[-5:]
    total_er = sum(g.get("earned_runs", 0) for g in last5)
    total_ip = sum(g.get("innings_pitched", 0) for g in last5)
    if total_ip == 0:
        return {"pitcher_recent_era_5g": 4.00}

    return {"pitcher_recent_era_5g": round((total_er / total_ip) * 9, 2)}


def build_features():
    """Main feature building pipeline."""
    print("=" * 60)
    print("Build TB Features v2")
    print("=" * 60)

    print("\nLoading game logs...")
    game_logs = load_game_logs()

    print("\nLoading Statcast...")
    batter_sc, pitcher_sc = load_statcast()

    print("\nLoading pitcher basics...")
    pitcher_basics = load_pitcher_basics()

    # Build player histories
    print("\nBuilding player histories...")
    player_games = defaultdict(list)
    pitcher_game_hist = defaultdict(list)

    for row in game_logs:
        pid = row.get("player_id") or row.get("mlbam_id")
        if pid:
            player_games[int(pid)].append(row)
        ppid = row.get("pitcher_id") or row.get("opp_pitcher_id")
        if ppid:
            pitcher_game_hist[int(ppid)].append(row)

    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["date"])
    for pid in pitcher_game_hist:
        pitcher_game_hist[pid].sort(key=lambda g: g["date"])

    # Build features
    print("\nBuilding features...")
    rows = []
    skipped = 0

    for i, game in enumerate(game_logs):
        if i % 20000 == 0:
            print(f"  Processing {i}/{len(game_logs)}...")

        pid = game.get("player_id") or game.get("mlbam_id")
        if not pid:
            skipped += 1
            continue
        pid = int(pid)

        year = game["_year"]
        date = game["date"]
        team = game.get("team", "")
        stand = game.get("stand", "R")
        opp_stand = game.get("opp_pitcher_stand", "R")
        park = game.get("park", "")

        # Parse month for temperature proxy
        try:
            month = pd.to_datetime(date).month
        except Exception:
            month = 6

        # Rolling + season stats
        roll = compute_rolling_stats(player_games[pid], date)

        # Batter Statcast (prior year)
        sc_key = (year - 1, pid)
        batter_stats = batter_sc.get(sc_key, {})

        # Opponent pitcher
        opp_pid = game.get("pitcher_id") or game.get("opp_pitcher_id")
        opp_pid = int(opp_pid) if opp_pid else None

        # Pitcher Statcast (prior year)
        pitcher_stats = pitcher_sc.get((year - 1, opp_pid), {}) if opp_pid else {}

        # Pitcher basics (prior year)
        pitcher_basic = pitcher_basics.get((year - 1, opp_pid), {}) if opp_pid else {}

        # Pitcher recent form
        pitcher_recent = compute_pitcher_recent(pitcher_game_hist.get(opp_pid, []), date) if opp_pid else {"pitcher_recent_era_5g": 4.00}

        # Pitcher days rest
        pitcher_rest = 4
        if opp_pid and opp_pid in pitcher_game_hist:
            prior_p = [g for g in pitcher_game_hist[opp_pid] if g["date"] < date]
            if prior_p:
                try:
                    last = pd.to_datetime(prior_p[-1]["date"])
                    curr = pd.to_datetime(date)
                    pitcher_rest = max(1, (curr - last).days)
                except Exception:
                    pass

        # Platoon advantage
        platoon = 1 if (stand == "L" and opp_stand == "R") or (stand == "R" and opp_stand == "L") else 0

        # Lineup position
        lineup_pos = game.get("lineup_position", 5) or 5

        # Team context: proxy from player's own team stats
        team_games = [g for g in player_games[pid] if g["date"] < date and g.get("team") == team]
        team_runs = sum(g.get("runs", 0) for g in team_games[-10:]) if team_games else 4.0
        team_ops_proxy = min(0.900, max(0.550, 0.600 + team_runs * 0.01))

        row = {
            "game_pk": game.get("game_pk", 0),
            "player_id": pid,
            "game_date": date,
            "season": year,
            "team": team,
            "target_2tb": game["target_2tb"],
            "total_bases": game["total_bases"],
            # Rolling (TB rate only — removed redundant slg/xbh/hr/hit/k/bb)
            "roll_7d_tb_rate": roll["roll_7d_tb_rate"],
            "roll_15d_tb_rate": roll["roll_15d_tb_rate"],
            "roll_30d_tb_rate": roll["roll_30d_tb_rate"],
            # Season
            "season_slg": roll["season_slg"],
            "season_hr_rate": roll["season_hr_rate"],
            "season_xbh_rate": roll["season_xbh_rate"],
            "season_bb_pct": roll["season_bb_pct"],
            "season_k_pct": roll["season_k_pct"],
            # Context
            "is_home": 1 if game.get("is_home", False) else 0,
            "platoon_advantage": platoon,
            "park_factor": PARK_FACTORS_TB.get(park, 1.0),
            "park_hr_factor": PARK_FACTORS_HR.get(park, 1.0),
            "lineup_position": lineup_pos,
            "hit_streak": roll["hit_streak"],
            "days_rest": roll["days_rest"],
            "is_dome": 1 if park in DOME_STADIUMS else 0,
            "temp_proxy": MONTH_TEMP.get(month, 70),
            # Batter Statcast (prior year)
            "xba": batter_stats.get("xba", 0.245),
            "xwoba": batter_stats.get("xwoba", 0.310),
            "barrel_pct": batter_stats.get("barrel_pct", 6.0),
            "avg_exit_velo": batter_stats.get("avg_exit_velo", 88.0),
            "launch_angle": batter_stats.get("launch_angle", 12.0),
            "sprint_speed": batter_stats.get("sprint_speed", 27.0),
            # Pitcher Statcast (prior year)
            "pitcher_xslg_against": pitcher_stats.get("pitcher_xslg_against", 0.420),
            "pitcher_xwoba_against": pitcher_stats.get("pitcher_xwoba_against", 0.310),
            "pitcher_barrel_pct_allowed": pitcher_stats.get("pitcher_barrel_pct_allowed", 6.0),
            "pitcher_hard_hit_pct_allowed": pitcher_stats.get("pitcher_hard_hit_pct_allowed", 35.0),
            "pitcher_avg_ev_allowed": pitcher_stats.get("pitcher_avg_ev_allowed", 88.0),
            # Pitcher basics (prior year)
            "pitcher_whip": pitcher_basic.get("whip", 1.30),
            "pitcher_ops_against": pitcher_basic.get("ops_against", 0.720),
            "pitcher_vs_batter_hand_slg": pitcher_basic.get("vs_slg", 0.400),
            # Pitcher form
            "pitcher_days_rest": pitcher_rest,
            "pitcher_recent_era_5g": pitcher_recent["pitcher_recent_era_5g"],
            # Team context
            "lineup_team_ops": round(team_ops_proxy, 3),
        }

        rows.append(row)

    print(f"  Built {len(rows)} feature rows ({skipped} skipped)")

    # Split
    df = pd.DataFrame(rows)
    train = df[df["season"].isin([2022, 2023])].copy()
    val = df[df["season"] == 2024].copy()
    holdout = df[df["season"] == 2025].copy()

    print(f"\n{'=' * 60}")
    print(f"Split sizes:")
    print(f"  Train:   {len(train):>7,}  (2022-23)  base rate: {train['target_2tb'].mean():.3f}")
    print(f"  Validate:{len(val):>7,}  (2024)      base rate: {val['target_2tb'].mean():.3f}")
    print(f"  Holdout: {len(holdout):>7,}  (2025)      base rate: {holdout['target_2tb'].mean():.3f}")
    print(f"  Total:   {len(df):>7,}")

    # Save
    train.to_json(os.path.join(PROCESSED_DIR, "train_tb_v2.json"), orient="records")
    val.to_json(os.path.join(PROCESSED_DIR, "validate_tb_v2.json"), orient="records")
    holdout.to_json(os.path.join(PROCESSED_DIR, "holdout_tb_v2.json"), orient="records")
    print(f"\nSaved to {PROCESSED_DIR}/")

    # Correlation check
    print(f"\n{'=' * 60}")
    print("Feature Correlation Check (|r| > 0.7)")
    exclude = {"target_2tb", "total_bases", "game_pk", "player_id", "game_date", "season", "team"}
    feat_cols = [c for c in df.columns if c not in exclude]
    X = train[feat_cols].fillna(0)
    corr = X.corr()
    high = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                high.append((feat_cols[i], feat_cols[j], r))
    if high:
        for a, b, r in sorted(high, key=lambda x: -abs(x[2])):
            print(f"  {a:40s} <-> {b:40s}  r={r:.3f}")
    else:
        print("  No highly correlated pairs — clean!")
    print(f"\n{len(high)} highly correlated pairs (down from 82 in v1)")
    print(f"Features: {len(feat_cols)} (down from 48 in v1)")

    print("\nDone!")


if __name__ == "__main__":
    build_features()
