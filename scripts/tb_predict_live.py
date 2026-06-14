#!/usr/bin/env python3
"""
tb_predict_live.py
===================
Fetch today's slate and predict 2+ TB probability for each player.

Uses:
  - MLB Stats API for today's games, lineups, and probable pitchers
  - Trained v2 models (logreg_tb_v2.pkl, xgb_tb_v2.pkl, lgbm_tb_v2.pkl)
    with 34-feature schema via scaler_tb_v2.pkl
  - Rolling stats computed from recent game logs (defaults when unavailable)
  - Real hydration from local data + MLB API gamelogs when available

Output: JSON with predictions for all players in lineup spots 1-5

Usage:
    python3 scripts/tb_predict_live.py
    python3 scripts/tb_predict_live.py --date 2025-06-01 --offline --max-predictions 10
"""

import argparse
import pandas as pd
import lightgbm
import json
import os
import pickle
import time
import zipfile
import numpy as np
import pandas as pd
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# 34-feature v2 schema (exact order from train_tb_v2.json)
FEATURES = [
    "roll_7d_tb_rate",
    "roll_15d_tb_rate",
    "roll_30d_tb_rate",
    "season_slg",
    "season_hr_rate",
    "season_xbh_rate",
    "season_bb_pct",
    "season_k_pct",
    "is_home",
    "platoon_advantage",
    "park_factor",
    "park_hr_factor",
    "lineup_position",
    "hit_streak",
    "days_rest",
    "is_dome",
    "temp_proxy",
    "xba",
    "xwoba",
    "barrel_pct",
    "avg_exit_velo",
    "launch_angle",
    "sprint_speed",
    "pitcher_xslg_against",
    "pitcher_xwoba_against",
    "pitcher_barrel_pct_allowed",
    "pitcher_hard_hit_pct_allowed",
    "pitcher_avg_ev_allowed",
    "pitcher_whip",
    "pitcher_ops_against",
    "pitcher_vs_batter_hand_slg",
    "pitcher_days_rest",
    "pitcher_recent_era_5g",
    "lineup_team_ops",
]

# League-average defaults for every v2 feature
DEFAULTS = {
    "roll_7d_tb_rate": 0.35,
    "roll_15d_tb_rate": 0.35,
    "roll_30d_tb_rate": 0.35,
    "season_slg": 0.400,
    "season_hr_rate": 0.020,
    "season_xbh_rate": 0.050,
    "season_bb_pct": 0.080,
    "season_k_pct": 0.220,
    "is_home": 0,
    "platoon_advantage": 0,
    "park_factor": 1.0,
    "park_hr_factor": 1.0,
    "lineup_position": 5,
    "hit_streak": 0,
    "days_rest": 1,
    "is_dome": 0,
    "temp_proxy": 70,
    "xba": 0.245,
    "xwoba": 0.310,
    "barrel_pct": 6.0,
    "avg_exit_velo": 88.0,
    "launch_angle": 12.0,
    "sprint_speed": 27.0,
    "pitcher_xslg_against": 0.420,
    "pitcher_xwoba_against": 0.310,
    "pitcher_barrel_pct_allowed": 6.0,
    "pitcher_hard_hit_pct_allowed": 35.0,
    "pitcher_avg_ev_allowed": 88.0,
    "pitcher_whip": 1.30,
    "pitcher_ops_against": 0.720,
    "pitcher_vs_batter_hand_slg": 0.400,
    "pitcher_days_rest": 4,
    "pitcher_recent_era_5g": 4.00,
    "lineup_team_ops": 0.640,
}

# ─────────────────────────────────────────────────────────────
# PARK FACTORS (from build_tb_features_v2.py)
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


# ─────────────────────────────────────────────────────────────
# LOCAL DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_local_player_games(date_str):
    """Load local full_game_logs for the year of date_str.

    If the year file is unavailable, fall back to the latest available
    year as prior-season proxy.  Returns list of game-log rows with
    total_bases and target_2tb computed.
    """
    try:
        target_year = int(date_str[:4])
    except (ValueError, TypeError):
        target_year = datetime.now().year

    years_to_try = [target_year] + sorted(
        [y for y in [2025, 2024, 2023, 2022] if y != target_year], reverse=True
    )
    for year in years_to_try:
        path = os.path.join(RAW_DIR, f"full_game_logs_{year}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            rows = []
            for row in data:
                if row.get("date", "") >= date_str:
                    continue
                tb = (
                    (row.get("hits", 0) - row.get("doubles", 0)
                     - row.get("triples", 0) - row.get("home_runs", 0))
                    + 2 * row.get("doubles", 0)
                    + 3 * row.get("triples", 0)
                    + 4 * row.get("home_runs", 0)
                )
                row["total_bases"] = tb
                row["target_2tb"] = 1 if tb >= 2 else 0
                rows.append(row)
            return rows
        except Exception:
            continue
    return []


def load_pitcher_histories(player_games):
    """Build pitcher_id -> list of pitching game rows from local game logs.

    full_game_logs only contain batter rows, so this will typically return
    empty dicts.  Pitcher recent/form features fall back to defaults.
    Kept for forward-compatibility if pitcher game logs are added later.
    """
    pitcher_games = defaultdict(list)
    for row in player_games:
        ppid = row.get("pitcher_id") or row.get("opp_pitcher_id")
        if ppid:
            pitcher_games[int(ppid)].append(row)
    for pid in pitcher_games:
        pitcher_games[pid].sort(key=lambda g: g.get("date", ""))
    return pitcher_games


def load_team_games(player_games):
    """Build team -> list of game rows from local game logs."""
    team_games = defaultdict(list)
    for row in player_games:
        team = row.get("team", "")
        if team:
            team_games[team].append(row)
    for t in team_games:
        team_games[t].sort(key=lambda g: g.get("date", ""))
    return team_games


def load_statcast_lookups(date_str):
    """Load statcast CSVs.  Use latest year strictly < target year if available.

    Returns (batter_dict, pitcher_dict) keyed by mlbam_id.
    """
    try:
        target_year = int(date_str[:4])
    except (ValueError, TypeError):
        target_year = datetime.now().year

    batter_sc = {}
    path = os.path.join(RAW_DIR, "statcast_batters.csv")
    if os.path.exists(path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            available_years = sorted(df["year"].unique(), reverse=True)
            best_year = None
            for y in available_years:
                if y < target_year:
                    best_year = y
                    break
            if best_year is None:
                best_year = available_years[0] if available_years else target_year
            sub = df[df["year"] == best_year]
            for _, row in sub.iterrows():
                pid = int(row["mlbam_id"])
                barrel = float(row.get("barrel_pct", 6) or 6)
                ev = float(row.get("avg_exit_velo", 88) or 88)
                launch_proxy = 5.0 + barrel * 0.4 + (ev - 85) * 0.15
                launch_proxy = float(np.clip(launch_proxy, 2.0, 25.0))
                batter_sc[pid] = {
                    "xba": float(row.get("xba", 0) or 0),
                    "xwoba": float(row.get("xwoba", 0) or 0),
                    "barrel_pct": barrel,
                    "avg_exit_velo": ev,
                    "launch_angle": round(launch_proxy, 1),
                    "sprint_speed": 27.0,
                }
        except Exception:
            pass

    pitcher_sc = {}
    path = os.path.join(RAW_DIR, "statcast_pitchers.csv")
    if os.path.exists(path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            available_years = sorted(df["year"].unique(), reverse=True)
            best_year = None
            for y in available_years:
                if y < target_year:
                    best_year = y
                    break
            if best_year is None:
                best_year = available_years[0] if available_years else target_year
            sub = df[df["year"] == best_year]
            for _, row in sub.iterrows():
                pid = int(row["mlbam_id"])
                pitcher_sc[pid] = {
                    "pitcher_xslg_against": float(row.get("pitcher_xslg_against", 0) or 0),
                    "pitcher_xwoba_against": float(row.get("pitcher_xwoba_against", 0) or 0),
                    "pitcher_barrel_pct_allowed": float(row.get("pitcher_barrel_pct_allowed", 0) or 0),
                    "pitcher_hard_hit_pct_allowed": float(row.get("pitcher_hard_hit_pct_allowed", 0) or 0),
                    "pitcher_avg_ev_allowed": float(row.get("pitcher_avg_ev_allowed", 0) or 0),
                }
        except Exception:
            pass

    return batter_sc, pitcher_sc


def load_player_names(date_str=None):
    """Load player_id -> player_name lookup from full_game_logs_YYYY.json files.

    When date_str is supplied, searches target year and up to 3 prior years
    (target_year, target_year-1, target_year-2, target_year-3) as fallbacks.
    Newer-year names take precedence over older-year names (processed first,
    do not overwrite existing player IDs).

    Logs warnings for missing lookup files, JSON load failures, and empty
    final lookup.

    Returns dict keyed by player_id (int) -> player_name (str).
    """
    import logging

    player_names = {}

    # Determine which years to check
    if date_str:
        target_year = int(date_str[:4])
        years_to_try = [target_year - i for i in range(4)]  # target, -1, -2, -3
        years_to_try = [y for y in years_to_try if y >= 2022]  # don't go below available data
    else:
        years_to_try = [2025, 2024, 2023, 2022]

    for year in years_to_try:
        game_log_path = os.path.join(RAW_DIR, f"full_game_logs_{year}.json")
        if not os.path.exists(game_log_path):
            logging.warning(f"Missing lookup file: full_game_logs_{year}.json")
            continue
        try:
            with open(game_log_path) as f:
                games = json.load(f)
            for row in games:
                pid = row.get("player_id")
                name = row.get("player_name")
                if pid and name:
                    pid_int = int(pid)
                    # Only add if not already present (newer years processed first)
                    if pid_int not in player_names:
                        player_names[pid_int] = name
        except Exception as e:
            logging.warning(f"JSON load failure for full_game_logs_{year}.json: {e}")
            continue

    if not player_names:
        logging.warning("Empty final lookup: no player names loaded")

    return player_names


def load_pitcher_basics(date_str):
    """Load pitcher basics zip.  Use latest year strictly < target year if available.

    Returns dict keyed by pitcher_id.
    """
    try:
        target_year = int(date_str[:4])
    except (ValueError, TypeError):
        target_year = datetime.now().year

    years_to_try = sorted(
        [y for y in [2025, 2024, 2023, 2022] if y < target_year], reverse=True
    )
    if not years_to_try:
        years_to_try = sorted([2025, 2024, 2023, 2022], reverse=True)

    for year in years_to_try:
        zip_path = os.path.join(RAW_DIR, f"pitchers_{year}.zip")
        if not os.path.exists(zip_path):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                data = json.loads(zf.read(zf.namelist()[0]))
            result = {}
            for row in data:
                pid = row.get("pitcher_id") or row.get("mlbam_id") or row.get("id")
                if pid:
                    result[int(pid)] = row
            return result
        except Exception:
            continue
    return {}


# ─────────────────────────────────────────────────────────────
# STAT COMPUTATION (from build_tb_features_v2.py)
# ─────────────────────────────────────────────────────────────

def compute_rolling_stats(player_games_list, target_date):
    """Compute rolling stats for one player from their game list.

    player_games_list: list of game rows for a single player, sorted by date.
    Returns dict with rolling/season/hit_streak/days_rest features.
    """
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
    prior = [g for g in player_games_list if g.get("date", "") < target_date]
    if not prior:
        return defaults

    season_ab = sum(g.get("at_bats", 0) for g in prior)
    if season_ab == 0:
        return defaults

    season_tb = sum(g.get("total_bases", 0) for g in prior)
    season_hr = sum(g.get("home_runs", 0) for g in prior)
    season_xbh = sum(g.get("doubles", 0) + g.get("triples", 0) + g.get("home_runs", 0) for g in prior)
    season_k = sum(g.get("strikeouts", 0) for g in prior)
    season_bb = sum(g.get("walks", 0) for g in prior)

    def rolling(n):
        w = prior[-n:] if len(prior) >= n else prior
        ab = sum(g.get("at_bats", 0) for g in w)
        if ab == 0:
            return 0.35
        return sum(g.get("total_bases", 0) for g in w) / ab

    streak = 0
    for g in reversed(prior):
        if g.get("hits", 0) > 0:
            streak += 1
        else:
            break

    days_rest = 1
    if len(prior) >= 2:
        try:
            d1 = datetime.strptime(prior[-1]["date"], "%Y-%m-%d")
            d2 = datetime.strptime(prior[-2]["date"], "%Y-%m-%d")
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


def _parse_ip(ip_val):
    """Parse innings pitched: handles strings like '5.1' (5.1 innings = 5.333)
    and '5.2' (5.2 innings = 5.667), as well as float/int values.
    """
    if ip_val is None:
        return 0.0
    if isinstance(ip_val, (int, float)):
        return float(ip_val)
    s = str(ip_val).strip()
    if not s:
        return 0.0
    # Handle "5.1" meaning 5 + 1/3, "5.2" meaning 5 + 2/3
    if "." in s:
        parts = s.split(".")
        whole = float(parts[0])
        frac_part = parts[1]
        if frac_part == "1":
            return whole + 1.0 / 3.0
        elif frac_part == "2":
            return whole + 2.0 / 3.0
        else:
            try:
                return float(s)
            except ValueError:
                return whole
    try:
        return float(s)
    except ValueError:
        return 0.0


def compute_pitcher_recent(pitcher_games_list, target_date):
    """Pitcher's recent form: ERA over last 5 starts."""
    prior = [g for g in pitcher_games_list if g.get("date", "") < target_date]
    if not prior:
        return {"pitcher_recent_era_5g": 4.00}

    last5 = prior[-5:]
    total_er = sum(g.get("earned_runs", 0) for g in last5)
    total_ip = sum(_parse_ip(g.get("innings_pitched", 0)) for g in last5)
    if total_ip == 0:
        return {"pitcher_recent_era_5g": 4.00}

    return {"pitcher_recent_era_5g": round((total_er / total_ip) * 9, 2)}


def compute_pitcher_days_rest(pitcher_games_list, target_date):
    """Days since pitcher's last appearance before target_date, clipped 1..10."""
    prior = [g for g in pitcher_games_list if g.get("date", "") < target_date]
    if not prior:
        return {"pitcher_days_rest": 4}
    try:
        last = datetime.strptime(prior[-1]["date"], "%Y-%m-%d")
        curr = datetime.strptime(target_date, "%Y-%m-%d")
        rest = max(1, min(10, (curr - last).days))
        return {"pitcher_days_rest": rest}
    except Exception:
        return {"pitcher_days_rest": 4}


def compute_team_ops_proxy(team_games_list, target_date):
    """Last 10 team games before target_date.  OPS proxy = 0.600 + avg_runs * 0.01."""
    prior = [g for g in team_games_list if g.get("date", "") < target_date]
    if not prior:
        return {"lineup_team_ops": 0.640}
    last10 = prior[-10:]
    avg_runs = sum(g.get("runs", 0) for g in last10) / len(last10)
    ops = 0.600 + avg_runs * 0.01
    ops = min(0.900, max(0.550, ops))
    return {"lineup_team_ops": round(ops, 3)}


# ─────────────────────────────────────────────────────────────
# MLB API GAMELLOG FETCH
# ─────────────────────────────────────────────────────────────

def fetch_player_gamelogs(player_ids, season, group, cache_dir, api_delay):
    """Fetch player gamelogs from MLB API and cache locally.

    Uses endpoint:
      https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group={group}&season={season}

    Returns list of parsed game-log rows in the same format as full_game_logs.
    """
    cache_path = os.path.join(cache_dir, f"gamelogs_{season}_{group}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    all_rows = []
    for pid in player_ids:
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=gameLog&group={group}&season={season}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
        except Exception:
            time.sleep(api_delay)
            continue

        # Parse splits
        for stat_block in data.get("stats", []):
            for split in stat_block.get("splits", []):
                game = split.get("game", {})
                stat = split.get("stat", {})
                if group == "hitting":
                    row = {
                        "date": split.get("date", "")[:10],
                        "game_pk": game.get("gamePk", 0),
                        "player_id": pid,
                        "team": split.get("team", {}).get("abbreviation", ""),
                        "opponent": split.get("opponent", {}).get("abbreviation", ""),
                        "home_away": "home" if split.get("isHome", False) else "away",
                        "at_bats": int(stat.get("atBats", 0) or 0),
                        "hits": int(stat.get("hits", 0) or 0),
                        "doubles": int(stat.get("doubles", 0) or 0),
                        "triples": int(stat.get("triples", 0) or 0),
                        "home_runs": int(stat.get("homeRuns", 0) or 0),
                        "strikeouts": int(stat.get("strikeOuts", 0) or 0),
                        "walks": int(stat.get("baseOnBalls", 0) or 0),
                        "year": season,
                    }
                    tb = (
                        (row["hits"] - row["doubles"] - row["triples"] - row["home_runs"])
                        + 2 * row["doubles"]
                        + 3 * row["triples"]
                        + 4 * row["home_runs"]
                    )
                    row["total_bases"] = tb
                    row["target_2tb"] = 1 if tb >= 2 else 0
                    all_rows.append(row)
                elif group == "pitching":
                    row = {
                        "date": split.get("date", "")[:10],
                        "game_pk": game.get("gamePk", 0),
                        "player_id": pid,
                        "team": split.get("team", {}).get("abbreviation", ""),
                        "opponent": split.get("opponent", {}).get("abbreviation", ""),
                        "home_away": "home" if split.get("isHome", False) else "away",
                        "innings_pitched": _parse_ip(stat.get("inningsPitched", 0)),
                        "earned_runs": int(stat.get("earnedRuns", 0) or 0),
                        "year": season,
                    }
                    all_rows.append(row)

        time.sleep(api_delay)

    # Cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(all_rows, f)
    except Exception:
        pass

    return all_rows


def merge_gamelog_rows(base_rows, new_rows):
    """Merge by (date, player_id, game_pk) without duplicates."""
    seen = set()
    merged = []
    for row in base_rows + new_rows:
        key = (row.get("date", ""), row.get("player_id", 0), row.get("game_pk", 0))
        if key not in seen:
            seen.add(key)
            merged.append(row)
    merged.sort(key=lambda r: (r.get("date", ""), r.get("player_id", 0)))
    return merged


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_models():
    """Load v2 model files and scaler. Fail loudly if any are missing."""
    assert len(FEATURES) == 34, f"Expected 34 features, got {len(FEATURES)}"
    missing_defaults = set(FEATURES) - set(DEFAULTS.keys())
    assert not missing_defaults, f"DEFAULTS missing keys: {missing_defaults}"

    models = {}
    model_files = {
        "logreg": "logreg_tb_v2.pkl",
        "xgb": "xgb_tb_v2.pkl",
        "lgbm": "lgbm_tb_v2.pkl",
    }
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing v2 model file: {path}. "
                f"Expected v2 models (logreg_tb_v2.pkl, xgb_tb_v2.pkl, lgbm_tb_v2.pkl) "
                f"but {filename} was not found."
            )
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name} from {filename}")

    scaler_path = os.path.join(MODELS_DIR, "scaler_tb_v2.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing v2 scaler: {scaler_path}. "
            f"Expected scaler_tb_v2.pkl but it was not found."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Loaded scaler from scaler_tb_v2.pkl")

    return models, scaler


# ─────────────────────────────────────────────────────────────
# GAME / LINEUP FETCHING
# ─────────────────────────────────────────────────────────────

def fetch_todays_games(date_str=None, offline=False):
    """Fetch games for a given date.

    If offline: load from data/raw/schedule_{year}.zip.
    If online: use MLB Stats API.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    games = []

    if offline:
        year = int(date_str[:4])
        zip_path = os.path.join(RAW_DIR, f"schedule_{year}.zip")
        if not os.path.exists(zip_path):
            print(f"  WARNING: No local schedule for {year}, trying 2025")
            zip_path = os.path.join(RAW_DIR, "schedule_2025.zip")
        if not os.path.exists(zip_path):
            print(f"  ERROR: No local schedule found")
            return []

        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read(zf.namelist()[0]))

        for game in data:
            if game.get("date", "") != date_str:
                continue
            game_pk = game.get("game_pk", 0)
            lineups = fetch_game_lineups(game_pk, date_str, offline=True)
            games.append({
                "game_pk": game_pk,
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_pitcher_id": game.get("home_pitcher_id"),
                "home_pitcher_name": game.get("home_pitcher_name", "TBD"),
                "away_pitcher_id": game.get("away_pitcher_id"),
                "away_pitcher_name": game.get("away_pitcher_name", "TBD"),
                "home_lineup": lineups.get("home", {}),
                "away_lineup": lineups.get("away", {}),
                "batter_hands": lineups.get("batter_hands", {}),
            })
    else:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={date_str}&hydrate=team,probablePitcher"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            resp = urllib.request.urlopen(req, timeout=20)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"  ERROR fetching schedule: {e}")
            return []

        for date_info in data.get("dates", []):
            for game in date_info.get("games", []):
                game_pk = game.get("gamePk", game.get("game_pk"))
                if not game_pk:
                    print(f"  WARNING: Skipping game with no gamePk/game_pk")
                    continue
                home_team = game["teams"]["home"]["team"]["abbreviation"]
                away_team = game["teams"]["away"]["team"]["abbreviation"]

                home_pitcher = game["teams"]["home"].get("probablePitcher", {})
                away_pitcher = game["teams"]["away"].get("probablePitcher", {})

                lineups = fetch_game_lineups(game_pk, date_str, offline=False)

                games.append({
                    "game_pk": game_pk,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pitcher_id": home_pitcher.get("id"),
                    "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                    "home_lineup": lineups.get("home", {}),
                    "away_lineup": lineups.get("away", {}),
                    "batter_hands": lineups.get("batter_hands", {}),
                })

    return games


def fetch_game_lineups(game_pk, date_str=None, offline=False):
    """Fetch batting order for a game.

    If offline: read from data/raw/historical_lineups.json.
    If online: use v1.1 feed/live.
    """
    if offline:
        try:
            with open(os.path.join(RAW_DIR, "historical_lineups.json")) as f:
                all_lineups = json.load(f)
        except Exception:
            return {"home": {}, "away": {}, "batter_hands": {}}

        # Key formats: "{date}_{game_pk}" or just lookup by game_pk
        if date_str:
            key = f"{date_str}_{game_pk}"
            if key in all_lineups:
                entry = all_lineups[key]
                return {
                    "home": entry.get("home_lineup", {}),
                    "away": entry.get("away_lineup", {}),
                    "batter_hands": entry.get("batter_hands", {}),
                }
            else:
                pass  # lineup not found for this key
        # Try direct game_pk
        str_pk = str(game_pk)
        if str_pk in all_lineups:
            entry = all_lineups[str_pk]
            return {
                "home": entry.get("home_lineup", {}),
                "away": entry.get("away_lineup", {}),
                "batter_hands": entry.get("batter_hands", {}),
            }
        return {"home": {}, "away": {}, "batter_hands": {}}

    # Online: use v1.1 feed/live
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
    except Exception:
        return {"home": {}, "away": {}, "batter_hands": {}}

    boxscore = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    game_data_players = data.get("gameData", {}).get("players", {})

    batter_hands = {}
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
            if 100 <= order <= 900 and order % 100 == 0:
                pos = order // 100
                if pos not in lineup.values():
                    lineup[str(pid)] = pos
            # Extract batter hand — check multiple sources in priority order
            bat_code = None
            # 1) boxscore players (may contain batSide directly)
            bs_from_box = pdata.get("batSide", {})
            if bs_from_box.get("code"):
                bat_code = bs_from_box["code"]
            # 2) gameData.players keyed by "ID{pid}"
            if not bat_code:
                gd_player = game_data_players.get(key, {})
                bs_from_gd = gd_player.get("batSide", {})
                if bs_from_gd.get("code"):
                    bat_code = bs_from_gd["code"]
            # 3) gameData.players keyed by str(pid)
            if not bat_code:
                gd_player2 = game_data_players.get(str(pid), {})
                bs_from_gd2 = gd_player2.get("batSide", {})
                if bs_from_gd2.get("code"):
                    bat_code = bs_from_gd2["code"]
            if bat_code:
                batter_hands[str(pid)] = bat_code
        result[side] = lineup

    result["batter_hands"] = batter_hands
    return result


# ─────────────────────────────────────────────────────────────
# FEATURE HYDRATION
# ─────────────────────────────────────────────────────────────

def hydrate_features(game, side, player_id, pos, player_games, pitcher_hist,
                     team_games, batter_sc, pitcher_sc, pitcher_basics,
                     live_game_info=None):
    """Build a feature vector for a player, starting from DEFAULTS and
    overriding with real data wherever available.

    Returns dict with all 34 FEATURES keys filled.
    """
    features = dict(DEFAULTS)
    game_date = game.get("date", "")  # set by main() loop
    pid = int(player_id)

    # ── Player rolling / season stats ──
    p_games = player_games.get(pid, [])
    if p_games:
        try:
            roll = compute_rolling_stats(p_games, game_date)
            for k, v in roll.items():
                if k in features:
                    features[k] = v
        except Exception:
            pass

    # ── Batter Statcast ──
    sc = batter_sc.get(pid, {})
    if sc:
        for k in ("xba", "xwoba", "barrel_pct", "avg_exit_velo", "launch_angle"):
            if k in sc and sc[k] is not None:
                features[k] = sc[k]
        if "sprint_speed" in sc and sc["sprint_speed"] is not None:
            features["sprint_speed"] = sc["sprint_speed"]

    # ── Pitcher features ──
    pitcher_id = game.get(f"{side}_pitcher_id")  # opposing pitcher
    if side == "away":
        pitcher_id = game.get("home_pitcher_id")
    else:
        pitcher_id = game.get("away_pitcher_id")

    if pitcher_id:
        pid_int = int(pitcher_id)

        # Pitcher Statcast
        psc = pitcher_sc.get(pid_int, {})
        if psc:
            for k in ("pitcher_xslg_against", "pitcher_xwoba_against",
                      "pitcher_barrel_pct_allowed", "pitcher_hard_hit_pct_allowed",
                      "pitcher_avg_ev_allowed"):
                if k in psc and psc[k] is not None:
                    features[k] = psc[k]

        # Pitcher basics
        pb = pitcher_basics.get(pid_int, {})
        if pb:
            if "whip" in pb and pb["whip"] is not None:
                try:
                    features["pitcher_whip"] = float(pb["whip"])
                except (ValueError, TypeError):
                    pass
            if "ops_against" in pb and pb["ops_against"] is not None:
                try:
                    features["pitcher_ops_against"] = float(pb["ops_against"])
                except (ValueError, TypeError):
                    pass
            # vs batter hand slg
            batter_hand = None
            if live_game_info and "batter_hands" in live_game_info:
                batter_hand = live_game_info["batter_hands"].get(str(player_id))
            if batter_hand == "L" and "vs_lhb" in pb:
                try:
                    features["pitcher_vs_batter_hand_slg"] = float(pb["vs_lhb"].get("slg", 0.400))
                except (ValueError, TypeError):
                    pass
            elif batter_hand == "R" and "vs_rhb" in pb:
                try:
                    features["pitcher_vs_batter_hand_slg"] = float(pb["vs_rhb"].get("slg", 0.400))
                except (ValueError, TypeError):
                    pass

        # Pitcher recent form
        try:
            p_hist = pitcher_hist.get(pid_int, [])
            if p_hist:
                recent = compute_pitcher_recent(p_hist, game_date)
                features["pitcher_recent_era_5g"] = recent.get("pitcher_recent_era_5g", 4.00)
                rest = compute_pitcher_days_rest(p_hist, game_date)
                features["pitcher_days_rest"] = rest.get("pitcher_days_rest", 4)
        except Exception:
            pass

    # ── Context features ──
    features["is_home"] = 1 if side == "home" else 0
    features["lineup_position"] = pos

    # Park factor — use home team's park
    home_team = game.get("home_team", "")
    features["park_factor"] = PARK_FACTORS_TB.get(home_team, 1.0)
    features["park_hr_factor"] = PARK_FACTORS_HR.get(home_team, 1.0)
    features["is_dome"] = 1 if home_team in DOME_STADIUMS else 0

    # Temperature proxy from month
    try:
        month = int(game_date[5:7])
    except (ValueError, IndexError):
        month = 6
    features["temp_proxy"] = MONTH_TEMP.get(month, 70)

    # Platoon advantage
    batter_hand = None
    if live_game_info and "batter_hands" in live_game_info:
        batter_hand = live_game_info["batter_hands"].get(str(player_id))
    pitcher_hand = None
    if pitcher_id:
        pb = pitcher_basics.get(int(pitcher_id), {})
        pitcher_hand = pb.get("hand", None)
    if batter_hand and pitcher_hand:
        if (batter_hand == "L" and pitcher_hand == "R") or (batter_hand == "R" and pitcher_hand == "L"):
            features["platoon_advantage"] = 1
        else:
            features["platoon_advantage"] = 0

    # Team OPS proxy
    team_key = game.get(f"{side}_team", "")
    tg = team_games.get(team_key, [])
    if tg:
        try:
            team_ops = compute_team_ops_proxy(tg, game_date)
            features["lineup_team_ops"] = team_ops.get("lineup_team_ops", 0.640)
        except Exception:
            pass

    return features


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2+ TB Live Predictor (v2 models)")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--offline", action="store_true",
                        help="Use local data only; no MLB API calls")
    parser.add_argument("--api-delay", type=float, default=1.2,
                        help="Seconds between MLB API calls (default: 1.2)")
    parser.add_argument("--cache-dir", default=RAW_DIR,
                        help="Cache directory for API responses (default: data/raw)")
    parser.add_argument("--max-predictions", type=int, default=0,
                        help="Max predictions to output (0 = no limit)")
    parser.add_argument("--skip-gamelog-fetch", action="store_true",
                        help="Skip MLB gamelog API fetch; use local/prior-year only")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"2+ TB Live Predictor (v2 models, 34 features)")
    print(f"  Date: {date_str}")
    print(f"  Mode: {'OFFLINE' if args.offline else 'ONLINE'}")

    models, scaler = load_models()
    print(f"  Models loaded: {list(models.keys())}")
    print(f"  Features: {len(FEATURES)}")

    # ── Fetch games ──
    games = fetch_todays_games(date_str=date_str, offline=args.offline)
    print(f"  Found {len(games)} games")

    if not games:
        print("No games found. Exiting.")
        # Write empty output
        output_path = os.path.join(os.path.dirname(__file__), "..", "results", "live_predictions.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([], f)
        print(f"Saved empty results to {output_path}")
        return

    # ── Load local histories ──
    print("Loading local player/pitcher histories...")
    local_games = load_local_player_games(date_str)
    print(f"  Loaded {len(local_games)} local game rows")

    # Build player_id -> sorted game list from local data
    player_games = defaultdict(list)
    for row in local_games:
        pid = row.get("player_id")
        if pid:
            player_games[int(pid)].append(row)
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g.get("date", ""))

    pitcher_hist = load_pitcher_histories(local_games)
    team_games = load_team_games(local_games)

    # ── Load statcast + pitcher basics ──
    print("Loading statcast lookups...")
    batter_sc, pitcher_sc = load_statcast_lookups(date_str)
    print(f"  Batter statcast: {len(batter_sc)} entries")
    print(f"  Pitcher statcast: {len(pitcher_sc)} entries")

    print("Loading pitcher basics...")
    pitcher_basics = load_pitcher_basics(date_str)
    print(f"  Pitcher basics: {len(pitcher_basics)} entries")

    # ── Optionally fetch current-season gamelogs from MLB API ──
    if not args.skip_gamelog_fetch:
        if args.offline:
            # Offline: load batter and pitcher gamelogs from cache for the current season
            season = int(date_str[:4])

            # --- Batters ---
            all_batter_ids = set()
            for game in games:
                for side in ["home", "away"]:
                    lineup = game.get(f"{side}_lineup", {})
                    for pid_str in lineup:
                        try:
                            all_batter_ids.add(int(pid_str))
                        except (ValueError, TypeError):
                            pass
            if all_batter_ids:
                cache_path = os.path.join(args.cache_dir, f"gamelogs_{season}_hitting.json")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path) as f:
                            cached_rows = json.load(f)
                    except Exception as e:
                        print(f"  WARNING: Failed to load cached batter gamelogs: {e}")
                    else:
                        batter_rows = [row for row in cached_rows if int(row.get("player_id", 0)) in all_batter_ids]
                        # Merge into local_games and rebuild player_games and team_games
                        merged = merge_gamelog_rows(local_games, batter_rows)
                        player_games = defaultdict(list)
                        for row in merged:
                            pid = row.get("player_id")
                            if pid:
                                player_games[int(pid)].append(row)
                        for pid in player_games:
                            player_games[pid].sort(key=lambda g: g.get("date", ""))
                        team_games = defaultdict(list)
                        for row in merged:
                            team = row.get("team", "")
                            if team:
                                team_games[team].append(row)
                        for t in team_games:
                            team_games[t].sort(key=lambda g: g.get("date", ""))
                        print(f"  Loaded {len(batter_rows)} cached batter gamelog rows from {cache_path}")

            # --- Pitchers ---
            all_pitcher_ids = set()
            for game in games:
                for side in ["home", "away"]:
                    ppid = game.get(f"{side}_pitcher_id")
                    if ppid:
                        try:
                            all_pitcher_ids.add(int(ppid))
                        except (ValueError, TypeError):
                            pass
            if all_pitcher_ids:
                cache_path = os.path.join(args.cache_dir, f"gamelogs_{season}_pitching.json")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path) as f:
                            cached_rows = json.load(f)
                    except Exception as e:
                        print(f"  WARNING: Failed to load cached pitcher gamelogs: {e}")
                    else:
                        pitcher_rows = [row for row in cached_rows if int(row.get("player_id", 0)) in all_pitcher_ids]
                        for row in pitcher_rows:
                            pid = int(row.get("player_id", 0))
                            pitcher_hist[pid].append(row)
                        for pid in pitcher_hist:
                            pitcher_hist[pid].sort(key=lambda g: g.get("date", ""))
                        print(f"  Loaded {len(pitcher_rows)} cached pitcher gamelog rows from {cache_path}")
        else:
            # Online mode: use the existing block that fetches and caches
            # Collect all player IDs from lineups and pitcher IDs
            all_batter_ids = set()
            all_pitcher_ids = set()
            for game in games:
                for side in ["home", "away"]:
                    lineup = game.get(f"{side}_lineup", {})
                    for pid_str in lineup:
                        try:
                            all_batter_ids.add(int(pid_str))
                        except (ValueError, TypeError):
                            pass
                    ppid = game.get(f"{side}_pitcher_id")
                    if ppid:
                        try:
                            all_pitcher_ids.add(int(ppid))
                        except (ValueError, TypeError):
                            pass

            season = int(date_str[:4])
            print(f"Fetching current-season gamelogs for {len(all_batter_ids)} batters...")
            try:
                new_batter_rows = fetch_player_gamelogs(
                    list(all_batter_ids), season, "hitting",
                    args.cache_dir, args.api_delay
                )
                print(f"  Fetched {len(new_batter_rows)} batter gamelog rows")
                merged = merge_gamelog_rows(local_games, new_batter_rows)
                # Rebuild player_games
                for row in new_batter_rows:
                    pid = row.get("player_id")
                    if pid:
                        player_games[int(pid)].append(row)
                for pid in player_games:
                    player_games[pid].sort(key=lambda g: g.get("date", ""))
                # Also update team_games so lineup_team_ops uses current-season context
                for row in new_batter_rows:
                    team = row.get("team", "")
                    if team:
                        team_games[team].append(row)
                for t in team_games:
                    team_games[t].sort(key=lambda g: g.get("date", ""))
            except Exception as e:
                print(f"  WARNING: Batter gamelog fetch failed: {e}")

            print(f"Fetching current-season gamelogs for {len(all_pitcher_ids)} pitchers...")
            try:
                new_pitcher_rows = fetch_player_gamelogs(
                    list(all_pitcher_ids), season, "pitching",
                    args.cache_dir, args.api_delay
                )
                print(f"  Fetched {len(new_pitcher_rows)} pitcher gamelog rows")
                for row in new_pitcher_rows:
                    pid = row.get("player_id")
                    if pid:
                        pitcher_hist[int(pid)].append(row)
                for pid in pitcher_hist:
                    pitcher_hist[pid].sort(key=lambda g: g.get("date", ""))
            except Exception as e:
                print(f"  WARNING: Pitcher gamelog fetch failed: {e}")
    # ── Predict ──
    all_predictions = []
    pred_count = 0

    for game in games:
        print(f"\n{game['away_team']} @ {game['home_team']}")
        live_info = {"batter_hands": game.get("batter_hands", {})}

        for side, team_key, opp_key in [
            ("away", "away_team", "home_team"),
            ("home", "home_team", "away_team"),
        ]:
            lineup = game.get(f"{side}_lineup", {})
            for player_id_str, pos in lineup.items():
                if pos > 5:  # Only lineup spots 1-5
                    continue

                if args.max_predictions > 0 and pred_count >= args.max_predictions:
                    break

                # Hydrate features
                game["date"] = date_str
                features = hydrate_features(
                    game=game,
                    side=side,
                    player_id=player_id_str,
                    pos=pos,
                    player_games=player_games,
                    pitcher_hist=pitcher_hist,
                    team_games=team_games,
                    batter_sc=batter_sc,
                    pitcher_sc=pitcher_sc,
                    pitcher_basics=pitcher_basics,
                    live_game_info=live_info,
                )

                X_raw = pd.DataFrame([[features[f] for f in FEATURES]], columns=FEATURES)
                if scaler:
                    X_scaled = scaler.transform(X_raw.values)
                else:
                    X_scaled = X_raw.values

                probas = []
                failed_models = []
                for name, model in models.items():
                        try:
                            if name == "lgbm":
                                import lightgbm
                                X_model = pd.DataFrame(X_scaled, columns=FEATURES)
                            else:
                                X_model = X_scaled
                            p = model.predict_proba(X_model)[0, 1]
                            probas.append(p)
                        except ImportError:
                            print("ERROR: LightGBM is required for the ensemble. Please run: pip install lightgbm")
                            raise SystemExit(1)
                        except Exception as e:
                            failed_models.append((name, str(e)))

                if failed_models:
                    for name, err in failed_models:
                        print(f"  WARNING: {name} prediction failed: {err}")

                if probas:
                    avg_proba = np.mean(probas)
                else:
                    avg_proba = 0.0

                all_predictions.append({
                    "game_pk": game["game_pk"],
                    "team": game[team_key],
                    "opponent": game[opp_key],
                    "player_id": int(player_id_str),
                    "lineup_position": pos,
                    "is_home": features["is_home"],
                    "predicted_proba_2tb": round(float(avg_proba), 4),
                    "model_count": len(probas),
                })
                pred_count += 1

            if args.max_predictions > 0 and pred_count >= args.max_predictions:
                break
        if args.max_predictions > 0 and pred_count >= args.max_predictions:
            break

    # Sort by probability
    all_predictions.sort(key=lambda x: x["predicted_proba_2tb"], reverse=True)

    print(f"\nTop 20 predictions:")
    for p in all_predictions[:20]:
        print(f"  {p['team']:3s} #{p['lineup_position']} player_{p['player_id']}: {p['predicted_proba_2tb']:.3f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "live_predictions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nSaved {len(all_predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
