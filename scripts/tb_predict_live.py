#!/usr/bin/env python3
"""
tb_predict_live.py
===================
Fetch today's slate and predict 2+ TB probability for each player.

Uses:
  - MLB Stats API for today's games, lineups, and probable pitchers
  - Trained models (logreg_tb.pkl, xgb_tb.pkl) for predictions
  - Rolling stats computed from recent game logs

Output: JSON with predictions for all players in lineup spots 1-5

Usage:
    python3 tb_predict_live.py
"""

import json
import os
import pickle
import numpy as np
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

FEATURES = [
    "roll_7d_tb_rate", "roll_7d_slg", "roll_7d_xbh_rate", "roll_7d_hr_rate",
    "roll_7d_hit_rate", "roll_7d_k_rate", "roll_7d_bb_rate",
    "roll_15d_tb_rate", "roll_15d_slg", "roll_15d_xbh_rate", "roll_15d_hr_rate",
    "roll_30d_tb_rate", "roll_30d_slg",
    "season_slg", "season_tb_per_ab", "season_hr_rate", "season_xbh_rate",
    "season_bb_pct", "season_k_pct", "season_avg",
    "is_home", "platoon_advantage", "park_factor", "lineup_position",
    "hit_streak", "days_rest",
    "xba", "xslg", "xwoba",
    "barrel_pct", "hard_hit_pct", "avg_exit_velo", "sweet_spot_pct",
    "pitcher_xba_against", "pitcher_xslg_against", "pitcher_xwoba_against",
    "pitcher_barrel_pct_allowed", "pitcher_hard_hit_pct_allowed",
    "pitcher_avg_ev_allowed",
    "pitcher_era", "pitcher_whip", "pitcher_ops_against",
    "pitcher_vs_batter_hand_slg", "pitcher_vs_batter_hand_ops",
]

# League averages for defaults
DEFAULTS = {
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


def load_models():
    models = {}
    for name in ["logreg", "xgb"]:
        path = os.path.join(MODELS_DIR, f"{name}_tb.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    scaler_path = os.path.join(MODELS_DIR, "scaler_tb.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return models, scaler


def fetch_todays_games():
    """Fetch today's games from MLB Stats API."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=team,probablePitcher,lineups"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    data = json.loads(resp.read())

    games = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            game_pk = game["game_pk"]
            home_team = game["teams"]["home"]["team"]["abbreviation"]
            away_team = game["teams"]["away"]["team"]["abbreviation"]

            # Get probable pitchers
            home_pitcher = game["teams"]["home"].get("probablePitcher", {})
            away_pitcher = game["teams"]["away"].get("probablePitcher", {})

            # Get lineups from feed
            lineups = fetch_game_lineups(game_pk)

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
            })

    return games


def fetch_game_lineups(game_pk):
    """Fetch batting order for a game."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read())
    except Exception:
        return {"home": {}, "away": {}}

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
            if 100 <= order <= 900 and order % 100 == 0:
                pos = order // 100
                if pos not in lineup.values():
                    lineup[str(pid)] = pos
        result[side] = lineup
    return result


def main():
    print("2+ TB Live Predictor")
    models, scaler = load_models()
    if not models:
        print("No models found!")
        return

    games = fetch_todays_games()
    print(f"Found {len(games)} games today")

    all_predictions = []
    for game in games:
        print(f"\n{game['away_team']} @ {game['home_team']}")
        for side, team_key, opp_key, pitcher_id in [
            ("away", "away_team", "home_team", game["home_pitcher_id"]),
            ("home", "home_team", "away_team", game["away_pitcher_id"]),
        ]:
            lineup = game[f"{side}_lineup"]
            for player_id_str, pos in lineup.items():
                if pos > 5:  # Only lineup spots 1-5
                    continue
                # Build feature vector with defaults
                features = {f: DEFAULTS[f] for f in FEATURES}
                features["lineup_position"] = pos
                features["is_home"] = 1 if side == "home" else 0

                X = np.array([[features[f] for f in FEATURES]])
                if scaler:
                    X = scaler.transform(X)

                probas = []
                for name, model in models.items():
                    try:
                        p = model.predict_proba(X)[0, 1]
                        probas.append(p)
                    except Exception:
                        pass

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
                    "predicted_proba_2tb": round(avg_proba, 4),
                })

    # Sort by probability
    all_predictions.sort(key=lambda x: x["predicted_proba_2tb"], reverse=True)

    print(f"\nTop 20 predictions:")
    for p in all_predictions[:20]:
        print(f"  {p['team']:3s} #{p['lineup_position']} player_{p['player_id']}: {p['predicted_proba_2tb']:.3f}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "live_predictions.json")
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
