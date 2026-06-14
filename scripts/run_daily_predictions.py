#!/usr/bin/env python3
"""
run_daily_predictions.py
========================

Daily prediction generation pipeline for the frozen 2TB model stack.

This script reuses the existing inference code from tb_predict_live.py to
generate predictions for a given date (default: today) and saves them in
an organized archive structure.

Outputs:
  - predictions/<year>/<yyyymmdd>_predictions.csv
  - predictions/<year>/<yyyymmdd>_summary.json
  - logs/prediction_runs/<yyyymmdd>.log

Usage:
    python3 scripts/run_daily_predictions.py
    python3 scripts/run_daily_predictions.py --date 2025-06-01
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
import numpy as np

# Add the scripts directory to the path to import tb_predict_live
sys.path.append(os.path.join(os.path.dirname(__file__)))
import tb_predict_live


def setup_logging(log_dir, date_str):
    """Set up logging to a file in log_dir/<date_str>.log."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{date_str}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Daily 2TB model prediction pipeline")
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local data only; no MLB API calls"
    )
    args = parser.parse_args()

    # Determine date
    if args.date:
        date_str = args.date
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format {date_str}. Expected YYYY-MM-DD.")
            sys.exit(1)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Setup directories
    predictions_base = os.path.join(os.path.dirname(__file__), "..", "predictions")
    logs_base = os.path.join(os.path.dirname(__file__), "..", "logs", "prediction_runs")
    year = date_str[:4]
    date_compact = date_str.replace("-", "")
    predictions_dir = os.path.join(predictions_base, year)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    csv_path = os.path.join(predictions_dir, f"{date_compact}_predictions.csv")
    json_path = os.path.join(predictions_dir, f"{date_compact}_summary.json")
    log_file = setup_logging(logs_base, date_compact)

    logging.info(f"Starting daily prediction pipeline for date: {date_str}")
    logging.info(f"Predictions CSV: {csv_path}")
    logging.info(f"Summary JSON: {json_path}")
    logging.info(f"Log file: {log_file}")

    # Check if output files already exist (to avoid overwriting)
    if os.path.exists(csv_path) or os.path.exists(json_path):
        logging.warning(f"Output files for {date_str} already exist. Skipping to avoid overwriting.")
        print(f"Output files for {date_str} already exist. Skipping.")
        sys.exit(0)

    start_time = datetime.now()

    try:
        # Load models and scaler using existing inference code
        logging.info("Loading models and scaler...")
        models, scaler = tb_predict_live.load_models()
        logging.info(f"Models loaded: {list(models.keys())}")

        # Fetch games for the given date
        logging.info(f"Fetching games for {date_str} (offline={args.offline})...")
        games = tb_predict_live.fetch_todays_games(date_str=date_str, offline=args.offline)
        logging.info(f"Found {len(games)} games")

        if not games:
            logging.warning("No games found. Exiting.")
            # Write empty outputs? The task says generate predictions for each player, so if no games, we output empty.
            # We'll create empty CSV and JSON summary with zeros.
            total_games = 0
            total_players = 0
            top_prediction = None
            top_probability = 0.0
        else:
            # Load local histories and other data (same as tb_predict_live)
            logging.info("Loading local player/pitcher histories...")
            local_games = tb_predict_live.load_local_player_games(date_str)
            logging.info(f"Loaded {len(local_games)} local game rows")

            # Build player_id -> sorted game list from local data
            player_games = {}
            for row in local_games:
                pid = row.get("player_id")
                if pid:
                    pid_int = int(pid)
                    if pid_int not in player_games:
                        player_games[pid_int] = []
                    player_games[pid_int].append(row)
            for pid in player_games:
                player_games[pid].sort(key=lambda g: g.get("date", ""))

            pitcher_hist = tb_predict_live.load_pitcher_histories(local_games)
            team_games = tb_predict_live.load_team_games(local_games)

            # Load statcast + pitcher basics
            logging.info("Loading statcast lookups...")
            batter_sc, pitcher_sc = tb_predict_live.load_statcast_lookups(date_str)
            logging.info(f"  Batter statcast: {len(batter_sc)} entries")
            logging.info(f"  Pitcher statcast: {len(pitcher_sc)} entries")

            logging.info("Loading pitcher basics...")
            pitcher_basics = tb_predict_live.load_pitcher_basics(date_str)
            logging.info(f"  Pitcher basics: {len(pitcher_basics)} entries")

            # Load player name lookup
            logging.info("Loading player name lookup...")
            player_names = tb_predict_live.load_player_names(date_str)
            logging.info(f"  Player names: {len(player_names)} entries")

            # Predict for each game and lineup spots 1-5
            logging.info("Generating predictions...")
            all_predictions = []
            pred_count = 0

            for game in games:
                # Add date to game dict for hydrate_features
                game["date"] = date_str
                live_info = {"batter_hands": game.get("batter_hands", {})}

                for side, team_key, opp_key in [
                    ("away", "away_team", "home_team"),
                    ("home", "home_team", "away_team"),
                ]:
                    lineup = game.get(f"{side}_lineup", {})
                    for player_id_str, pos in lineup.items():
                        if pos > 5:  # Only lineup spots 1-5
                            continue

                        # Hydrate features
                        features = tb_predict_live.hydrate_features(
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

                        # Prepare feature vector
                        X_raw = np.array([[features[f] for f in tb_predict_live.FEATURES]])
                        if scaler:
                            X_scaled = scaler.transform(X_raw)
                        else:
                            X_scaled = X_raw

                        # Get predictions from each model
                        probas = []
                        failed_models = []
                        for name, model in models.items():
                            try:
                                if name == "lgbm":
                                    # LightGBM expects a DataFrame with feature names
                                    import lightgbm
                                    X_model = tb_predict_live.pd.DataFrame(X_scaled, columns=tb_predict_live.FEATURES)
                                else:
                                    X_model = X_scaled
                                p = model.predict_proba(X_model)[0, 1]
                                probas.append(p)
                            except ImportError:
                                logging.error("LightGBM is required for the ensemble.")
                                raise
                            except Exception as e:
                                failed_models.append((name, str(e)))

                        if failed_models:
                            for name, err in failed_models:
                                logging.warning(f"{name} prediction failed: {err}")

                        if probas:
                            avg_proba = np.mean(probas)
                        else:
                            avg_proba = 0.0

                        all_predictions.append({
                            "date": date_str,
                            "game_pk": game["game_pk"],
                            "team": game[team_key],
                            "opponent": game[opp_key],
                            "player_id": int(player_id_str),
                            "player_name": player_names.get(int(player_id_str), "Unknown"),
                            "lineup_position": pos,
                            "is_home": features["is_home"],
                            "predicted_proba_2tb": round(float(avg_proba), 4),
                            "model_count": len(probas),
                        })
                        pred_count += 1

            # Sort by probability descending
            all_predictions.sort(key=lambda x: x["predicted_proba_2tb"], reverse=True)
            total_games = len(games)
            total_players = len(all_predictions)

            logging.info(f"Generated {total_players} predictions from {total_games} games.")

            # Determine top prediction
            if all_predictions:
                top_pred = all_predictions[0]
                top_prediction = {
                    "player_id": top_pred["player_id"],
                    "player_name": top_pred.get("player_name", "Unknown"),
                    "team": top_pred["team"],
                    "opponent": top_pred["opponent"],
                    "lineup_position": top_pred["lineup_position"],
                    "predicted_proba_2tb": top_pred["predicted_proba_2tb"],
                }
                top_probability = top_pred["predicted_proba_2tb"]
            else:
                top_prediction = None
                top_probability = 0.0

        # Save predictions CSV
        logging.info(f"Saving predictions to {csv_path}")
        if all_predictions:
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "date", "game_pk", "team", "opponent", "player_id",
                    "player_name", "lineup_position", "is_home", "predicted_proba_2tb", "model_count"
                ])
                # Write header
                writer.writeheader()
                for pred in all_predictions:
                    writer.writerow(pred)
        else:
            # Write empty CSV with header
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "date", "game_pk", "team", "opponent", "player_id",
                    "player_name", "lineup_position", "is_home", "predicted_proba_2tb", "model_count"
                ])
                writer.writeheader()

        # Save summary JSON
        end_time = datetime.now()
        pipeline_runtime_seconds = (end_time - start_time).total_seconds()

        summary = {
            "date": date_str,
            "total_games": total_games,
            "total_players": total_players,
            "top_prediction": top_prediction,
            "top_probability": top_probability,
            "pipeline_runtime_seconds": round(pipeline_runtime_seconds, 2),
            "model_version": "v2",
        }

        logging.info(f"Saving summary to {json_path}")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also write live_predictions.json for the dashboard (list of predictions)
        live_path = os.path.join(os.path.dirname(__file__), "..", "results", "live_predictions.json")
        try:
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            with open(live_path, "w") as f:
                json.dump(all_predictions, f, indent=2)
            logging.info(f"Wrote live predictions JSON for dashboard: {live_path}")
        except Exception as e:
            logging.error(f"Failed to write live predictions JSON: {e}")

        logging.info(
            f"Pipeline completed successfully in {pipeline_runtime_seconds:.2f} seconds. "
            f"Games: {total_games}, Players: {total_players}, Top probability: {top_probability:.4f}"
        )

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()