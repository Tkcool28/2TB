#!/usr/bin/env python3
"""
smoke_test_live_v2.py
=====================
Smoke test for the v2 live predictor.

Verifies:
  - v2 models + scaler load cleanly
  - 34-feature schema is intact
  - DEFAULTS covers every feature
  - All 3 v2 models produce probabilities in [0, 1] on a default feature vector
  - Optionally runs the live script offline with a known date and asserts
    it completes without the LightGBM feature-name warning

Usage:
    python3 scripts/smoke_test_live_v2.py
    python3 scripts/smoke_test_live_v2.py --offline-date 2024-06-01 --max-predictions 5
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

# ── Mirror the constants from tb_predict_live.py ──
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LIVE_SCRIPT = os.path.join(os.path.dirname(__file__), "tb_predict_live.py")

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


def load_models():
    """Load v2 model files and scaler.  Mirrors tb_predict_live.load_models()."""
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
            raise FileNotFoundError(f"Missing v2 model file: {path}")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

    scaler_path = os.path.join(MODELS_DIR, "scaler_tb_v2.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing v2 scaler: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return models, scaler


def build_default_vector():
    """Return a numpy array of shape (1, 34) from DEFAULTS in FEATURES order."""
    return np.array([[DEFAULTS[f] for f in FEATURES]])


def main():
    parser = argparse.ArgumentParser(description="Smoke test for v2 live predictor")
    parser.add_argument("--offline-date", default=None,
                        help="YYYY-MM-DD — run offline live prediction for this date")
    parser.add_argument("--max-predictions", type=int, default=5,
                        help="Max predictions for offline run (default: 5)")
    args = parser.parse_args()

    report = {}

    # ── 1. Load models & scaler ──
    models, scaler = load_models()

    # ── 2. Schema checks ──
    assert len(FEATURES) == 34, "Feature count mismatch"
    missing = set(FEATURES) - set(DEFAULTS.keys())
    assert not missing, f"DEFAULTS missing: {missing}"

    # ── 3. Predict on default vector ──
    X_raw = build_default_vector()
    X_scaled = scaler.transform(X_raw)

    model_probs = {}
    all_ok = True
    for name, model in models.items():
        # LightGBM was fitted with feature names — pass DataFrame to suppress warning.
        # Other models were fitted on bare numpy — pass numpy.
        X_model = pd.DataFrame(X_scaled, columns=FEATURES) if name == "lgbm" else X_scaled
        p = model.predict_proba(X_model)[0, 1]
        p = float(p)
        model_probs[name] = round(p, 6)
        if not (0.0 <= p <= 1.0):
            all_ok = False

    report["status"] = "ok" if all_ok else "fail"
    report["feature_count"] = len(FEATURES)
    report["models"] = model_probs

    # ── 4. Optional offline subprocess test ──
    if args.offline_date:
        cmd = [
            sys.executable, LIVE_SCRIPT,
            "--date", args.offline_date,
            "--offline",
            "--max-predictions", str(args.max_predictions),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )

        offline_report = {"date": args.offline_date}

        # Assert return code
        rc_ok = proc.returncode == 0
        offline_report["return_code"] = proc.returncode
        offline_report["return_code_ok"] = rc_ok

        # Assert no LGBM warning
        combined = proc.stdout + proc.stderr
        has_lgbm_warning = "X does not have valid feature names" in combined
        offline_report["warning_check"] = "pass" if not has_lgbm_warning else "fail"

        # Assert output file
        output_path = os.path.join(RESULTS_DIR, "live_predictions.json")
        file_ok = os.path.exists(output_path)
        offline_report["output_file_exists"] = file_ok

        predictions = []
        if file_ok:
            with open(output_path) as f:
                predictions = json.load(f)
            offline_report["prediction_count"] = len(predictions)
            offline_report["within_limit"] = len(predictions) <= args.max_predictions
        else:
            offline_report["prediction_count"] = 0
            offline_report["within_limit"] = False

        # Overall offline status
        offline_ok = rc_ok and not has_lgbm_warning and file_ok and offline_report["within_limit"]
        offline_report["offline_status"] = "ok" if offline_ok else "fail"

        report["offline"] = offline_report

        # Update top-level status
        if not offline_ok:
            report["status"] = "fail"

    # ── Print JSON to stdout ──
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
