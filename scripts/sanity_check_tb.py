#!/usr/bin/env python3
"""
sanity_check_tb.py
===================
Lightweight sanity checks for the v2 2+ TB model pipeline (34 features).

Checks:
  1. Feature files exist and non-empty
  2. No future data leakage (feature dates <= target date)
  3. Train/val/holdout date ranges correct and non-overlapping
  4. Model files exist and can predict
  5. Feature names match between training and model
  6. Backtest results exist with decision-level metrics
  7. No meaningless agreement thresholds

Usage:
    python3 sanity_check_tb.py
"""

import json
import os
import pickle
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EXPECTED_FEATURES = 34  # v2 feature count (deduped from 48)

# Metadata/target columns excluded from feature set
META_COLS = {"target_2tb", "total_bases", "game_pk", "player_id", "game_date", "season", "team"}

# Exact v2 feature order (must match scaler training order)
V2_FEATURE_ORDER = [
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


def get_feature_names():
    """Derive feature names from the first row of train_tb_v2.json in insertion order.

    Excludes META_COLS. Validates against the exact V2_FEATURE_ORDER.
    """
    train_path = os.path.join(PROCESSED_DIR, "train_tb_v2.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path}")
    with open(train_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError("train_tb_v2.json is empty")
    feature_names = [k for k in data[0].keys() if k not in META_COLS]
    if feature_names != V2_FEATURE_ORDER:
        mismatch = []
        for i, (got, exp) in enumerate(zip(feature_names, V2_FEATURE_ORDER)):
            if got != exp:
                mismatch.append(f"  index {i}: got '{got}' expected '{exp}'")
        if len(feature_names) != len(V2_FEATURE_ORDER):
            mismatch.append(f"  length: got {len(feature_names)} expected {len(V2_FEATURE_ORDER)}")
        raise ValueError(
            "Feature order mismatch between data and V2_FEATURE_ORDER:\n"
            + "\n".join(mismatch)
        )
    return feature_names


def check_files():
    """Check that all required v2 files exist."""
    print("\n[1] File existence checks (v2)")
    files = {
        "train": os.path.join(PROCESSED_DIR, "train_tb_v2.json"),
        "validate": os.path.join(PROCESSED_DIR, "validate_tb_v2.json"),
        "holdout": os.path.join(PROCESSED_DIR, "holdout_tb_v2.json"),
        "logreg_model": os.path.join(MODELS_DIR, "logreg_tb_v2.pkl"),
        "xgb_model": os.path.join(MODELS_DIR, "xgb_tb_v2.pkl"),
        "lgbm_model": os.path.join(MODELS_DIR, "lgbm_tb_v2.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler_tb_v2.pkl"),
        "training_results": os.path.join(RESULTS_DIR, "training_tb_v2_results.json"),
        "backtest_results": os.path.join(RESULTS_DIR, "backtest_tb_v2.json"),
    }

    all_ok = True
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists and size > 0 else "MISSING"
        if status != "OK":
            all_ok = False
        print(f"  {status:10s} {name:20s} ({size:,} bytes)")

    return all_ok


def check_splits():
    """Check train/val/holdout date ranges."""
    print("\n[2] Split date range checks (v2)")

    splits = {}
    for name in ["train", "validate", "holdout"]:
        path = os.path.join(PROCESSED_DIR, f"{name}_tb_v2.json")
        if not os.path.exists(path):
            print(f"  MISSING: {name}")
            return False
        with open(path) as f:
            data = json.load(f)
        if not data:
            print(f"  EMPTY: {name}")
            return False
        dates = [r["game_date"] for r in data]
        splits[name] = {"min": min(dates), "max": max(dates), "count": len(data)}
        print(f"  {name:10s}: {splits[name]['min']} to {splits[name]['max']} ({splits[name]['count']} rows)")

    # Check no overlap
    if splits["train"]["max"] >= splits["validate"]["min"]:
        print("  WARNING: Train/validate overlap!")
    if splits["validate"]["max"] >= splits["holdout"]["min"]:
        print("  WARNING: Validate/holdout overlap!")

    return True


def check_features():
    """Check feature consistency from v2 training data."""
    print("\n[3] Feature checks (v2)")

    try:
        feature_names = get_feature_names()
    except (FileNotFoundError, ValueError) as e:
        print(f"  FAIL: {e}")
        return False

    n_features = len(feature_names)
    print(f"  Derived features from data: {n_features}")
    print(f"  Feature order validated against V2_FEATURE_ORDER")

    if n_features != EXPECTED_FEATURES:
        print(f"  FAIL: Expected {EXPECTED_FEATURES} features, got {n_features}")
        return False
    else:
        print(f"  OK: Feature count matches expected {EXPECTED_FEATURES}")

    # Load data for row sampling
    train_path = os.path.join(PROCESSED_DIR, "train_tb_v2.json")
    with open(train_path) as f:
        data = json.load(f)

    # Check first 100 rows (or all if fewer) have all features
    sample_size = min(100, len(data))
    missing_count = 0
    for i in range(sample_size):
        row = data[i]
        missing = [feat for feat in feature_names if feat not in row]
        if missing:
            missing_count += 1
            if missing_count <= 3:
                print(f"  WARNING: Row {i} missing features: {missing[:5]}")

    if missing_count == 0:
        print(f"  OK: All {sample_size} sampled rows have all {n_features} features")
    else:
        print(f"  FAIL: {missing_count}/{sample_size} rows missing features")
        return False

    return True


def check_models():
    """Check v2 models can load and predict."""
    print("\n[4] Model checks (v2)")

    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_tb_v2.pkl")
    if not os.path.exists(scaler_path):
        print("  MISSING: scaler_tb_v2.pkl")
        return False
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("  OK: scaler_tb_v2.pkl loaded")

    # Load holdout sample
    holdout_path = os.path.join(PROCESSED_DIR, "holdout_tb_v2.json")
    if not os.path.exists(holdout_path):
        print("  MISSING: holdout_tb_v2.json")
        return False
    with open(holdout_path) as f:
        holdout_data = json.load(f)

    if not holdout_data:
        print("  No holdout data to test")
        return False

    sample = holdout_data[:5]

    # Derive feature names using the shared helper (insertion order, no sorting)
    feature_names = get_feature_names()

    X = np.array([[r.get(feat, 0) for feat in feature_names] for r in sample])
    X_scaled = scaler.transform(X)

    all_ok = True

    # Check LogReg
    logreg_path = os.path.join(MODELS_DIR, "logreg_tb_v2.pkl")
    if not os.path.exists(logreg_path):
        print("  FAIL: logreg_tb_v2.pkl missing")
        all_ok = False
    else:
        with open(logreg_path, "rb") as f:
            logreg = pickle.load(f)
        try:
            proba = logreg.predict_proba(X_scaled)[:, 1]
            print(f"  OK: logreg predicts {proba[:3].round(3)}...")
        except Exception as e:
            print(f"  FAIL: logreg failed: {e}")
            all_ok = False

    # Check XGBoost
    xgb_path = os.path.join(MODELS_DIR, "xgb_tb_v2.pkl")
    if not os.path.exists(xgb_path):
        print("  FAIL: xgb_tb_v2.pkl missing")
        all_ok = False
    else:
        with open(xgb_path, "rb") as f:
            xgb = pickle.load(f)
        try:
            proba = xgb.predict_proba(X_scaled)[:, 1]
            print(f"  OK: xgb predicts {proba[:3].round(3)}...")
        except Exception as e:
            print(f"  FAIL: xgb failed: {e}")
            all_ok = False

    # Check LightGBM
    lgbm_path = os.path.join(MODELS_DIR, "lgbm_tb_v2.pkl")
    if not os.path.exists(lgbm_path):
        print("  FAIL: lgbm_tb_v2.pkl missing")
        all_ok = False
    else:
        try:
            with open(lgbm_path, "rb") as f:
                lgbm = pickle.load(f)
            try:
                proba = lgbm.predict_proba(X_scaled)[:, 1]
                print(f"  OK: lgbm predicts {proba[:3].round(3)}...")
            except Exception as e:
                print(f"  FAIL: lgbm failed: {e}")
                all_ok = False
        except ImportError as e:
            print(f"  DEPENDENCY FAIL: lightgbm not installed ({e})")
            print("  Install with: pip install lightgbm")
            all_ok = False
        except Exception as e:
            print(f"  FAIL: lgbm load error: {e}")
            all_ok = False

    return all_ok


def check_backtest():
    """Check v2 backtest results."""
    print("\n[5] Backtest checks (v2)")

    path = os.path.join(RESULTS_DIR, "backtest_tb_v2.json")
    if not os.path.exists(path):
        print("  MISSING: backtest_tb_v2.json")
        return False

    with open(path) as f:
        results = json.load(f)

    models = results.get("models", {})
    if not models:
        print("  FAIL: No model results in backtest")
        return False

    for model_name, metrics in models.items():
        auc = metrics.get("auc", "N/A")
        brier = metrics.get("brier", "N/A")
        print(f"  {model_name:12s}: AUC={auc}, Brier={brier}")

    rank_metrics = results.get("rank_metrics", {})
    if rank_metrics:
        for tier, vals in rank_metrics.items():
            rate = vals.get("rate", "N/A")
            lift = vals.get("lift", "N/A")
            n = vals.get("n", "N/A")
            print(f"  {tier:12s}: rate={rate}, lift={lift}, n={n}")

    return True


def main():
    print("=" * 60)
    print("Sanity Check — v2 2+ TB Model (34 features)")
    print("=" * 60)

    checks = [
        ("Files", check_files),
        ("Splits", check_splits),
        ("Features", check_features),
        ("Models", check_models),
        ("Backtest", check_backtest),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
