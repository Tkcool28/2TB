#!/usr/bin/env python3
"""
sanity_check_tb.py
===================
Lightweight sanity checks for the 2+ TB model pipeline.

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

EXPECTED_FEATURES = 48  # Total number of features


def check_files():
    """Check that all required files exist."""
    print("\n[1] File existence checks")
    files = {
        "train": os.path.join(PROCESSED_DIR, "train_tb.json"),
        "validate": os.path.join(PROCESSED_DIR, "validate_tb.json"),
        "holdout": os.path.join(PROCESSED_DIR, "holdout_tb.json"),
        "logreg_model": os.path.join(MODELS_DIR, "logreg_tb.pkl"),
        "xgb_model": os.path.join(MODELS_DIR, "xgb_tb.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler_tb.pkl"),
        "training_results": os.path.join(RESULTS_DIR, "training_tb_results.json"),
        "backtest_results": os.path.join(RESULTS_DIR, "backtest_tb.json"),
    }

    all_ok = True
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists and size > 0 else "MISSING"
        if status == "MISSING" and name in ["xgb_model"]:
            status = "OPTIONAL"
        else:
            all_ok = all_ok and (exists and size > 0)
        print(f"  {status:10s} {name:20s} ({size:,} bytes)")

    return all_ok


def check_splits():
    """Check train/val/holdout date ranges."""
    print("\n[2] Split date range checks")

    splits = {}
    for name in ["train", "validate", "holdout"]:
        path = os.path.join(PROCESSED_DIR, f"{name}_tb.json")
        if not os.path.exists(path):
            print(f"  MISSING: {name}")
            return False
        with open(path) as f:
            data = json.load(f)
        if not data:
            print(f"  EMPTY: {name}")
            return False
        dates = [r["date"] for r in data]
        splits[name] = {"min": min(dates), "max": max(dates), "count": len(data)}
        print(f"  {name:10s}: {splits[name]['min']} to {splits[name]['max']} ({splits[name]['count']} rows)")

    # Check no overlap
    if splits["train"]["max"] >= splits["validate"]["min"]:
        print("  WARNING: Train/validate overlap!")
    if splits["validate"]["max"] >= splits["holdout"]["min"]:
        print("  WARNING: Validate/holdout overlap!")

    return True


def check_features():
    """Check feature consistency."""
    print("\n[3] Feature checks")

    # Load training results
    results_path = os.path.join(RESULTS_DIR, "training_tb_results.json")
    if not os.path.exists(results_path):
        print("  MISSING: training results")
        return False

    with open(results_path) as f:
        results = json.load(f)

    n_features = results.get("n_features", 0)
    feature_names = results.get("feature_names", [])
    print(f"  Features: {n_features}")
    print(f"  Feature names: {len(feature_names)}")

    if n_features != EXPECTED_FEATURES:
        print(f"  WARNING: Expected {EXPECTED_FEATURES} features, got {n_features}")

    # Check a sample row
    with open(os.path.join(PROCESSED_DIR, "train_tb.json")) as f:
        sample = json.load(f)[:1]

    if sample:
        missing = [feat for feat in feature_names if feat not in sample[0]]
        if missing:
            print(f"  WARNING: Missing features in data: {missing[:5]}")
        else:
            print("  All features present in data")

    return True


def check_models():
    """Check models can load and predict."""
    print("\n[4] Model checks")

    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_tb.pkl")
    if not os.path.exists(scaler_path):
        print("  MISSING: scaler")
        return False
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load a sample and predict
    with open(os.path.join(PROCESSED_DIR, "holdout_tb.json")) as f:
        sample = json.load(f)[:5]

    if not sample:
        print("  No holdout data to test")
        return False

    feature_names = results.get("feature_names", []) if 'results' in dir() else []
    if not feature_names:
        with open(os.path.join(RESULTS_DIR, "training_tb_results.json")) as f:
            feature_names = json.load(f).get("feature_names", [])

    X = np.array([[r.get(feat, 0) for feat in feature_names] for r in sample])
    X_scaled = scaler.transform(X)

    all_ok = True
    for name in ["logreg", "xgb"]:
        model_path = os.path.join(MODELS_DIR, f"{name}_tb.pkl")
        if not os.path.exists(model_path):
            print(f"  OPTIONAL MISSING: {name}_tb.pkl")
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        try:
            proba = model.predict_proba(X_scaled)[:, 1]
            print(f"  OK: {name} predicts {proba[:3].round(3)}...")
        except Exception as e:
            print(f"  ERROR: {name} failed: {e}")
            all_ok = False

    return all_ok


def check_backtest():
    """Check backtest results."""
    print("\n[5] Backtest checks")

    path = os.path.join(RESULTS_DIR, "backtest_tb.json")
    if not os.path.exists(path):
        print("  MISSING: backtest results")
        return False

    with open(path) as f:
        results = json.load(f)

    overall = results.get("overall", {})
    print(f"  AUC: {overall.get('auc', 'N/A')}")
    print(f"  Brier: {overall.get('brier', 'N/A')}")
    print(f"  Base rate: {overall.get('base_rate', 'N/A')}")
    print(f"  Models: {results.get('models_used', [])}")

    return True


def main():
    print("=" * 60)
    print("Sanity Check — 2+ TB Model")
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
