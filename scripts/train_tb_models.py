#!/usr/bin/env python3
"""
train_tb_models.py
===================
Train Logistic Regression and XGBoost on TB features.

Models:
  - Logistic Regression (calibrated, interpretable baseline)
  - XGBoost (nonlinear interactions)

Saves:
  - models/logreg_tb.pkl
  - models/xgb_tb.pkl
  - models/scaler_tb.pkl
  - results/training_tb_results.json

Usage:
    python3 train_tb_models.py
"""

import json
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def load_datasets():
    datasets = {}
    for name in ["train", "validate", "holdout"]:
        path = os.path.join(PROCESSED_DIR, f"{name}_tb.json")
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found — run build_tb_features.py first")
            continue
        with open(path) as f:
            datasets[name] = json.load(f)
        print(f"  Loaded {name}: {len(datasets[name]):,} rows")
    return datasets


def prep(data, feature_cols):
    X, y = [], []
    for row in data:
        x = [row.get(f, 0) for f in feature_cols]
        X.append(x)
        y.append(row["target_2tb"])
    return np.array(X), np.array(y)


def evaluate_model(model, X, y, name):
    """Evaluate a model and return metrics."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X)

    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = 0.0
    brier = brier_score_loss(y, proba)
    try:
        ll = log_loss(y, proba)
    except Exception:
        ll = 0.0

    print(f"  {name}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Brier:    {brier:.4f}")
    print(f"    LogLoss:  {ll:.4f}")
    print(f"    Base rate: {y.mean():.3f}")

    return {
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "log_loss": ll,
        "base_rate": float(y.mean()),
        "n_samples": len(y),
    }


def main():
    print("=" * 60)
    print("Train TB Models")
    print("=" * 60)

    datasets = load_datasets()
    if "train" not in datasets:
        print("No training data found!")
        return

    # Prepare data
    X_train, y_train = prep(datasets["train"], FEATURES)
    X_val, y_val = prep(datasets["validate"], FEATURES) if "validate" in datasets else (None, None)
    X_hold, y_hold = prep(datasets["holdout"], FEATURES) if "holdout" in datasets else (None, None)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_hold_scaled = scaler.transform(X_hold) if X_hold is not None else None

    results = {}

    # === Model A: Logistic Regression ===
    print("\n--- Logistic Regression ---")
    logreg = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    logreg.fit(X_train_scaled, y_train)

    # Calibrate
    logreg_cal = CalibratedClassifierCV(logreg, cv=5, method="isotonic")
    logreg_cal.fit(X_train_scaled, y_train)

    results["logreg"] = {}
    results["logreg"]["train"] = evaluate_model(logreg_cal, X_train_scaled, y_train, "Train")
    if X_val_scaled is not None:
        results["logreg"]["validate"] = evaluate_model(logreg_cal, X_val_scaled, y_val, "Validate")
    if X_hold_scaled is not None:
        results["logreg"]["holdout"] = evaluate_model(logreg_cal, X_hold_scaled, y_hold, "Holdout")

    # Feature importance (coefficients)
    coef_importance = dict(zip(FEATURES, logreg.coef_[0].tolist()))
    results["logreg"]["feature_importance"] = coef_importance

    # Save
    with open(os.path.join(MODELS_DIR, "logreg_tb.pkl"), "wb") as f:
        pickle.dump(logreg_cal, f)
    print("  Saved models/logreg_tb.pkl")

    # === Model B: XGBoost ===
    print("\n--- XGBoost ---")
    try:
        from xgboost import XGBClassifier

        # Calculate scale_pos_weight for class imbalance
        neg = sum(y_train == 0)
        pos = sum(y_train == 1)
        spw = neg / pos if pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        eval_set = [(X_train_scaled, y_train)]
        if X_val_scaled is not None:
            eval_set.append((X_val_scaled, y_val))

        xgb.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        results["xgb"] = {}
        results["xgb"]["train"] = evaluate_model(xgb, X_train_scaled, y_train, "Train")
        if X_val_scaled is not None:
            results["xgb"]["validate"] = evaluate_model(xgb, X_val_scaled, y_val, "Validate")
        if X_hold_scaled is not None:
            results["xgb"]["holdout"] = evaluate_model(xgb, X_hold_scaled, y_hold, "Holdout")

        # Feature importance
        xgb_importance = dict(zip(FEATURES, xgb.feature_importances_.tolist()))
        results["xgb"]["feature_importance"] = xgb_importance

        # Save
        with open(os.path.join(MODELS_DIR, "xgb_tb.pkl"), "wb") as f:
            pickle.dump(xgb, f)
        print("  Saved models/xgb_tb.pkl")

    except ImportError:
        print("  XGBoost not installed — skipping")
        results["xgb"] = {"error": "xgboost not installed"}

    # Save scaler
    with open(os.path.join(MODELS_DIR, "scaler_tb.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("  Saved models/scaler_tb.pkl")

    # Save results
    results["feature_names"] = FEATURES
    results["n_features"] = len(FEATURES)
    with open(os.path.join(RESULTS_DIR, "training_tb_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved results/training_tb_results.json")

    # Print top features
    print("\n--- Top 10 Features (LogReg |coefficient|) ---")
    sorted_coef = sorted(coef_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_coef[:10]:
        print(f"  {feat:40s} {val:+.4f}")

    if "xgb" in results and "feature_importance" in results["xgb"]:
        print("\n--- Top 10 Features (XGBoost importance) ---")
        sorted_xgb = sorted(results["xgb"]["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for feat, val in sorted_xgb[:10]:
            print(f"  {feat:40s} {val:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
