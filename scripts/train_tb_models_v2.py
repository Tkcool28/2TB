#!/usr/bin/env python3
"""
train_tb_models_v2.py
=====================
Train 2+ TB models on v2 features.
Adds LightGBM to the ensemble (LogReg + XGBoost + LGBM).

Usage:
    python3 train_tb_models_v2.py
"""

import json
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EXCLUDE_COLS = {"target_2tb", "total_bases", "game_pk", "player_id", "game_date", "season", "team"}


def load_data():
    """Load v2 feature datasets."""
    with open(os.path.join(PROCESSED_DIR, "train_tb_v2.json")) as f:
        train = pd.DataFrame(json.load(f))
    with open(os.path.join(PROCESSED_DIR, "validate_tb_v2.json")) as f:
        val = pd.DataFrame(json.load(f))
    with open(os.path.join(PROCESSED_DIR, "holdout_tb_v2.json")) as f:
        holdout = pd.DataFrame(json.load(f))
    return train, val, holdout


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def evaluate(model_name, y_true, y_pred, split_name):
    """Print evaluation metrics."""
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    ll = log_loss(y_true, y_pred)
    acc = ((y_pred >= 0.5).astype(int) == y_true).mean()
    print(f"  {split_name}:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    AUC:      {auc:.4f}")
    print(f"    Brier:    {brier:.4f}")
    print(f"    LogLoss:  {ll:.4f}")
    print(f"    Base rate: {y_true.mean():.3f}")
    return {"accuracy": acc, "auc": auc, "brier": brier, "logloss": ll, "base_rate": y_true.mean()}


def main():
    print("=" * 60)
    print("Train TB Models v2")
    print("=" * 60)

    # Load data
    print("\nLoading v2 data...")
    train, val, holdout = load_data()
    feat_cols = get_feature_cols(train)
    print(f"  Features: {len(feat_cols)}")
    print(f"  Train: {len(train):,}, Val: {len(val):,}, Holdout: {len(holdout):,}")

    X_train = train[feat_cols].fillna(0).values
    y_train = train["target_2tb"].values
    X_val = val[feat_cols].fillna(0).values
    y_val = val["target_2tb"].values
    X_hold = holdout[feat_cols].fillna(0).values
    y_hold = holdout["target_2tb"].values

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_hold_sc = scaler.transform(X_hold)

    results = {}

    # ─── Logistic Regression ───
    print("\n--- Logistic Regression ---")
    logreg = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    logreg.fit(X_train_sc, y_train)
    results["logreg"] = {}
    for split, Xs, ys in [("Train", X_train_sc, y_train), ("Validate", X_val_sc, y_val), ("Holdout", X_hold_sc, y_hold)]:
        pred = logreg.predict_proba(Xs)[:, 1]
        results["logreg"][split.lower()] = evaluate("LogReg", ys, pred, split)
    with open(os.path.join(MODELS_DIR, "logreg_tb_v2.pkl"), "wb") as f:
        pickle.dump(logreg, f)
    print(f"  Saved models/logreg_tb_v2.pkl")

    # Top features
    coef = pd.Series(np.abs(logreg.coef_[0]), index=feat_cols).sort_values(ascending=False)
    print("\n  Top 10 LogReg features (|coefficient|):")
    for feat, c in coef.head(10).items():
        sign = "+" if logreg.coef_[0][feat_cols.index(feat)] > 0 else "-"
        print(f"    {sign} {feat:40s} {c:.4f}")

    # ─── XGBoost ───
    print("\n--- XGBoost ---")
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=42,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        results["xgb"] = {}
        for split, Xs, ys in [("Train", X_train, y_train), ("Validate", X_val, y_val), ("Holdout", X_hold, y_hold)]:
            pred = xgb_model.predict_proba(Xs)[:, 1]
            results["xgb"][split.lower()] = evaluate("XGBoost", ys, pred, split)
        with open(os.path.join(MODELS_DIR, "xgb_tb_v2.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
        print(f"  Saved models/xgb_tb_v2.pkl")

        # Feature importance
        imp = pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("\n  Top 10 XGBoost features:")
        for feat, c in imp.head(10).items():
            print(f"    {feat:40s} {c:.4f}")
    except ImportError:
        print("  XGBoost not available, skipping")

    # ─── LightGBM ───
    print("\n--- LightGBM ---")
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        results["lgbm"] = {}
        for split, Xs, ys in [("Train", X_train, y_train), ("Validate", X_val, y_val), ("Holdout", X_hold, y_hold)]:
            pred = lgb_model.predict_proba(Xs)[:, 1]
            results["lgbm"][split.lower()] = evaluate("LightGBM", ys, pred, split)
        with open(os.path.join(MODELS_DIR, "lgbm_tb_v2.pkl"), "wb") as f:
            pickle.dump(lgb_model, f)
        print(f"  Saved models/lgbm_tb_v2.pkl")

        # Feature importance
        imp = pd.Series(lgb_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print("\n  Top 10 LightGBM features:")
        for feat, c in imp.head(10).items():
            print(f"    {feat:40s} {c:.4f}")
    except ImportError:
        print("  LightGBM not available, skipping")
        print("  Install with: pip install lightgbm")

    # ─── Ensemble (average of available models) ───
    print("\n--- Ensemble (Average) ---")
    ensemble_preds = {}
    for split, Xs, Xs_sc, ys in [
        ("Train", X_train, X_train_sc, y_train),
        ("Validate", X_val, X_val_sc, y_val),
        ("Holdout", X_hold, X_hold_sc, y_hold),
    ]:
        preds = []
        # LogReg prediction
        preds.append(logreg.predict_proba(Xs_sc)[:, 1])
        # XGBoost prediction
        if "xgb" in results:
            preds.append(xgb_model.predict_proba(Xs)[:, 1])
        # LightGBM prediction
        if "lgbm" in results:
            preds.append(lgb_model.predict_proba(Xs)[:, 1])
        # Average
        avg_pred = np.mean(preds, axis=0)
        ensemble_preds[split.lower()] = avg_pred
        evaluate("Ensemble", ys, avg_pred, split)

    # ─── Summary ───
    print(f"\n{'=' * 60}")
    print("Summary — Holdout AUC")
    print(f"{'=' * 60}")
    for model_name in ["logreg", "xgb", "lgbm"]:
        if model_name in results:
            auc = results[model_name]["holdout"]["auc"]
            print(f"  {model_name:12s}: {auc:.4f}")
    if ensemble_preds:
        ens_auc = roc_auc_score(y_hold, ensemble_preds["holdout"])
        print(f"  {'ensemble':12s}: {ens_auc:.4f}")

    # Save results
    with open(os.path.join(RESULTS_DIR, "training_tb_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(MODELS_DIR, "scaler_tb_v2.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved results/training_tb_v2_results.json")
    print(f"Saved models/scaler_tb_v2.pkl")
    print("\nDone!")


if __name__ == "__main__":
    main()
