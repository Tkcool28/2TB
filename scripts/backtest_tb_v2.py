#!/usr/bin/env python3
"""
backtest_tb_v2.py
=================
Backtest 2+ TB model on v2 holdout data.

Usage:
    python3 backtest_tb_v2.py
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EXCLUDE_COLS = {"target_2tb", "total_bases", "game_pk", "player_id", "game_date", "season", "team"}


def main():
    print("=" * 60)
    print("Backtest 2+ TB Model v2")
    print("=" * 60)

    # Load holdout
    with open(os.path.join(PROCESSED_DIR, "holdout_tb_v2.json")) as f:
        holdout = pd.DataFrame(json.load(f))

    feat_cols = [c for c in holdout.columns if c not in EXCLUDE_COLS]
    X = holdout[feat_cols].fillna(0).values
    y = holdout["target_2tb"].values

    print(f"  Holdout rows: {len(holdout)}")
    print(f"  Features: {len(feat_cols)}")

    # Load models
    with open(os.path.join(MODELS_DIR, "scaler_tb_v2.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "logreg_tb_v2.pkl"), "rb") as f:
        logreg = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "xgb_tb_v2.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "lgbm_tb_v2.pkl"), "rb") as f:
        lgbm_model = pickle.load(f)

    X_sc = scaler.transform(X)

    # Predictions
    pred_lr = logreg.predict_proba(X_sc)[:, 1]
    pred_xgb = xgb_model.predict_proba(X)[:, 1]
    pred_lgbm = lgbm_model.predict_proba(X)[:, 1]
    pred_ensemble = (pred_lr + pred_xgb + pred_lgbm) / 3.0

    # Overall metrics
    print("\n--- Overall Metrics ---")
    for name, pred in [("LogReg", pred_lr), ("XGBoost", pred_xgb), ("LightGBM", pred_lgbm), ("Ensemble", pred_ensemble)]:
        auc = roc_auc_score(y, pred)
        brier = brier_score_loss(y, pred)
        print(f"  {name:12s}: AUC={auc:.4f}  Brier={brier:.4f}")

    # Rank-based metrics (ensemble)
    print("\n--- Rank-Based Metrics (Ensemble) ---")
    holdout = holdout.copy()
    holdout["pred"] = pred_ensemble

    # Group by date for ranking
    dates = holdout["game_date"].unique()
    top_rates = {}
    for top_n in [1, 3, 5, 10]:
        hits = 0
        total = 0
        for date in dates:
            day = holdout[holdout["game_date"] == date].sort_values("pred", ascending=False)
            if len(day) >= top_n:
                top = day.head(top_n)
                hits += top["target_2tb"].sum()
                total += top_n
        rate = hits / total if total > 0 else 0
        base = y.mean()
        lift = rate / base if base > 0 else 0
        top_rates[top_n] = (rate, lift, total // top_n)
        print(f"  Top-{top_n:>2d}: hit_rate={rate:.3f}, lift={lift:.2f}x (n={total // top_n})")

    # Top-10%
    hits = 0
    total = 0
    for date in dates:
        day = holdout[holdout["game_date"] == date].sort_values("pred", ascending=False)
        n = max(1, int(len(day) * 0.1))
        top = day.head(n)
        hits += top["target_2tb"].sum()
        total += n
    rate = hits / total if total > 0 else 0
    lift = rate / y.mean()
    print(f"  Top-10%: hit_rate={rate:.3f}, lift={lift:.2f}x")

    # Calibration
    print("\n--- Calibration (Ensemble) ---")
    holdout["bucket"] = pd.cut(holdout["pred"], bins=10)
    for bucket, group in holdout.groupby("bucket", observed=True):
        actual = group["target_2tb"].mean()
        n = len(group)
        print(f"  [{bucket.left:.2f}-{bucket.right:.2f}]: actual={actual:.3f}, n={n}")

    # Baseline comparison
    print("\n--- Baseline Comparison ---")
    random_top3 = y.mean()
    model_top3 = top_rates[3][0]
    print(f"  Random top-3 hit rate: {random_top3:.3f}")
    print(f"  Model top-3 hit rate:  {model_top3:.3f}")
    print(f"  Lift over random:      {model_top3 / random_top3:.2f}x")

    # Save
    results = {
        "holdout_size": len(holdout),
        "base_rate": float(y.mean()),
        "models": {
            "logreg": {"auc": roc_auc_score(y, pred_lr), "brier": brier_score_loss(y, pred_lr)},
            "xgb": {"auc": roc_auc_score(y, pred_xgb), "brier": brier_score_loss(y, pred_xgb)},
            "lgbm": {"auc": roc_auc_score(y, pred_lgbm), "brier": brier_score_loss(y, pred_lgbm)},
            "ensemble": {"auc": roc_auc_score(y, pred_ensemble), "brier": brier_score_loss(y, pred_ensemble)},
        },
        "rank_metrics": {f"top_{k}": {"rate": v[0], "lift": v[1], "n": v[2]} for k, v in top_rates.items()},
    }

    with open(os.path.join(RESULTS_DIR, "backtest_tb_v2.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results/backtest_tb_v2.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
