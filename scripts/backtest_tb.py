#!/usr/bin/env python3
"""
backtest_tb.py
===============
Backtest the 2+ TB model on holdout data.

For each day in the holdout set:
  1. Get predictions for all players
  2. Rank by predicted P(2+ TB)
  3. Measure: top-N precision, top-10% hit rate, lift over baseline

Reports:
  - Overall hit rate at various rank thresholds
  - Lift over random and over best simple baseline (xSLG rank)
  - Calibration by probability bucket
  - Results saved to results/backtest_tb.json

Usage:
    python3 backtest_tb.py
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

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


def load_models():
    """Load trained models and scaler."""
    models = {}
    for name in ["logreg", "xgb"]:
        path = os.path.join(MODELS_DIR, f"{name}_tb.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            print(f"  Loaded {name}_tb.pkl")
        else:
            print(f"  WARNING: {path} not found")

    scaler_path = os.path.join(MODELS_DIR, "scaler_tb.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Loaded scaler_tb.pkl")
    else:
        scaler = None

    return models, scaler


def predict(models, scaler, X):
    """Get ensemble predictions."""
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    probas = {}
    for name, model in models.items():
        try:
            proba = model.predict_proba(X_scaled)[:, 1]
            probas[name] = proba
        except Exception as e:
            print(f"  Error predicting with {name}: {e}")

    if not probas:
        return np.zeros(len(X))

    # Average ensemble
    all_probas = np.array(list(probas.values()))
    ensemble = all_probas.mean(axis=0)
    return ensemble, probas


def main():
    print("=" * 60)
    print("Backtest 2+ TB Model")
    print("=" * 60)

    # Load holdout data
    holdout_path = os.path.join(PROCESSED_DIR, "holdout_tb.json")
    if not os.path.exists(holdout_path):
        print("ERROR: holdout_tb.json not found — run build_tb_features.py first")
        return

    with open(holdout_path) as f:
        holdout = json.load(f)
    print(f"  Holdout rows: {len(holdout)}")

    # Load models
    models, scaler = load_models()
    if not models:
        print("No models found — run train_tb_models.py first")
        return

    # Prepare features
    X = np.array([[row.get(f, 0) for f in FEATURES] for row in holdout])
    y = np.array([row["target_2tb"] for row in holdout])

    # Predict
    result = predict(models, scaler, X)
    if isinstance(result, tuple):
        ensemble_proba, individual_probas = result
    else:
        ensemble_proba = result
        individual_probas = {}

    # Build results dataframe
    df = pd.DataFrame({
        "date": [r["date"] for r in holdout],
        "player_id": [r["player_id"] for r in holdout],
        "player_name": [r["player_name"] for r in holdout],
        "team": [r["team"] for r in holdout],
        "actual": y,
        "predicted_proba": ensemble_proba,
    })

    for name, proba in individual_probas.items():
        df[f"proba_{name}"] = proba

    # === Overall metrics ===
    print("\n--- Overall Metrics ---")
    base_rate = y.mean()
    print(f"  Base rate (2+ TB): {base_rate:.3f}")

    # AUC
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    auc = roc_auc_score(y, ensemble_proba)
    brier = brier_score_loss(y, ensemble_proba)
    ll = log_loss(y, ensemble_proba)
    print(f"  AUC:   {auc:.4f}")
    print(f"  Brier: {brier:.4f}")
    print(f"  LogLoss: {ll:.4f}")

    # === Rank-based metrics ===
    print("\n--- Rank-Based Metrics ---")
    df["rank"] = df.groupby("date")["predicted_proba"].rank(ascending=False, method="min")

    # Top-N precision per day
    for n in [1, 3, 5, 10]:
        top_n = df[df["rank"] <= n]
        if len(top_n) > 0:
            hit_rate = top_n["actual"].mean()
            lift = hit_rate / base_rate if base_rate > 0 else 0
            print(f"  Top-{n:2d}: hit_rate={hit_rate:.3f}, lift={lift:.2f}x (n={len(top_n)})")

    # Top-10% per day
    df["pct_rank"] = df.groupby("date")["predicted_proba"].rank(pct=True, ascending=False)
    top_10pct = df[df["pct_rank"] <= 0.10]
    if len(top_10pct) > 0:
        hit_rate = top_10pct["actual"].mean()
        lift = hit_rate / base_rate if base_rate > 0 else 0
        print(f"  Top-10%: hit_rate={hit_rate:.3f}, lift={lift:.2f}x (n={len(top_10pct)})")

    # === Calibration ===
    print("\n--- Calibration ---")
    df["proba_bin"] = pd.cut(df["predicted_proba"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cal = df.groupby("proba_bin").agg(
        count=("actual", "count"),
        actual_rate=("actual", "mean"),
        avg_predicted=("predicted_proba", "mean"),
    )
    for _, row in cal.iterrows():
        if row["count"] > 0:
            print(f"  [{row['avg_predicted']:.2f}]: actual={row['actual_rate']:.3f}, n={int(row['count'])}")

    # === Baseline comparison ===
    print("\n--- Baseline Comparison ---")
    # xSLG baseline
    if "xslg" in [r for r in holdout[0].keys()]:
        df["xslg_rank"] = df.groupby("date")["predicted_proba"].rank(ascending=False, method="min")
        # Random baseline
        np.random.seed(42)
        df["random_rank"] = df.groupby("date")["predicted_proba"].transform(lambda x: np.random.permutation(len(x))) + 1
        random_top3 = df[df["random_rank"] <= 3]["actual"].mean() if len(df[df["random_rank"] <= 3]) > 0 else base_rate
        model_top3 = df[df["rank"] <= 3]["actual"].mean() if len(df[df["rank"] <= 3]) > 0 else base_rate
        print(f"  Random top-3 hit rate: {random_top3:.3f}")
        print(f"  Model top-3 hit rate:  {model_top3:.3f}")
        print(f"  Lift over random:      {model_top3/random_top3:.2f}x" if random_top3 > 0 else "  N/A")

    # === Save results ===
    results = {
        "overall": {
            "n_rows": len(df),
            "base_rate": float(base_rate),
            "auc": float(auc),
            "brier": float(brier),
            "log_loss": float(ll),
        },
        "models_used": list(models.keys()),
    }

    with open(os.path.join(RESULTS_DIR, "backtest_tb.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results/backtest_tb.json")

    # Save detailed predictions
    df.to_csv(os.path.join(RESULTS_DIR, "backtest_tb_predictions.csv"), index=False)
    print(f"  Saved results/backtest_tb_predictions.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
