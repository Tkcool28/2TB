# MODEL_MANIFEST.md

## Purpose

This manifest documents the two model versions (v1 legacy, v2 current) for the
2TB total-bases model. **It exists to prevent mixing v1 and v2 feature schemas
and model files, which will crash or silently produce garbage.**

---

## v1 — Legacy (do not use for new live predictions)

- **Feature schema:** 48 features (old, deduplicated in v2 to 34).
- **Model files:**
  - `models/logreg_tb.pkl`
  - `models/xgb_tb.pkl`
  - `models/scaler_tb.pkl`
- **Training script:** `scripts/train_tb_models.py`
- **Live script behavior (v1 era):** v1 features + v1 models.
- **Status:** Legacy. Kept for reference only. Do not use in new live prediction
  pipelines.

---

## v2 — Current

- **Feature schema:** 34 features (deduplicated from v1's 48).
- **Model files:**
  - `models/logreg_tb_v2.pkl`
  - `models/xgb_tb_v2.pkl`
  - `models/lgbm_tb_v2.pkl` (LightGBM is new in v2)
  - `models/scaler_tb_v2.pkl`
- **Training script:** `scripts/train_tb_models_v2.py`
- **Feature order:** Must match exactly what is defined in
  `data/processed/train_tb_v2.json`. Order is validated by
  `scripts/sanity_check_tb.py`.
- **Results:**
  - `results/training_tb_v2_results.json`
  - `results/backtest_tb_v2.json`
- **Performance:** ~0.596 AUC, ~49.1% top-1 hit rate on 2025 holdout.

---

## Do not mix v1 and v2

- v1 models expect **48 features** in the old order.
- v2 models expect **34 features** in the deduplicated v2 order.
- Loading v2 models with v1 features (or vice versa) will either crash
  immediately or silently produce wrong predictions.
- Always pair: v2 schema → v2 scaler → v2 models. v1 → v1.

---

## Current live prediction status

- `scripts/tb_predict_live.py` now targets **v2** models and v2 feature schema.
- It currently uses **league-average defaults** for real player/pitcher/context
  features until live hydration is implemented.
- Status: **shape-correct** (produces valid predictions), but not yet
  **data-rich** (placeholder values instead of real rolling stats, statcast,
  platoon splits, park factors, etc.).

---

## Owner / next step

Next separate work is **live feature hydration**: integrating real rolling
stats, statcast data, pitcher stats, platoon/park/context features so that
`tb_predict_live.py` moves from league-average defaults to real per-player
and per-game inputs.
