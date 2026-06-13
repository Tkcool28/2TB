# 2TB — MLB 2+ Total Bases Prediction Model

Predicts the probability that a player records **2+ total bases** in a game.

## Current v2 model stack

The v2 stack uses **34 features** (deduped from the old 48-feature v1 schema) and a 3-model ensemble:

- **Logistic Regression** (`models/logreg_tb_v2.pkl`) — calibrated baseline
- **XGBoost** (`models/xgb_tb_v2.pkl`) — nonlinear interactions
- **LightGBM** (`models/lgbm_tb_v2.pkl`) — gradient-boosted, best single-model AUC
- **Scaler** (`models/scaler_tb_v2.pkl`) — StandardScaler fit on v2 training data

All models expect the same 34-feature vector. The **exact feature order** is defined in `data/processed/train_tb_v2.json` and validated by `scripts/sanity_check_tb.py`.

## v2 Performance Metrics

- **Ensemble AUC:** 0.5955
- **Top-1 Hit Rate:** 49.15%
- **Holdout Size:** 60,653 games (from 2025 season)

## v2 artifacts

| File | Purpose |
|------|---------|
| `models/logreg_tb_v2.pkl` | Logistic Regression model |
| `models/xgb_tb_v2.pkl` | XGBoost model |
| `models/lgbm_tb_v2.pkl` | LightGBM model |
| `models/scaler_tb_v2.pkl` | StandardScaler for v2 features |
| `results/training_tb_v2_results.json` | Training metrics (AUC, log-loss, etc.) |
| `results/backtest_tb_v2.json` | Backtest results on holdout seasons |

## v2 scripts

| Script | What it does |
|--------|-------------|
| `scripts/build_tb_features_v2.py` | Builds the 34-feature v2 dataset |
| `scripts/train_tb_models_v2.py` | Trains all three v2 models |
| `scripts/backtest_tb_v2.py` | Runs backtest on holdout data |
| `scripts/sanity_check_tb.py` | Validates feature schema and model outputs |

## Data

- **Game logs & Statcast:** 2022–2025
- **Location:** Raw and processed data reside on the VPS (not hosted on GitHub).
- **Historical lineups:** pulled from MLB Stats API v1.1

## Live predictions

`scripts/tb_predict_live.py` now loads the v2 models and expects 34 features.

**Important:** it still uses league-average defaults for player, pitcher, and context features until the separate live feature hydration work is implemented. Current live rankings are **shape-correct but not fully data-rich** — they will improve once live statcast and lineup data are wired in.

## Useful commands

```bash
# Validate v2 feature schema and model sanity
python3 scripts/sanity_check_tb.py

# Run live predictions (v2 models, league-average defaults)
python3 scripts/tb_predict_live.py

# Backtest v2 models on holdout seasons
python3 scripts/backtest_tb_v2.py
```

## Dashboard

Live at: https://totals.tkhermes.duckdns.org

## Daily Prediction Runner

The frozen 2TB model stack can now be invoked daily to generate player‑level hit predictions.

### Usage
```bash
# Run predictions for today (default)
python scripts/run_daily_predictions.py

# Run for a specific date (back‑fill / validation)
python scripts/run_daily_predictions.py --date 2025-06-01
```

*The script will:
- Determine the MLB slate for the requested date.
- Load the four frozen models (`logreg_tb_v2.pkl`, `xgb_tb_v2.pkl`, `lgbm_tb_v2.pkl`, `scaler_tb_v2.pkl`).
- Hydrate lineup, pitcher, and player features using the existing `tb_predict_live` pipeline.
- Produce an ensemble probability (average of the three model outputs).
- Write a sorted CSV of all predictions and a JSON summary.

### Output locations
- **Predictions CSV**: `predictions/<year>/<YYYY‑MM‑DD>_predictions.csv`
- **Summary JSON**: `predictions/<year>/<YYYY‑MM‑DD>_summary.json`
- **Run log**: `logs/prediction_runs/<YYYY‑MM‑DD>.log`

All files are version‑stable — the script never overwrites an existing file; it aborts with a clear error if a file for the same date already exists.

### Notes
- No model retraining or feature changes are performed — the pipeline is read‑only with respect to the 34‑feature schema.
- The script can be scheduled with `cron` or any task runner to run automatically each day.

---
