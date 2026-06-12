# 2TB — MLB 2+ Total Bases Prediction Model

Predicts the probability that a player records **2+ total bases** in a game.

## Current v2 model stack

The v2 stack uses **34 features** (deduped from the old 48-feature v1 schema) and a 3-model ensemble:

- **Logistic Regression** (`models/logreg_tb_v2.pkl`) — calibrated baseline
- **XGBoost** (`models/xgb_tb_v2.pkl`) — nonlinear interactions
- **LightGBM** (`models/lgbm_tb_v2.pkl`) — gradient-boosted, best single-model AUC
- **Scaler** (`models/scaler_tb_v2.pkl`) — StandardScaler fit on v2 training data

All models expect the same 34-feature vector. The **exact feature order** is defined in `data/processed/train_tb_v2.json` and validated by `scripts/sanity_check_tb.py`.

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

- Game logs: 2022–2025
- Statcast (batters & pitchers): 2022–2025
- Historical lineups: pulled from MLB Stats API v1.1

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
