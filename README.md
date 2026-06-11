# 2TB — MLB 2+ Total Bases Prediction Model

Predicts the probability that a player records **2+ total bases** in a game.

## Structure
- `scripts/` — Feature engineering, training, backtesting
- `data/raw/` — Raw data (game logs, statcast, schedules)
- `data/processed/` — Feature datasets (train/val/holdout)
- `models/` — Trained model artifacts (.pkl)
- `results/` — Training metrics, backtest results
- `dashboard/` — Streamlit dashboard

## Models
- Logistic Regression (calibrated baseline)
- XGBoost (nonlinear interactions)

## Data
- Game logs: 2022–2025
- Statcast (batters & pitchers): 2022–2025
- Historical lineups: pulled from MLB Stats API v1.1

## Dashboard
Live at: https://totals.tkhermes.duckdns.org
