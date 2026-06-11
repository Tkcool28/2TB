#!/usr/bin/env python3
"""
dashboard/app.py
=================
Streamlit dashboard for 2+ Total Bases predictions.

Shows:
  - Today's slate with P(2+ TB) for each player (lineup spots 1-5)
  - Ranked table with color-coded probabilities
  - Game-by-game breakdown
  - Model confidence indicators

Usage:
    streamlit run dashboard/app.py --server.port 8502
"""

import json
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_data(ttl=300)
def load_predictions():
    """Load live predictions."""
    path = os.path.join(RESULTS_DIR, "live_predictions.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=3600)
def load_backtest():
    """Load backtest results."""
    path = os.path.join(RESULTS_DIR, "backtest_tb.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def load_training():
    """Load training results."""
    path = os.path.join(RESULTS_DIR, "training_tb_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    st.set_page_config(
        page_title="2+ TB Predictor",
        page_icon="⚾",
        layout="wide",
    )

    st.title("⚾ 2+ Total Bases Predictor")
    st.markdown("Predicts each player's probability of recording **2+ total bases** in today's games.")
    st.markdown("_*Lineup spots 1-5 only — where the best hitters bat._")

    # Load data
    predictions = load_predictions()
    backtest = load_backtest()
    training = load_training()

    # === Sidebar ===
    st.sidebar.header("⚙️ Settings")

    min_proba = st.sidebar.slider("Min P(2+ TB)", 0.0, 0.5, 0.0, 0.05)
    max_players = st.sidebar.slider("Max players to show", 10, 100, 50)

    if backtest:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Model Performance")
        overall = backtest.get("overall", {})
        st.sidebar.metric("AUC", f"{overall.get('auc', 0):.3f}")
        st.sidebar.metric("Brier Score", f"{overall.get('brier', 0):.3f}")
        st.sidebar.metric("Base Rate", f"{overall.get('base_rate', 0):.3f}")

    # === Main content ===
    if not predictions:
        st.warning("No predictions available. Run `python scripts/tb_predict_live.py` first.")

        # Show training results if available
        if training:
            st.subheader("📈 Training Results (Holdout)")
            col1, col2, col3 = st.columns(3)
            for model_name in ["logreg", "xgb"]:
                if model_name in training and "holdout" in training[model_name]:
                    metrics = training[model_name]["holdout"]
                    col1.metric(f"{model_name.upper()} AUC", f"{metrics.get('auc', 0):.3f}")
                    col2.metric(f"{model_name.upper()} Brier", f"{metrics.get('brier', 0):.3f}")
                    col3.metric(f"{model_name.upper()} Acc", f"{metrics.get('accuracy', 0):.3f}")
        return

    df = pd.DataFrame(predictions)

    # Filter
    df = df[df["predicted_proba_2tb"] >= min_proba]
    df = df.head(max_players)

    # === Top picks ===
    st.subheader("🔥 Top Picks Today")
    top10 = df.head(10)

    cols = st.columns(5)
    for i, (_, row) in enumerate(top10.iterrows()):
        col = cols[i % 5]
        proba = row["predicted_proba_2tb"]
        # Color coding
        if proba >= 0.5:
            color = "🟢"
        elif proba >= 0.35:
            color = "🟡"
        else:
            color = "🔴"

        col.metric(
            label=f"{color} {row['team']} #{row['lineup_position']}",
            value=f"{proba:.1%}",
            delta=f"vs {row['opponent']}",
        )

    # === Full table ===
    st.subheader("📋 Full Rankings")

    # Format for display
    display_df = df[["team", "opponent", "lineup_position", "is_home", "predicted_proba_2tb"]].copy()
    display_df.columns = ["Team", "Opp", "Spot", "Home", "P(2+ TB)"]
    display_df["Home"] = display_df["Home"].map({1: "✅", 0: "❌"})
    display_df["P(2+ TB)"] = display_df["P(2+ TB)"].apply(lambda x: f"{x:.1%}")

    # Color bar for probability
    def color_proba(val):
        proba = float(val.strip("%")) / 100
        if proba >= 0.5:
            return "background-color: #2ecc71; color: white"
        elif proba >= 0.35:
            return "background-color: #f39c12; color: white"
        elif proba >= 0.25:
            return "background-color: #e67e22; color: white"
        else:
            return "background-color: #e74c3c; color: white"

    styled = display_df.style.applymap(color_proba, subset=["P(2+ TB)"])
    st.dataframe(styled, use_container_width=True, height=600)

    # === By game ===
    st.subheader("🏟️ By Game")
    if "game_pk" in df.columns:
        games = df.groupby(["game_pk", "team", "opponent"]).first().reset_index()
        for _, game_row in games.iterrows():
            game_pk = game_row["game_pk"]
            game_df = df[df["game_pk"] == game_pk].sort_values("predicted_proba_2tb", ascending=False)
            if len(game_df) == 0:
                continue
            away = game_df[game_df["is_home"] == 0]["team"].iloc[0] if len(game_df[game_df["is_home"] == 0]) > 0 else "?"
            home = game_df[game_df["is_home"] == 1]["team"].iloc[0] if len(game_df[game_df["is_home"] == 1]) > 0 else "?"
            with st.expander(f"{away} @ {home}"):
                st.dataframe(
                    game_df[["team", "lineup_position", "predicted_proba_2tb"]].rename(
                        columns={"team": "Team", "lineup_position": "Spot", "predicted_proba_2tb": "P(2+ TB)"}
                    ),
                    use_container_width=True,
                )

    # === Footer ===
    st.markdown("---")
    st.caption("Model: LogReg + XGBoost ensemble | Data: MLB Stats API + Statcast | Updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
