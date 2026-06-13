#!/usr/bin/env python3
"""
dashboard/app.py
================
Standalone Streamlit dashboard for the 2TB model.

Reads saved files only — does NOT call model inference.
  - predictions/<year>/<yyyymmdd>_predictions.csv   (today's slate)
  - predictions/<year>/<yyyymmdd>_summary.json       (today's metadata)
  - results/<year>/<yyyymmdd>_results.csv            (graded rows)
  - results/<year>/<yyyymmdd>_results_summary.json   (daily metrics)

Usage:
    streamlit run dashboard/app.py
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR  = REPO_ROOT / "predictions"
RESULT_DIR = REPO_ROOT / "results"

MT = ZoneInfo("America/Denver")

# Module-level logger for data-loader warnings shown in the UI
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_denver() -> str:
    """Return today's date string in America/Denver timezone."""
    return datetime.now(MT).strftime("%Y-%m-%d")


def _date_compact(d: str) -> str:
    return d.replace("-", "")


def _collect_warning(container: list, path: Path, error: Exception):
    """Record a warning about a corrupt/unreadable file."""
    msg = f"{path}: {error}"
    container.append(msg)
    warnings.warn(msg, stacklevel=2)
    _log.warning(msg)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_todays_predictions(date_str: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load predictions CSV for a given date.

    Returns (df, warnings_list).  Defaults to the newest available date.
    """
    warn: list[str] = []
    if date_str is None:
        date_str = _today_denver()
    year = date_str[:4]
    compact = _date_compact(date_str)
    csv_path = PRED_DIR / year / f"{compact}_predictions.csv"
    if not csv_path.is_file():
        return pd.DataFrame(), warn
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        _collect_warning(warn, csv_path, e)
        return pd.DataFrame(), warn
    return df, warn


@st.cache_data(ttl=300)
def load_todays_summary(date_str: str | None = None) -> tuple[dict, list[str]]:
    """Load prediction summary JSON for a given date.

    Returns (summary_dict, warnings_list).
    """
    warn: list[str] = []
    if date_str is None:
        date_str = _today_denver()
    year = date_str[:4]
    compact = _date_compact(date_str)
    json_path = PRED_DIR / year / f"{compact}_summary.json"
    if not json_path.is_file():
        return {}, warn
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data, warn
    except Exception as e:
        _collect_warning(warn, json_path, e)
        return {}, warn


@st.cache_data(ttl=300)
def load_all_result_summaries() -> tuple[pd.DataFrame, list[str]]:
    """Load every results_summary.json into a DataFrame keyed by date.

    Returns (df, warnings_list).  Malformed files are skipped with warnings.
    """
    rows: list[dict] = []
    warn: list[str] = []
    for path in sorted(RESULT_DIR.rglob("*_results_summary.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_file"] = str(path)
            rows.append(data)
        except Exception as e:
            _collect_warning(warn, Path(path), e)
    if not rows:
        return pd.DataFrame(), warn
    df = pd.DataFrame(rows)
    col_map = {
        "game_date": "date",
        "total_predictions": "total",
        "graded_predictions": "graded",
        "ungraded_predictions": "ungraded",
        "total_hits_2tb": "hits_2tb",
        "hit_rate": "hit_rate",
        "top_10_hit_rate": "top10_hr",
        "top_20_hit_rate": "top20_hr",
        "top_prediction_hit": "top_prediction_hit",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df, warn


def discover_prediction_dates() -> list[str]:
    """Return sorted list of available prediction date strings (YYYY-MM-DD)."""
    dates: list[str] = []
    for csv_file in sorted(PRED_DIR.rglob("*_predictions.csv")):
        stem = csv_file.stem  # "20250601_predictions"
        compact = stem.replace("_predictions", "")
        if len(compact) == 8 and compact.isdigit():
            dates.append(f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}")
    return dates


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _weighted_agg(window: pd.DataFrame) -> dict:
    """Compute weighted performance metrics for a window of result summaries.

    Hit rate = sum(hits_2tb) / sum(graded_predictions)  — weighted, not averaged.
    Top-10/20: averaged over daily rates (no numerator/denominator available).
    """
    if window.empty:
        return {}
    graded = window["graded"].sum() if "graded" in window.columns else 0
    hits  = window["hits_2tb"].sum() if "hits_2tb" in window.columns else 0

    row: dict = {
        "Days Tracked": len(window),
        "Total Graded": int(graded),
    }

    # Weighted hit rate
    if graded:
        row["Hit Rate"] = hits / graded
    else:
        row["Hit Rate"] = None

    # Top-10 / Top-20: we only have daily rates, not per-day eligible-row counts,
    # so we report the average of daily rates and label them accordingly.
    if "top10_hr" in window.columns and not window["top10_hr"].isna().all():
        row["Avg Top-10 HR"] = window["top10_hr"].mean()
    else:
        row["Avg Top-10 HR"] = None

    if "top20_hr" in window.columns and not window["top20_hr"].isna().all():
        row["Avg Top-20 HR"] = window["top20_hr"].mean()
    else:
        row["Avg Top-20 HR"] = None

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="2TB Dashboard",
        page_icon="⚾",
        layout="wide",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )

    st.title("⚾ 2+ Total Bases — Model Dashboard")
    st.caption(
        "Reads saved prediction and result files. "
        "Run the daily workflow to generate fresh data."
    )

    # ------------------------------------------------------------------
    # Sidebar — date picker
    # ------------------------------------------------------------------
    st.sidebar.header("⚙️ Settings")

    pred_dates = discover_prediction_dates()

    # Default: newest available date (not necessarily today)
    default_date = pred_dates[-1] if pred_dates else _today_denver()

    selected_date = st.sidebar.selectbox(
        "Prediction date",
        options=pred_dates if pred_dates else [_today_denver()],
        index=len(pred_dates) - 1 if pred_dates else 0,
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    all_warnings: list[str] = []

    pred_df, w1 = load_todays_predictions(selected_date)
    all_warnings.extend(w1)

    pred_sum, w2 = load_todays_summary(selected_date)
    all_warnings.extend(w2)

    results_df, w3 = load_all_result_summaries()
    all_warnings.extend(w3)

    # Show file-level warnings in sidebar
    if all_warnings:
        st.sidebar.warning("Data warnings detected — see footer for details.")

    # ------------------------------------------------------------------
    # Detect player_name column
    # ------------------------------------------------------------------
    has_player_name = "player_name" in pred_df.columns if not pred_df.empty else False

    # ------------------------------------------------------------------
    # Tab layout
    # ------------------------------------------------------------------
    tab_pred, tab_results, tab_history = st.tabs([
        "📋 Today's Predictions",
        "✅ Latest Results",
        "📈 Recent Performance",
    ])

    # ================================================================
    # TAB 1 — Today's Predictions
    # ================================================================
    with tab_pred:
        st.header(f"Predictions — {selected_date}")

        if pred_sum:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Games", pred_sum.get("total_games", "—"))
            c2.metric("Players", pred_sum.get("total_players", "—"))
            top_pred = pred_sum.get("top_prediction") or {}
            c3.metric(
                "Top Player Prob",
                f"{top_pred.get('predicted_proba_2tb', 0):.1%}"
                if top_pred.get("predicted_proba_2tb") is not None
                else "—",
            )
            c4.metric("Avg Models", f"{top_pred.get('model_count', '—')}")
            if top_pred:
                name_str = ""
                if top_pred.get("player_name"):
                    name_str = f" ({top_pred['player_name']})"
                st.markdown(
                    f"**Best pick:** Player {top_pred.get('player_id', '?')}{name_str} "
                    f"({top_pred.get('team', '?')}) — "
                    f"vs {top_pred.get('opponent', '?')} "
                    f"batting #{top_pred.get('lineup_position', '?')}"
                )

        if pred_df.empty:
            st.warning("No predictions found for this date.")
        else:
            # Rank column
            pred_df = pred_df.sort_values("predicted_proba_2tb", ascending=False).reset_index(drop=True)
            pred_df.index += 1  # 1-based ranking
            pred_df.index.name = "Rank"

            # Build display columns — player_name first if available
            base_cols = ["player_id", "team", "opponent", "lineup_position",
                         "is_home", "predicted_proba_2tb", "model_count"]
            if has_player_name:
                # Insert player_name right after Rank (i.e. first)
                display_cols = ["player_name"] + base_cols
            else:
                display_cols = base_cols

            avail_cols = [c for c in display_cols if c in pred_df.columns]
            disp = pred_df[avail_cols].copy()

            rename_map = {
                "player_name": "Player",
                "player_id": "Player ID",
                "team": "Team",
                "opponent": "Opp",
                "lineup_position": "Spot",
                "is_home": "Home",
                "predicted_proba_2tb": "P(2+ TB)",
                "model_count": "# Models",
            }
            disp = disp.rename(columns={k: v for k, v in rename_map.items() if k in disp.columns})

            # Format probability
            if "P(2+ TB)" in disp.columns:
                disp["P(2+ TB)"] = disp["P(2+ TB)"].apply(lambda x: f"{float(x):.1%}")

            if "Home" in disp.columns:
                disp["Home"] = disp["Home"].map({1: "✅", 0: "❌"})

            # Bar chart of top 20
            chart_df = pred_df.head(20).copy()
            if has_player_name:
                chart_df["label"] = chart_df["player_name"].astype(str)
            else:
                chart_df["label"] = (
                    chart_df.get("team", "").astype(str)
                    + " #"
                    + chart_df.get("lineup_position", "").astype(str)
                )
            st.bar_chart(
                chart_df.set_index("label")["predicted_proba_2tb"],
                use_container_width=True,
            )

            st.dataframe(disp, use_container_width=True, height=500)

            # Download
            csv_data = pred_df.to_csv(index=True)
            st.download_button("Download CSV", csv_data, f"{selected_date}_predictions.csv")

        # Player name limitation note
        if not has_player_name and not pred_df.empty:
            st.info(
                "💡 **Player names not shown** — the current prediction CSV schema does not "
                "include a `player_name` column. To add names, the upstream prediction runner "
                "(`scripts/run_daily_predictions.py`) would need to join player IDs to names "
                "via the MLB Stats API `/people/{id}` endpoint or a local lookup table."
            )

    # ================================================================
    # TAB 2 — Latest Results
    # ================================================================
    with tab_results:
        st.header("Latest Graded Results")

        if results_df.empty:
            st.info("No graded results yet. Run the grader after games are played.")
        else:
            latest = results_df.iloc[-1]
            st.subheader(f"Date: {latest.get('date', '—')}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Predictions", latest.get("total", "—"))
            c2.metric("Graded", latest.get("graded", "—"))
            hr = latest.get("hit_rate")
            c3.metric("Hit Rate", f"{hr:.1%}" if pd.notna(hr) else "—")
            top10 = latest.get("top10_hr")
            c4.metric("Top-10 Hit Rate", f"{top10:.1%}" if pd.notna(top10) else "—")

            c5, c6 = st.columns(2)
            top20 = latest.get("top20_hr")
            c5.metric("Top-20 Hit Rate", f"{top20:.1%}" if pd.notna(top20) else "—")
            top_hit = latest.get("top_prediction_hit")
            c6.metric("Top Prediction Hit", "✅" if top_hit is True else ("❌" if top_hit is False else "—"))

    # ================================================================
    # TAB 3 — Recent Performance
    # ================================================================
    with tab_history:
        st.header("Historical Performance")

        if results_df.empty:
            st.info("Not enough data yet.")
        else:
            if "date" not in results_df.columns or results_df["date"].isna().all():
                st.warning("Results found but no valid dates parsed.")
                st.dataframe(results_df)
                return

            max_date = results_df["date"].max()

            def _window(days: int) -> pd.DataFrame:
                cutoff = max_date - pd.Timedelta(days=days)
                return results_df[results_df["date"] >= cutoff]

            w7  = _window(7)
            w30 = _window(30)

            summary_rows = []
            for label, w in [("Last 7 Days", w7), ("Last 30 Days", w30), ("Overall", results_df)]:
                row = _weighted_agg(window=w)
                if row:
                    row["Period"] = label
                    summary_rows.append(row)

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows).set_index("Period")

                # Format for display
                display_summary = summary_df.copy()
                for col in ("Hit Rate", "Avg Top-10 HR", "Avg Top-20 HR"):
                    if col in display_summary.columns:
                        display_summary[col] = display_summary[col].apply(
                            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                        )
                st.table(display_summary)

            # Note about Top-10/20 aggregation
            st.caption(
                "💡 **Hit Rate** is weighted: sum(hits) / sum(graded). "
                "**Top-10 / Top-20 HR** are averages of daily rates (per-day "
                "eligible-row counts are not available in the current summary schema)."
            )

            # Line chart of hit rate over time
            chart_cols = ["date"]
            for c in ["hit_rate", "top10_hr", "top20_hr"]:
                if c in results_df.columns:
                    chart_cols.append(c)
            chart_df = results_df[chart_cols].dropna(subset=["date"]).set_index("date")
            if not chart_df.empty:
                chart_rename = {
                    "hit_rate": "Hit Rate",
                    "top10_hr": "Top-10 HR",
                    "top20_hr": "Top-20 HR",
                }
                chart_df = chart_df.rename(columns={k: v for k, v in chart_rename.items() if k in chart_df.columns})
                st.line_chart(chart_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")

    # Show data warnings in footer
    if all_warnings:
        st.subheader("⚠️ Data Warnings")
        for w in all_warnings:
            st.warning(w)

    now_str = datetime.now(MT).strftime("%Y-%m-%d %H:%M %Z")
    st.caption(
        f"2TB Model Dashboard v2 | LogReg + XGBoost + LightGBM ensemble | "
        f"Updated: {now_str}"
    )


if __name__ == "__main__":
    main()
