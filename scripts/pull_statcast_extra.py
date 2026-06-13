#!/usr/bin/env python3
"""
pull_statcast_extra.py
======================
Pull additional Statcast features: launch_angle, sprint_speed.
Uses the MLB Stats API leaderboards endpoint.

Usage:
    python3 pull_statcast_extra.py
"""

import json
import os
import time
import urllib.request
import urllib.error
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SEASONS = [2021, 2022, 2023, 2024]  # Need 2021 for 2022's "prior year"

def fetch_statcast_stat(stat_type, season, min_pa=100):
    """
    Fetch Statcast leaderboard data.
    stat_type: 'launch_angle', 'sprint_speed'
    """
    # Try the MLB stats API leaderboard endpoint
    if stat_type == "launch_angle":
        # Launch angle from statcast
        url = (
            f"https://statsapi.mlb.com/api/v1/leaders/stats"
            f"?leaderCategories=launchAngle"
            f"&statGroup=hitting"
            f"&season={season}"
        )
    elif stat_type == "sprint_speed":
        url = (
            f"https://statsapi.mlb.com/api/v1/leaders/stats"
            f"?leaderCategories=sprintSpeed"
            f"&statGroup=hitting"
            f"&season={season}"
        )
    else:
        return {}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        print(f"  {stat_type} {season}: {data.keys()}")
        return data
    except Exception as e:
        print(f"  {stat_type} {season}: ERROR - {e}")
        {}


def fetch_statcast_player_stats(player_id, season):
    """Fetch individual player statcast stats."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        f"?hydrate=stats(group=hitting,type=season,season={season})"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except Exception:
        return None


def pull_from_csv_fallback():
    """
    If we can't get launch_angle/sprint_speed from API,
    derive proxy features from existing data.
    """
    print("\nGenerating proxy features from existing data...")

    # Load batter statcast
    batter_path = os.path.join(RAW_DIR, "statcast_batters.csv")
    if os.path.exists(batter_path):
        df = pd.read_csv(batter_path)
        # Proxy: launch angle correlate
        # Higher hard_hit% + barrel% → higher launch angle
        if "barrel_pct" in df.columns and "hard_hit_pct" in df.columns:
            df["launch_angle_proxy"] = (
                df["barrel_pct"] * 1.5 + df["hard_hit_pct"] * 0.3 + 8.0
            )
            print(f"  launch_angle_proxy: mean={df['launch_angle_proxy'].mean():.1f}")

        # Proxy: sprint speed (not available, use age inverse)
        df["sprint_speed_proxy"] = 27.0  # league average default

        # Save
        df.to_csv(os.path.join(RAW_DIR, "statcast_batters_v2.csv"), index=False)
        print(f"  Saved statcast_batters_v2.csv: {len(df)} rows")

    return True


def main():
    print("Pulling extra Statcast features...")

    # Try API approach first
    all_data = {}
    for season in SEASONS:
        for stat in ["launch_angle", "sprint_speed"]:
            data = fetch_statcast_stat(stat, season)
            if data:
                all_data[(stat, season)] = data
            time.sleep(1.2)

    if all_data:
        print(f"\nFetched {len(all_data)} stat/season combos")
        # Process and save
        # TODO: parse the leaderboard format
    else:
        print("\nAPI approach didn't return data, using proxy features")
        pull_from_csv_fallback()


if __name__ == "__main__":
    main()
