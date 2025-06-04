#!/usr/bin/env python3
"""
F1 Fantasy Optimizer Suite
Complete pipeline for VFM calculation, track affinity analysis, and team optimization
Enhanced with improved affinity calculations and FP2 pace integration
"""

import pandas as pd
import numpy as np
import re
import sys
import os
import json
import itertools
import time
import requests
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def get_races_completed(base_path):
    """Calculate number of races completed based on race columns in data file"""
    try:
        driver_data = pd.read_csv(f'{base_path}driver_race_data.csv')
        race_columns = [col for col in driver_data.columns if col.startswith('Race')]
        return len(race_columns)
    except Exception as e:
        print(f"Error reading driver data: {e}")
        return None


def get_expected_race_pace(session_key):
    """
    Calculate the expected race pace for each driver after a given practice session.
    Parameters:
      session_key (int): The unique identifier for the practice session.
    Returns:
      pandas.DataFrame: A DataFrame containing driver_number and their average lap time.
    """
    url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    lap_data = response.json()

    df = pd.DataFrame(lap_data)
    if df.empty:
        raise Exception("No lap data found for the given session_key.")

    if "lap_duration" in df.columns:
        lap_time_col = "lap_duration"
    elif all(col in df.columns for col in ["duration_sector_1", "duration_sector_2", "duration_sector_3"]):
        df["calculated_lap_time"] = (
            df["duration_sector_1"] + df["duration_sector_2"] + df["duration_sector_3"]
        )
        lap_time_col = "calculated_lap_time"
    else:
        raise Exception(f"Could not find lap time data. Available columns: {df.columns.tolist()}")

    df = df[df[lap_time_col].notnull()]
    df = df[df[lap_time_col] > 0]
    df[lap_time_col] = pd.to_numeric(df[lap_time_col], errors="coerce")
    df = df[df[lap_time_col].notnull()]
    df = df[df[lap_time_col] < 200]
    if df.empty:
        raise Exception("No valid lap times found after filtering.")

    driver_stats = df.groupby("driver_number")[lap_time_col].agg(
        ["mean", "std", "count"]
    ).reset_index()

    driver_stats = driver_stats[driver_stats["count"] >= 3]
    if driver_stats.empty:
        raise Exception("No drivers with at least 3 laps found.")

    filtered_laps = []
    for _, driver_stat in driver_stats.iterrows():
        dn = driver_stat["driver_number"]
        mean_time = driver_stat["mean"]
        std_time = driver_stat["std"]

        driver_laps = df[df["driver_number"] == dn][lap_time_col]
        if std_time > 0:
            lb = mean_time - 2 * std_time
            ub = mean_time + 2 * std_time
            valid = driver_laps[(driver_laps >= lb) & (driver_laps <= ub)]
        else:
            valid = driver_laps

        if not valid.empty:
            filtered_laps.append({
                "driver_number": dn,
                "average_lap_time": valid.mean()
            })

    if not filtered_laps:
        raise Exception("No valid lap data remaining after outlier filtering.")

    result_df = pd.DataFrame(filtered_laps)
    result_df = result_df.sort_values(by="average_lap_time").reset_index(drop=True)
    return result_df


def get_user_configuration():
    """Get configuration from user input"""
    print("F1 Fantasy Optimizer Configuration")
    print("="*80)

    config = {}

    print("\nData Location:")
    default_path = "/content/drive/MyDrive/F1 Fantasy/"
    base_path = input(f"Enter base path for data files [{default_path}]: ").strip()
    config["base_path"] = base_path if base_path else default_path

    if not config["base_path"].endswith("/"):
        config["base_path"] += "/"

    races_completed = get_races_completed(config["base_path"])
    if races_completed is None:
        print("Error: Could not read driver data file.")
        sys.exit(1)
    config["races_completed"] = races_completed
    print(f"\nDetected {races_completed} races completed.")

    # Add meeting_key and year from calendar.csv
    calendar = pd.read_csv(os.path.join(config["base_path"], "calendar.csv"), skipinitialspace=True)
    next_race = f"Race{races_completed + 1}"
    row = calendar[calendar["Race"] == next_race]
    if row.empty:
        print(f"Error: '{next_race}' not found in calendar.csv.")
        sys.exit(1)
    config["next_meeting_key"] = int(row.iloc[0]["meeting_key"])
    config["next_race_year"] = int(row.iloc[0]["year"])

    print("\nCurrent Team - Drivers (5 required):")
    drivers = []
    for i in range(5):
        while True:
            d = input(f"  Driver {i+1}: ").strip()
            if d:
                drivers.append(d)
                break
            print("  Driver name cannot be empty.")
    config["current_drivers"] = drivers

    print("\nCurrent Team - Constructors (2 required):")
    constructors = []
    for i in range(2):
        while True:
            c = input(f"  Constructor {i+1}: ").strip()
            if c:
                constructors.append(c)
                break
            print("  Constructor name cannot be empty.")
    config["current_constructors"] = constructors

    print("\nBudget:")
    while True:
        try:
            rem = input("Remaining budget (in millions) [1.0]: ").strip()
            config["remaining_budget"] = float(rem) if rem else 1.0
            break
        except ValueError:
            print("  Please enter a valid number.")

    print("\nSwap Limits:")
    while True:
        try:
            s1 = input("Maximum swaps for Step 1 (next race) [2]: ").strip()
            config["step1_swaps"] = int(s1) if s1 else 2
            if config["step1_swaps"] < 0:
                print("  Swaps cannot be negative.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")

    while True:
        try:
            s2 = input("Maximum swaps for Step 2 (race after next) [2]: ").strip()
            config["step2_swaps"] = int(s2) if s2 else 2
            if config["step2_swaps"] < 0:
                print("  Swaps cannot be negative.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")

    print("\nVFM Weighting Scheme:")
    print("  1. Equal weights")
    print("  2. Linear decay (older races weighted less)")
    print("  3. Exponential decay")
    print("  4. Moderate decay")
    print("  5. Trend-based (adaptive based on performance trend)")

    scheme_map = {
        "1": "equal",
        "2": "linear_decay",
        "3": "exp_decay",
        "4": "moderate_decay",
        "5": "trend_based"
    }

    while True:
        choice = input("Select scheme (1-5) [5]: ").strip()
        if not choice:
            config["weighting_scheme"] = "trend_based"
            break
        elif choice in scheme_map:
            config["weighting_scheme"] = scheme_map[choice]
            break
        else:
            print("  Please select a valid option (1-5).")

    print("\nRisk Tolerance:")
    print("  1. Low (prefer consistent performers)")
    print("  2. Medium (balanced approach)")
    print("  3. High (prioritize track-specific performance)")

    risk_map = {"1": "low", "2": "medium", "3": "high"}
    while True:
        choice = input("Select risk tolerance (1-3) [2]: ").strip()
        if not choice:
            config["risk_tolerance"] = "medium"
            break
        elif choice in risk_map:
            config["risk_tolerance"] = risk_map[choice]
            break
        else:
            print("  Please select a valid option (1-3).")

    while True:
        try:
            m = input("\nMultiplier for best driver [2]: ").strip()
            config["multiplier"] = int(m) if m else 2
            if config["multiplier"] < 1:
                print("  Multiplier must be at least 1.")
                continue
            break
        except ValueError:
            print("  Please enter a valid integer.")

    print("\nFP2 Race Pace Integration:")
    use_fp2 = input("Use FP2 race pace data? (y/n) [n]: ").strip().lower()
    config["use_fp2_pace"] = (use_fp2 == "y")

    print("\nEnable parallel processing? (y/n) [y]:")
    parallel = input().strip().lower()
    config["use_parallel"] = (parallel != "n")

    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    print(f"Base path: {config['base_path']}")
    print(f"Races completed: {config['races_completed']}")
    print(f"Next meeting_key: {config['next_meeting_key']}, Year: {config['next_race_year']}")
    print(f"Current drivers: {', '.join(config['current_drivers'])}")
    print(f"Current constructors: {', '.join(config['current_constructors'])}")
    print(f"Remaining budget: ${config['remaining_budget']}M")
    print(f"Step 1 swaps: {config['step1_swaps']}")
    print(f"Step 2 swaps: {config['step2_swaps']}")
    print(f"Weighting scheme: {config['weighting_scheme']}")
    print(f"Risk tolerance: {config['risk_tolerance']}")
    print(f"Multiplier: {config['multiplier']}")
    print(f"Use FP2 pace: {'Yes' if config['use_fp2_pace'] else 'No'}")
    print(f"Parallel processing: {'Enabled' if config['use_parallel'] else 'Disabled'}")

    confirm = input("\nProceed with this configuration? (y/n) [y]: ").strip().lower()
    if confirm == "n":
        print("Configuration cancelled.")
        sys.exit(0)

    return config


class F1VFMCalculator:
    """Calculate Value For Money (VFM) scores with outlier removal and FP2 pace integration"""

    def __init__(self, config):
        self.config = config
        self.base_path = config["base_path"]
        self.scheme = config["weighting_scheme"]
        self.use_fp2_pace = config.get("use_fp2_pace", False)
        self.pace_weight = config.get("pace_weight", 0.25)
        self.pace_modifier_type = config.get("pace_modifier_type", "conservative")

    def calculate_vfm(self, race_data_file, vfm_data_file, entity_type="driver", weights=None):
        """Calculate VFM scores with outlier removal and optional FP2 pace integration"""
        race_df = self._calculate_base_vfm(race_data_file, entity_type, weights)

        if self.use_fp2_pace:
            try:
                mk = self.config["next_meeting_key"]
                yr = self.config["next_race_year"]
                sessions_url = (
                    f"https://api.openf1.org/v1/sessions"
                    f"?meeting_key={mk}&session_name=Practice%202&year={yr}"
                )
                resp = requests.get(sessions_url)
                if resp.status_code != 200:
                    raise Exception(f"Failed to fetch session info (status {resp.status_code})")
                sessions_list = resp.json()
                if not sessions_list:
                    raise Exception(
                        f"No sessions found for meeting_key={mk}, year={yr}, session_name='Practice 2'"
                    )
                session_key = int(sessions_list[0]["session_key"])
                pace_data = get_expected_race_pace(session_key)
                race_df = self._apply_pace_modifiers(race_df, pace_data, entity_type)
                print(f"Applied FP2 pace modifiers (session_key={session_key})")
            except Exception as e:
                print(f"Warning: Could not apply FP2 pace data: {e}")
                print("Continuing with historical VFM only...")

        race_df.to_csv(vfm_data_file, index=False)
        return race_df

    def _calculate_base_vfm(self, race_data_file, entity_type, weights):
        """Calculate base VFM scores with outlier removal"""
        race_df = pd.read_csv(race_data_file, skipinitialspace=True)
        race_df.columns = [col.strip() for col in race_df.columns]

        if "Cost" not in race_df.columns:
            raise ValueError("Input data must contain a 'Cost' column.")

        race_columns = [col for col in race_df.columns if col.startswith("Race")]
        num_races = len(race_columns)
        race_df["Cost_Numeric"] = race_df["Cost"].apply(
            lambda x: float(re.sub(r"[^\d.]", "", str(x)))
        )

        entity_col = "Driver" if entity_type.lower() == "driver" else "Constructor"
        race_df = self._remove_outliers(race_df, entity_col, race_columns)

        if self.scheme == "trend_based":
            race_df = self._calculate_trend_based_vfm(race_df, entity_col, race_columns, num_races)
        else:
            race_df = self._calculate_weighted_vfm(race_df, race_columns, num_races, weights)

        race_df["VFM"] = (race_df["Weighted_Points"] / race_df["Cost_Numeric"]).round(2)

        if entity_type.lower() == "driver":
            if self.scheme == "trend_based":
                result_df = race_df[["Driver", "Team", "Cost", "VFM", "Performance_Trend", "Weights_Used"]]
            else:
                result_df = race_df[["Driver", "Team", "Cost", "VFM", "Weights_Used"]]
        else:
            if self.scheme == "trend_based":
                result_df = race_df[["Constructor", "Cost", "VFM", "Performance_Trend", "Weights_Used"]]
            else:
                result_df = race_df[["Constructor", "Cost", "VFM", "Weights_Used"]]

        result_df = result_df.sort_values("VFM", ascending=False)
        return result_df

    def _apply_pace_modifiers(self, race_df, pace_data, entity_type):
        """Apply FP2 pace-based VFM modifications"""
        pace_scores = self._calculate_pace_scores(pace_data)
        driver_mapping = self._load_driver_number_mapping()
        if driver_mapping is None:
            print("Warning: No driver mapping available. Skipping FP2 pace integration.")
            return race_df

        pace_scores = pace_scores.merge(driver_mapping, on="driver_number", how="left")
        entity_col = "Driver" if entity_type.lower() == "driver" else "Constructor"

        race_df["VFM_Pre_Pace"] = race_df["VFM"].copy()
        race_df["Pace_Score"] = 0.0
        race_df["Pace_Modifier"] = 1.0

        for _, row in pace_scores.iterrows():
            if entity_type.lower() == "driver":
                name = row.get("driver_name", "")
            else:
                name = row.get("team_name", "")

            if pd.notna(name) and name in race_df[entity_col].values:
                mask = race_df[entity_col] == name
                score = row["pace_score"]
                modifier = self._calculate_pace_modifier(score)

                race_df.loc[mask, "Pace_Score"] = score
                race_df.loc[mask, "Pace_Modifier"] = modifier
                race_df.loc[mask, "VFM"] = race_df.loc[mask, "VFM"] * modifier

        return race_df

    def _calculate_pace_scores(self, avg_lap_times_df):
        """Convert lap times to relative performance scores"""
        df = avg_lap_times_df.copy()
        fastest_time = df["average_lap_time"].min()
        df["gap_to_fastest"] = (df["average_lap_time"] - fastest_time) / fastest_time * 100

        max_gap = df["gap_to_fastest"].max()
        if max_gap > 0:
            df["pace_score"] = 100 * (1 - df["gap_to_fastest"] / max_gap)
        else:
            df["pace_score"] = 100.0

        return df

    def _calculate_pace_modifier(self, pace_score):
        """Convert pace score to VFM modifier"""
        if self.pace_modifier_type == "aggressive":
            base_mod = 0.6 + (pace_score / 100) * 0.8
        else:
            base_mod = 0.8 + (pace_score / 100) * 0.4

        risk = self.config.get("risk_tolerance", "medium")
        if risk == "low":
            damp = 0.5
            mod = 1.0 + (base_mod - 1.0) * damp
        elif risk == "high":
            amp = 1.5
            mod = 1.0 + (base_mod - 1.0) * amp
        else:
            mod = base_mod

        return mod

    def _load_driver_number_mapping(self):
        """Load driver number to name mapping"""
        mapping_file = os.path.join(self.base_path, "driver_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                return pd.read_csv(mapping_file)
            except Exception as e:
                print(f"Error loading driver mapping: {e}")

        try:
            _ = pd.read_csv(os.path.join(self.base_path, "driver_race_data.csv"))
            print("No driver mapping file found. Create 'driver_mapping.csv' with columns: driver_number, driver_name, team_name")
            return None
        except:
            return None

    def _remove_outliers(self, df, entity_col, race_columns):
        """Remove race results outside 2 standard deviations"""
        df_clean = df.copy()
        for ent in df_clean[entity_col].unique():
            mask = df_clean[entity_col] == ent
            data = df_clean.loc[mask, race_columns].values.flatten()

            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            lb = mean_val - 2 * std_val
            ub = mean_val + 2 * std_val

            for rc in race_columns:
                v = df_clean.loc[mask, rc].values[0]
                if v < lb or v > ub:
                    df_clean.loc[mask, rc] = np.nan

        return df_clean

    def _calculate_trend_based_vfm(self, df, entity_col, race_columns, num_races):
        """Calculate VFM using trend-based weights"""
        entity_weights = {}
        trends = {}

        for ent in df[entity_col].unique():
            ent_data = df[df[entity_col] == ent]
            pts = []
            idxs = []
            for i, col in enumerate(race_columns):
                val = ent_data[col].values[0]
                if not np.isnan(val):
                    pts.append(val)
                    idxs.append(i)

            if len(pts) >= 3:
                x = np.array(idxs)
                coeffs = np.polyfit(x, pts, 1)
                slope = coeffs[0]
                threshold = 1.7
                full_w = [np.nan] * num_races

                if slope > threshold:
                    base_w = [0.6 ** (len(idxs) - i) for i in range(len(idxs))]
                    trends[ent] = "Improving"
                elif slope < -threshold:
                    base_w = [0.7 ** (len(idxs) - i) for i in range(len(idxs))]
                    trends[ent] = "Declining"
                else:
                    base_w = [0.8 + (0.2 * i / (len(idxs) - 1)) for i in range(len(idxs))]
                    trends[ent] = "Stable"

                for i, idx in enumerate(idxs):
                    full_w[idx] = base_w[i]
                entity_weights[ent] = full_w

            else:
                full_w = [np.nan] * num_races
                if len(pts) == 1:
                    base_w = [1.0]
                else:
                    min_w = 0.7
                    base_w = [min_w + ((1.0 - min_w) / (len(pts) - 1)) * i for i in range(len(pts))]

                for i, idx in enumerate(idxs):
                    full_w[idx] = base_w[i]
                entity_weights[ent] = full_w
                trends[ent] = "Insufficient data"

        df["Weighted_Points"] = 0.0
        df["Performance_Trend"] = ""
        df["Weights_Used"] = ""

        for ent in df[entity_col].unique():
            mask = df[entity_col] == ent
            wsum = 0.0
            total_w = 0.0
            for i, col in enumerate(race_columns):
                rv = df.loc[mask, col].values[0]
                w = entity_weights[ent][i]
                if not np.isnan(rv) and not np.isnan(w):
                    wsum += rv * w
                    total_w += w

            if total_w > 0:
                df.loc[mask, "Weighted_Points"] = wsum / total_w

            valid_ws = [w for w in entity_weights[ent] if not np.isnan(w)]
            wstr = ",".join([f"{w:.2f}" for w in valid_ws])
            df.loc[mask, "Weights_Used"] = wstr
            df.loc[mask, "Performance_Trend"] = trends.get(ent, "Unknown")

        return df

    def _calculate_weighted_vfm(self, df, race_columns, num_races, weights):
        """Calculate VFM using fixed weighting schemes"""
        if weights is None:
            if self.scheme == "equal":
                weights = [1] * num_races
            elif self.scheme == "linear_decay":
                weights = [(i + 1) / num_races for i in range(num_races)]
            elif self.scheme == "exp_decay":
                weights = [max(0, 0.8 ** (num_races - i - 1)) for i in range(num_races)]
            elif self.scheme == "moderate_decay":
                min_w = 0.6
                if num_races == 1:
                    weights = [1.0]
                else:
                    weights = [
                        max(0, min_w + ((1.0 - min_w) / (num_races - 1)) * i)
                        for i in range(num_races)
                    ]
            else:
                weights = [1] * num_races

        weights = [max(0, w) for w in weights]
        df["Weighted_Points"] = 0.0
        for idx in df.index:
            wsum = 0.0
            total_w = 0.0
            for i, col in enumerate(race_columns):
                rv = df.loc[idx, col]
                if not np.isnan(rv):
                    wsum += rv * weights[i]
                    total_w += weights[i]
            if total_w > 0:
                df.loc[idx, "Weighted_Points"] = wsum / total_w

        df["Weights_Used"] = ""
        return df

    def run(self):
        """Run VFM calculations for both drivers and constructors"""
        print("Calculating VFM scores...")

        driver_race_file = f"{self.base_path}driver_race_data.csv"
        driver_vfm_file = f"{self.base_path}driver_vfm.csv"
        driver_vfm = self.calculate_vfm(driver_race_file, driver_vfm_file, "driver")

        constructor_race_file = f"{self.base_path}constructor_race_data.csv"
        constructor_vfm_file = f"{self.base_path}constructor_vfm.csv"
        constructor_vfm = self.calculate_vfm(constructor_race_file, constructor_vfm_file, "constructor")

        print(f"VFM calculation complete. Files saved to {self.base_path}")
        return driver_vfm, constructor_vfm


class F1TrackAffinityCalculator:
    """Calculate track affinity scores based on historical performance with enhanced algorithms"""

    def __init__(self, config):
        self.config = config
        self.base_path = config["base_path"]

    def run(self):
        """Run track affinity calculations"""
        print("Calculating track affinities with enhanced algorithms...")

        driver_points = pd.read_csv(f"{self.base_path}driver_race_data.csv")
        constructor_points = pd.read_csv(f"{self.base_path}constructor_race_data.csv")
        race_calendar = pd.read_csv(f"{self.base_path}calendar.csv")
        track_characteristics = pd.read_csv(f"{self.base_path}tracks.csv")

        race_columns = [col for col in driver_points.columns if col.startswith("Race")]
        constructor_race_columns = [col for col in constructor_points.columns if col.startswith("Race")]

        driver_points_clean = self._remove_outliers_advanced(driver_points, "Driver", race_columns)
        constructor_points_clean = self._remove_outliers_advanced(constructor_points, "Constructor", constructor_race_columns)

        driver_performance = self._prepare_performance_data(driver_points_clean, race_columns, "Driver")
        constructor_performance = self._prepare_performance_data(constructor_points_clean, constructor_race_columns, "Constructor")

        driver_circuit_performance = self._merge_track_data(driver_performance, race_calendar, track_characteristics)
        constructor_circuit_performance = self._merge_track_data(constructor_performance, race_calendar, track_characteristics)

        driver_perf_encoded, constructor_perf_encoded, track_chars_encoded = self._encode_categoricals(
            driver_circuit_performance, constructor_circuit_performance, track_characteristics
        )

        char_importance = self._calculate_characteristic_importance(track_chars_encoded)

        driver_char_affinity = self._calculate_enhanced_affinity(driver_perf_encoded, "Driver", char_importance)
        constructor_char_affinity = self._calculate_enhanced_affinity(constructor_perf_encoded, "Constructor", char_importance)

        driver_char_affinity_df = pd.DataFrame(driver_char_affinity).T.fillna(0)
        constructor_char_affinity_df = pd.DataFrame(constructor_char_affinity).T.fillna(0)

        driver_track_affinity = self._calculate_track_affinity(driver_char_affinity_df, track_chars_encoded)
        constructor_track_affinity = self._calculate_track_affinity(constructor_char_affinity_df, track_chars_encoded)

        driver_final_output = self._create_final_output(track_characteristics, driver_track_affinity)
        constructor_final_output = self._create_final_output(track_characteristics, constructor_track_affinity)

        driver_char_affinity_df.round(3).to_csv(f"{self.base_path}driver_characteristic_affinities.csv")
        driver_final_output.round(3).to_csv(f"{self.base_path}driver_affinity.csv", index=False)
        constructor_char_affinity_df.round(3).to_csv(f"{self.base_path}constructor_characteristic_affinities.csv")
        constructor_final_output.round(3).to_csv(f"{self.base_path}constructor_affinity.csv", index=False)

        print(f"Enhanced track affinity calculation complete. Files saved to {self.base_path}")
        return driver_final_output, constructor_final_output

    def _remove_outliers_advanced(self, df, entity_col, race_columns):
        """Enhanced outlier detection using IQR and rolling statistics"""
        df_clean = df.copy()
        for ent in df_clean[entity_col].unique():
            mask = df_clean[entity_col] == ent
            data = df_clean.loc[mask, race_columns].values.flatten()
            data = data[~np.isnan(data)]

            if len(data) < 4:
                continue

            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lb = Q1 - 1.5 * IQR
            ub = Q3 + 1.5 * IQR

            if len(data) >= 6:
                rolling_std = pd.Series(data).rolling(window=3, center=True).std()
                dyn_thresh = rolling_std.mean() * 2.5
                if not np.isnan(dyn_thresh):
                    ent_mean = np.mean(data)
                    lb = max(lb, ent_mean - dyn_thresh)
                    ub = min(ub, ent_mean + dyn_thresh)

            for rc in race_columns:
                val = df_clean.loc[mask, rc].values[0]
                if not np.isnan(val) and (val < lb or val > ub):
                    df_clean.loc[mask, rc] = np.nan

        return df_clean

    def _prepare_performance_data(self, df, race_columns, entity_type):
        """Prepare performance data in long format with race ordering"""
        id_vars = ["Driver", "Team"] if entity_type == "Driver" else ["Constructor"]
        melted = df.melt(
            id_vars=id_vars,
            value_vars=race_columns,
            var_name="Race",
            value_name="Points"
        )
        melted["Race_Number"] = melted["Race"].str.extract(r"(\d+)").astype(int)
        melted = melted.sort_values(["Race_Number"])
        return melted

    def _merge_track_data(self, perf_df, race_calendar, track_characteristics):
        """Merge performance data with track information"""
        merged = perf_df.merge(race_calendar, on="Race", how="left")
        merged = merged.merge(track_characteristics, on=["Grand Prix", "Circuit"], how="left")
        return merged

    def _encode_categoricals(self, driver_perf, constructor_perf, track_chars):
        """Encode categorical variables"""
        categorical_cols = ["Overtaking Opportunities", "Track Speed", "Expected Temperatures"]
        encoders = {}
        track_encoded = track_chars.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            track_encoded[col + "_encoded"] = le.fit_transform(track_encoded[col])
            encoders[col] = le

        driver_enc = driver_perf.copy()
        constructor_enc = constructor_perf.copy()

        for col in categorical_cols:
            if col in driver_enc.columns:
                driver_enc[col + "_encoded"] = encoders[col].transform(driver_enc[col])
            if col in constructor_enc.columns:
                constructor_enc[col + "_encoded"] = encoders[col].transform(constructor_enc[col])

        return driver_enc, constructor_enc, track_encoded

    def _calculate_characteristic_importance(self, track_encoded):
        """Calculate dynamic weights for each characteristic based on variance and impact"""
        char_cols = [
            "Corners", "Length (km)",
            "Overtaking Opportunities_encoded",
            "Track Speed_encoded",
            "Expected Temperatures_encoded"
        ]
        importance = {}
        variances = []

        for char in char_cols:
            var = np.var(track_encoded[char])
            variances.append(var)

        max_var = max(variances) if variances else 0

        for i, char in enumerate(char_cols):
            if max_var > 0:
                importance[char] = 0.5 + (variances[i] / max_var) * 1.5
            else:
                importance[char] = 1.0

        return importance

    def _calculate_enhanced_affinity(self, df, entity_col, char_importance):
        """Calculate enhanced entity affinity to track characteristics"""
        df_valid = df.dropna(subset=["Points"])
        char_cols = [
            "Corners", "Length (km)",
            "Overtaking Opportunities_encoded",
            "Track Speed_encoded",
            "Expected Temperatures_encoded"
        ]
        interaction_pairs = [
            ("Corners", "Track Speed_encoded"),
            ("Length (km)", "Overtaking Opportunities_encoded"),
            ("Track Speed_encoded", "Expected Temperatures_encoded")
        ]

        entity_affinity = {}

        for ent in df_valid[entity_col].unique():
            ent_data = df_valid[df_valid[entity_col] == ent].copy()
            if len(ent_data) < 3:
                continue

            ent_data = ent_data.sort_values("Race_Number")
            n_races = len(ent_data)
            scores = {}

            for char in char_cols:
                if char in ent_data.columns and ent_data[char].notna().sum() >= 2:
                    x = ent_data[char].values
                    y = ent_data["Points"].values

                    long_corr = self._calculate_robust_correlation(x, y)
                    short_corr = None
                    threshold = max(2, int(0.4 * n_races))
                    if n_races >= threshold:
                        x_recent = x[-threshold:]
                        y_recent = y[-threshold:]
                        short_corr = self._calculate_robust_correlation(x_recent, y_recent)

                    if short_corr is not None and not np.isnan(short_corr):
                        blend = 0.7 * long_corr + 0.3 * short_corr
                    else:
                        blend = long_corr

                    conf_w = self._calculate_confidence_weight(x, y)
                    imp_w = char_importance.get(char, 1.0)
                    scores[char] = blend * conf_w * imp_w
                else:
                    scores[char] = 0

            for c1, c2 in interaction_pairs:
                if (
                    c1 in ent_data.columns and c2 in ent_data.columns and
                    ent_data[c1].notna().sum() >= 2 and ent_data[c2].notna().sum() >= 2
                ):
                    inter = ent_data[c1].values * ent_data[c2].values
                    pts = ent_data["Points"].values
                    inter_corr = self._calculate_robust_correlation(inter, pts)
                    conf_w = self._calculate_confidence_weight(inter, pts)
                    scores[f"{c1}_x_{c2}"] = inter_corr * conf_w * 0.5
                else:
                    scores[f"{c1}_x_{c2}"] = 0

            entity_affinity[ent] = scores

        return entity_affinity

    def _calculate_robust_correlation(self, x, y):
        """Calculate robust correlation using multiple methods"""
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            return 0

        xv = x[mask]
        yv = y[mask]
        corrs = []

        lin = np.corrcoef(xv, yv)[0, 1]
        if not np.isnan(lin):
            corrs.append(lin)

        if len(xv) >= 3:
            x_sq = xv ** 2
            quad = np.corrcoef(x_sq, yv)[0, 1]
            if not np.isnan(quad):
                corrs.append(quad)

        if len(np.unique(xv)) > 2:
            med = np.median(xv)
            thresh_mask = xv > med
            if len(np.unique(thresh_mask)) > 1:
                thr_corr = np.corrcoef(thresh_mask.astype(int), yv)[0, 1]
                if not np.isnan(thr_corr):
                    corrs.append(thr_corr)

        return max(corrs, key=abs) if corrs else 0

    def _calculate_confidence_weight(self, x, y):
        """Calculate confidence weight using bootstrap resampling"""
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return 0.1

        xv = x[mask]
        yv = y[mask]
        boot_corrs = []
        n_boot = min(50, len(xv) * 10)

        for _ in range(n_boot):
            idxs = np.random.choice(len(xv), size=len(xv), replace=True)
            sx = xv[idxs]
            sy = yv[idxs]
            corr = np.corrcoef(sx, sy)[0, 1]
            if not np.isnan(corr):
                boot_corrs.append(corr)

        if len(boot_corrs) < 5:
            return 0.1

        ci = np.percentile(boot_corrs, [25, 75])
        width = ci[1] - ci[0]
        max_w = 1.0
        conf_w = max(0.1, 1.0 - min(width / max_w, 0.9))
        return conf_w

    def _calculate_track_affinity(self, char_affinity_df, track_chars_encoded):
        """Calculate entity affinity for each track with interaction effects"""
        base_cols = [
            "Corners", "Length (km)",
            "Overtaking Opportunities_encoded",
            "Track Speed_encoded",
            "Expected Temperatures_encoded"
        ]
        interaction_cols = [c for c in char_affinity_df.columns if "_x_" in c]
        affinities = {}

        for ent in char_affinity_df.index:
            ent_scores = {}
            ent_aff = char_affinity_df.loc[ent]

            for _, track in track_chars_encoded.iterrows():
                score = 0.0
                total_w = 0.0

                for char in base_cols:
                    if char in track and char in ent_aff:
                        val = track[char]
                        aff = ent_aff[char]

                        if char == "Corners":
                            norm = (val - 10) / (27 - 10)
                        elif char == "Length (km)":
                            norm = (val - 3.337) / (7.004 - 3.337)
                        else:
                            mx = track_chars_encoded[char].max()
                            norm = val / mx if mx > 0 else 0

                        norm = np.clip(norm, 0, 1)
                        contrib = aff * norm
                        score += contrib
                        total_w += abs(aff)

                for inter in interaction_cols:
                    if inter in ent_aff:
                        c1, c2 = inter.split("_x_")
                        if c1 in track and c2 in track:
                            v1 = track[c1]
                            v2 = track[c2]

                            if c1 == "Corners":
                                n1 = (v1 - 10) / (27 - 10)
                            elif c1 == "Length (km)":
                                n1 = (v1 - 3.337) / (7.004 - 3.337)
                            else:
                                m1 = track_chars_encoded[c1].max()
                                n1 = v1 / m1 if m1 > 0 else 0

                            if c2 == "Corners":
                                n2 = (v2 - 10) / (27 - 10)
                            elif c2 == "Length (km)":
                                n2 = (v2 - 3.337) / (7.004 - 3.337)
                            else:
                                m2 = track_chars_encoded[c2].max()
                                n2 = v2 / m2 if m2 > 0 else 0

                            n1 = np.clip(n1, 0, 1)
                            n2 = np.clip(n2, 0, 1)
                            inter_val = n1 * n2
                            aff_i = ent_aff[inter]
                            contrib = aff_i * inter_val
                            score += contrib
                            total_w += abs(aff_i)

                if total_w > 0:
                    final_score = score / total_w
                else:
                    final_score = 0

                affinities[ent] = affinities.get(ent, {})
                affinities[ent][track["Circuit"]] = final_score

        return affinities

    def _create_final_output(self, track_characteristics, track_affinity):
        """Create final output table with affinities"""
        affinity_df = pd.DataFrame(track_affinity).T.fillna(0)
        final = track_characteristics[["Grand Prix", "Circuit"]].copy()

        for ent in affinity_df.index:
            final[f"{ent}_affinity"] = final["Circuit"].map(
                affinity_df.loc[ent].to_dict()
            )

        return final.fillna(0)


class F1TeamOptimizer:
    """Optimize F1 Fantasy team selections for upcoming races"""

    def __init__(self, config):
        self.config = config
        self.base_path = config["base_path"]
        self.multiplier = config["multiplier"]
        self.risk_tolerance = config["risk_tolerance"]

        self.drivers_df = None
        self.constructors_df = None
        self.track_affinity_df = None
        self.constructor_affinity_df = None
        self.race_calendar_df = None
        self.max_budget = None
        self.current_team_cost = None

        self._set_risk_weights()

        self.step1_cache = {}
        self.step2_cache = {}

        self.performance_stats = {
            "patterns_evaluated": 0,
            "cache_hits": 0,
            "optimization_time": 0
        }

    def _set_risk_weights(self):
        """Set weights based on risk tolerance"""
        if self.risk_tolerance == "low":
            self.affinity_weight = 0.5
            self.vfm_weight = 1.5
        elif self.risk_tolerance == "high":
            self.affinity_weight = 1.5
            self.vfm_weight = 0.5
        else:
            self.affinity_weight = 1.0
            self.vfm_weight = 1.0

    def load_data(self):
        """Load all required data files"""
        try:
            self.drivers_df = pd.read_csv(f"{self.base_path}driver_vfm.csv")
            self.constructors_df = pd.read_csv(f"{self.base_path}constructor_vfm.csv")
            self.track_affinity_df = pd.read_csv(f"{self.base_path}driver_affinity.csv")
            self.constructor_affinity_df = pd.read_csv(f"{self.base_path}constructor_affinity.csv")
            self.race_calendar_df = pd.read_csv(f"{self.base_path}calendar.csv")

            self.drivers_df["Cost_Value"] = (
                self.drivers_df["Cost"].str.replace(r"[$M]", "", regex=True).astype(float)
            )
            self.constructors_df["Cost_Value"] = (
                self.constructors_df["Cost"].str.replace(r"[$M]", "", regex=True).astype(float)
            )

            self.step1_race = f"Race{self.config['races_completed'] + 1}"
            self.step2_race = f"Race{self.config['races_completed'] + 2}"

            s1 = self.race_calendar_df[self.race_calendar_df["Race"] == self.step1_race]
            s2 = self.race_calendar_df[self.race_calendar_df["Race"] == self.step2_race]
            if s1.empty or s2.empty:
                raise ValueError("Race information not found")

            self.step1_circuit = s1.iloc[0]["Circuit"]
            self.step1_gp = s1.iloc[0]["Grand Prix"]
            self.step2_circuit = s2.iloc[0]["Circuit"]
            self.step2_gp = s2.iloc[0]["Grand Prix"]

            self._prepare_track_affinities()

            cd_cost = self.drivers_df[
                self.drivers_df["Driver"].isin(self.config["current_drivers"])
            ]["Cost_Value"].sum()
            cc_cost = self.constructors_df[
                self.constructors_df["Constructor"].isin(self.config["current_constructors"])
            ]["Cost_Value"].sum()
            self.current_team_cost = cd_cost + cc_cost
            self.max_budget = self.current_team_cost + self.config["remaining_budget"]

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _prepare_track_affinities(self):
        """Prepare track-specific VFM adjustments"""
        self.drivers_df["VFM_Original"] = self.drivers_df["VFM"].copy()
        self.constructors_df["VFM_Original"] = self.constructors_df["VFM"].copy()

        driver_affinity_cols = [
            col for col in self.track_affinity_df.columns if col.endswith("_affinity")
        ]
        constructor_affinity_cols = [
            col for col in self.constructor_affinity_df.columns if col.endswith("_affinity")
        ]

        for step, circuit in [(1, self.step1_circuit), (2, self.step2_circuit)]:
            row = self.track_affinity_df[self.track_affinity_df["Circuit"] == circuit]
            if not row.empty:
                for col in driver_affinity_cols:
                    name = col.replace("_affinity", "")
                    if name in self.drivers_df["Driver"].values:
                        affinity = row.iloc[0][col] if pd.notna(row.iloc[0][col]) else 0
                        idx = self.drivers_df[self.drivers_df["Driver"] == name].index[0]
                        base_vfm = self.drivers_df.loc[idx, "VFM_Original"]
                        adj_vfm = base_vfm * (1 + affinity * self.affinity_weight)

                        self.drivers_df.loc[idx, f"Step {step}_Affinity"] = affinity
                        self.drivers_df.loc[idx, f"Step {step}_VFM"] = adj_vfm

            row2 = self.constructor_affinity_df[self.constructor_affinity_df["Circuit"] == circuit]
            if not row2.empty:
                for col in constructor_affinity_cols:
                    name2 = col.replace("_affinity", "")
                    if name2 in self.constructors_df["Constructor"].values:
                        affinity2 = row2.iloc[0][col] if pd.notna(row2.iloc[0][col]) else 0
                        idx2 = self.constructors_df[self.constructors_df["Constructor"] == name2].index[0]
                        base_vfm2 = self.constructors_df.loc[idx2, "VFM_Original"]
                        adj_vfm2 = base_vfm2 * (1 + affinity2 * self.affinity_weight)

                        self.constructors_df.loc[idx2, f"Step {step}_Affinity"] = affinity2
                        self.constructors_df.loc[idx2, f"Step {step}_VFM"] = adj_vfm2

    def _get_team_data(self, drivers, constructors, step):
        """Get team data with step-specific VFM"""
        d_df = self.drivers_df[self.drivers_df["Driver"].isin(drivers)].copy()
        c_df = self.constructors_df[self.constructors_df["Constructor"].isin(constructors)].copy()

        d_df["VFM"] = d_df[f"Step {step}_VFM"]
        c_df["VFM"] = c_df[f"Step {step}_VFM"]
        return d_df, c_df

    def evaluate_team(self, drivers, constructors, step):
        """Evaluate team performance"""
        cache = self.step1_cache if step == 1 else self.step2_cache
        key = "|".join(sorted(drivers)) + "#" + "|".join(sorted(constructors))
        if key in cache:
            self.performance_stats["cache_hits"] += 1
            return cache[key]

        d_df, c_df = self._get_team_data(drivers, constructors, step)
        cost = d_df["Cost_Value"].sum() + c_df["Cost_Value"].sum()
        if cost > self.max_budget:
            cache[key] = (-1, -1, cost, None)
            return -1, -1, cost, None

        base_vfm = d_df["VFM"].sum() + c_df["VFM"].sum()
        count = len(drivers) + len(constructors)
        best_ppm = -1
        best_pts = -1
        best_driver = None

        for _, row in d_df.iterrows():
            boosted = base_vfm + (self.multiplier - 1) * row["VFM"]
            ppm = boosted / count
            total_pts = ppm * cost
            if total_pts > best_pts:
                best_pts = total_pts
                best_ppm = ppm
                best_driver = row["Driver"]

        cache[key] = (best_pts, best_ppm, cost, best_driver)
        return best_pts, best_ppm, cost, best_driver

    def generate_swap_patterns(self, current_drivers, current_constructors, max_swaps):
        """Generate all valid swap patterns"""
        patterns = []
        min_d = self.drivers_df["Cost_Value"].min()
        min_c = self.constructors_df["Cost_Value"].min()

        for total in range(max_swaps + 1):
            for d_swaps in range(min(total + 1, len(current_drivers) + 1)):
                c_swaps = total - d_swaps
                if c_swaps > len(current_constructors):
                    continue

                potential_savings = d_swaps * min_d + c_swaps * min_c
                if self.current_team_cost - potential_savings > self.max_budget:
                    continue

                for out_d in itertools.combinations(current_drivers, d_swaps):
                    for out_c in itertools.combinations(current_constructors, c_swaps):
                        patterns.append((out_d, out_c))

        return patterns

    def evaluate_swap_pattern(self, pattern, current_drivers, current_constructors,
                              available_drivers, available_constructors, step):
        """Evaluate a specific swap pattern"""
        out_d, out_c = pattern
        best_result = {
            "points": -1,
            "ppm": -1,
            "swaps": [],
            "drivers": current_drivers,
            "constructors": current_constructors,
            "cost": 0,
            "boost_driver": None
        }

        for new_ds in itertools.combinations(available_drivers, len(out_d)):
            cand_d = [d for d in current_drivers if d not in out_d] + list(new_ds)
            for new_cs in itertools.combinations(available_constructors, len(out_c)):
                cand_c = [c for c in current_constructors if c not in out_c] + list(new_cs)

                pts, ppm, cost, boost = self.evaluate_team(cand_d, cand_c, step)
                if pts > best_result["points"]:
                    swaps = [("Driver", old, new) for old, new in zip(out_d, new_ds)]
                    swaps += [("Constructor", old, new) for old, new in zip(out_c, new_cs)]
                    best_result = {
                        "points": pts,
                        "ppm": ppm,
                        "swaps": swaps,
                        "drivers": cand_d,
                        "constructors": cand_c,
                        "cost": cost,
                        "boost_driver": boost
                    }

        self.performance_stats["patterns_evaluated"] += 1
        return best_result

    def optimize_step(self, current_drivers, current_constructors, max_swaps, step):
        """Optimize team for a specific step"""
        patterns = self.generate_swap_patterns(current_drivers, current_constructors, max_swaps)
        avail_d = [d for d in self.drivers_df["Driver"] if d not in current_drivers]
        avail_c = [c for c in self.constructors_df["Constructor"] if c not in current_constructors]

        best_result = {
            "points": -1,
            "ppm": -1,
            "swaps": [],
            "drivers": current_drivers,
            "constructors": current_constructors,
            "cost": 0,
            "boost_driver": None
        }

        base_pts, base_ppm, base_cost, base_boost = self.evaluate_team(current_drivers, current_constructors, step)
        if base_pts > 0:
            best_result = {
                "points": base_pts,
                "ppm": base_ppm,
                "swaps": [],
                "drivers": current_drivers,
                "constructors": current_constructors,
                "cost": base_cost,
                "boost_driver": base_boost
            }

        if self.config["use_parallel"] and len(patterns) > 50:
            with Pool(processes=cpu_count()) as pool:
                results = pool.starmap(
                    self.evaluate_swap_pattern,
                    [
                        (
                            p, current_drivers, current_constructors,
                            avail_d, avail_c, step
                        )
                        for p in patterns
                    ]
                )
                for r in results:
                    if r["points"] > best_result["points"]:
                        best_result = r
        else:
            for p in patterns:
                r = self.evaluate_swap_pattern(
                    p, current_drivers, current_constructors,
                    avail_d, avail_c, step
                )
                if r["points"] > best_result["points"]:
                    best_result = r

        return best_result

    def run_dual_step_optimization(self):
        """Run the complete two-step optimization"""
        start = time.time()
        cd = self.config["current_drivers"]
        cc = self.config["current_constructors"]

        base_s1 = self.evaluate_team(cd, cc, 1)
        base_s2 = self.evaluate_team(cd, cc, 2)

        print(f"\nOptimizing for {self.step1_race} ({self.step1_circuit}) and {self.step2_race} ({self.step2_circuit})")
        print(f"Current team - Step 1: {base_s1[0]:.1f} points, Step 2: {base_s2[0]:.1f} points")

        best = {
            "step1_result": None,
            "step2_result": None,
            "final_points": base_s2[0],
            "step1_points": base_s1[0]
        }

        print("\nEvaluating Step 1 options...")
        patterns1 = self.generate_swap_patterns(cd, cc, self.config["step1_swaps"])

        for i, pat in enumerate(patterns1):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(patterns1)} patterns", end="\r")

            s1_res = self.evaluate_swap_pattern(
                pat, cd, cc,
                [d for d in self.drivers_df["Driver"] if d not in cd],
                [c for c in self.constructors_df["Constructor"] if c not in cc],
                1
            )
            if s1_res["points"] <= 0:
                continue

            s2_res = self.optimize_step(
                s1_res["drivers"], s1_res["constructors"],
                self.config["step2_swaps"], 2
            )
            if (
                s2_res["points"] > best["final_points"] or
                (s2_res["points"] == best["final_points"] and s1_res["points"] > best["step1_points"])
            ):
                best = {
                    "step1_result": s1_res,
                    "step2_result": s2_res,
                    "final_points": s2_res["points"],
                    "step1_points": s1_res["points"]
                }

        self.performance_stats["optimization_time"] = time.time() - start
        return best, base_s1, base_s2

    def print_results(self, optimization_result, base_s1, base_s2):
        """Print optimization results"""
        s1 = optimization_result["step1_result"]
        s2 = optimization_result["step2_result"]

        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)

        if s1 and s1["swaps"]:
            print(f"\nStep 1 - {self.step1_race} at {self.step1_circuit}:")
            print("Swaps:")
            for etype, old, new in s1["swaps"]:
                if etype == "Driver":
                    od = self.drivers_df[self.drivers_df["Driver"] == old].iloc[0]
                    nd = self.drivers_df[self.drivers_df["Driver"] == new].iloc[0]
                    print(f"  {old} → {new} (${od['Cost_Value']}M → ${nd['Cost_Value']}M)")
                else:
                    oc = self.constructors_df[self.constructors_df["Constructor"] == old].iloc[0]
                    nc = self.constructors_df[self.constructors_df["Constructor"] == new].iloc[0]
                    print(f"  {old} → {new} (${oc['Cost_Value']}M → ${nc['Cost_Value']}M)")

            print(f"Expected points: {s1['points']:.1f} (+{s1['points'] - base_s1[0]:.1f})")
            print(f"Boost driver: {s1['boost_driver']}")
        else:
            print(f"\nStep 1 - No changes recommended")

        if s2:
            print(f"\nStep 2 - {self.step2_race} at {self.step2_circuit}:")
            if s2["swaps"]:
                print("Additional swaps:")
                for etype, old, new in s2["swaps"]:
                    if etype == "Driver":
                        od = self.drivers_df[self.drivers_df["Driver"] == old].iloc[0]
                        nd = self.drivers_df[self.drivers_df["Driver"] == new].iloc[0]
                        print(f"  {old} → {new} (${od['Cost_Value']}M → ${nd['Cost_Value']}M)")
                    else:
                        oc = self.constructors_df[self.constructors_df["Constructor"] == old].iloc[0]
                        nc = self.constructors_df[self.constructors_df["Constructor"] == new].iloc[0]
                        print(f"  {old} → {new} (${oc['Cost_Value']}M → ${nc['Cost_Value']}M)")
            else:
                print("No additional swaps")

            print(f"Expected points: {s2['points']:.1f} (+{s2['points'] - base_s2[0]:.1f})")
            print(f"Boost driver: {s2['boost_driver']}")
            print(f"Final budget used: ${s2['cost']:.1f}M / ${self.max_budget:.1f}M")

        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total improvement: {optimization_result['final_points'] - base_s2[0]:.1f} points")
        print(f"Patterns evaluated: {self.performance_stats['patterns_evaluated']:,}")
        print(f"Cache hits: {self.performance_stats['cache_hits']:,}")
        print(f"Time taken: {self.performance_stats['optimization_time']:.1f}s")

        if self.config.get("use_fp2_pace", False):
            print("\n" + "-"*80)
            print("FP2 PACE INTEGRATION")
            print("-"*80)
            print(f"Next meeting_key: {self.config['next_meeting_key']}, Year: {self.config['next_race_year']}")

    def save_results(self, optimization_result):
        """Save optimization results to JSON"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "races": {
                "step1": {"race": self.step1_race, "circuit": self.step1_circuit},
                "step2": {"race": self.step2_race, "circuit": self.step2_circuit},
            },
            "optimization": optimization_result,
            "performance_stats": self.performance_stats,
        }
        fname = f"{self.base_path}optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {fname}")

    def run(self):
        """Run the complete optimization"""
        if not self.load_data():
            return False

        optimization_result, base_s1, base_s2 = self.run_dual_step_optimization()
        self.print_results(optimization_result, base_s1, base_s2)
        self.save_results(optimization_result)
        return True


def main():
    """Main execution function"""
    print("F1 Fantasy Optimizer Suite - Enhanced Edition with FP2 Integration")
    print("=" * 80)

    try:
        config = get_user_configuration()

        vfm_calculator = F1VFMCalculator(config)
        affinity_calculator = F1TrackAffinityCalculator(config)
        team_optimizer = F1TeamOptimizer(config)

        print("\nStarting optimization pipeline...")

        print("\n" + "="*50)
        print("STEP 1: VFM CALCULATION")
        print("="*50)
        driver_vfm, constructor_vfm = vfm_calculator.run()

        print("\n" + "="*50)
        print("STEP 2: TRACK AFFINITY ANALYSIS")
        print("="*50)
        driver_affinity, constructor_affinity = affinity_calculator.run()

        print("\n" + "="*50)
        print("STEP 3: TEAM OPTIMIZATION")
        print("="*50)
        success = team_optimizer.run()

        if success:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"All results saved to: {config['base_path']}")
            print("\nGenerated files:")
            print("- driver_vfm.csv (VFM scores for drivers)")
            print("- constructor_vfm.csv (VFM scores for constructors)")
            print("- driver_characteristic_affinities.csv (Driver characteristic correlations)")
            print("- driver_affinity.csv (Driver track affinities)")
            print("- constructor_characteristic_affinities.csv (Constructor characteristic correlations)")
            print("- constructor_affinity.csv (Constructor track affinities)")
            print("- optimization_[timestamp].json (Optimization results)")
        else:
            print("\nOptimization failed. Please check your data files and configuration.")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
