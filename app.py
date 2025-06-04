# app.py

import os
import json
import traceback
import pandas as pd
import numpy as np
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from f1_optimizer import (
    F1VFMCalculator,
    F1TrackAffinityCalculator,
    F1TeamOptimizer,
    get_races_completed,
    get_expected_race_pace,
)
from data_cache import DataCache

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["DEFAULT_DATA_FOLDER"] = "default_data"
app.secret_key = "your-secret-key-here"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["DEFAULT_DATA_FOLDER"], exist_ok=True)

sessions = {}
data_cache = DataCache()


def get_data_folder(session_id=None):
    if session_id and session_id in sessions:
        folder = sessions[session_id]["folder"]
        return folder if folder.endswith("/") else folder + "/"
    elif has_default_data():
        folder = app.config["DEFAULT_DATA_FOLDER"]
        return folder if folder.endswith("/") else folder + "/"
    return None


def has_default_data():
    required_files = [
        "driver_race_data.csv",
        "constructor_race_data.csv",
        "calendar.csv",
        "tracks.csv",
    ]
    base = app.config["DEFAULT_DATA_FOLDER"]
    for fname in required_files:
        if not os.path.exists(os.path.join(base, fname)):
            return False
    return True


def has_driver_mapping():
    mapping_path = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
    return os.path.exists(mapping_path)


def load_default_data():
    if not has_default_data():
        return None

    try:
        base = app.config["DEFAULT_DATA_FOLDER"] + "/"
        races_completed = get_races_completed(base, data_cache)

        driver_df = data_cache.load_csv(os.path.join(base, "driver_race_data.csv"))
        constructor_df = data_cache.load_csv(os.path.join(base, "constructor_race_data.csv"))

        drivers_list = sorted(driver_df["Driver"].astype(str).unique().tolist())
        constructors_list = sorted(constructor_df["Constructor"].astype(str).unique().tolist())

        return {
            "races_completed": races_completed,
            "drivers": drivers_list,
            "constructors": constructors_list,
            "has_driver_mapping": has_driver_mapping(),
        }
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check_default_data")
def check_default_data():
    default_info = load_default_data()
    return jsonify({"has_default": default_info is not None, "data": default_info})


@app.route("/check_driver_mapping")
def check_driver_mapping():
    return jsonify({"exists": has_driver_mapping()})


@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        update_default = request.form.get("update_default", "false") == "true"
        session_id = request.form.get("session_id")
        if not session_id or update_default:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if update_default:
            target_folder = app.config["DEFAULT_DATA_FOLDER"]
        else:
            target_folder = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
            os.makedirs(target_folder, exist_ok=True)

        required_files = [
            "driver_race_data.csv",
            "constructor_race_data.csv",
            "calendar.csv",
            "tracks.csv",
        ]

        uploaded_files = []
        for field_name, file in request.files.items():
            if not file or not file.filename:
                continue

            filename = None
            for expected in required_files:
                if expected in field_name:
                    filename = expected
                    break
            if filename is None:
                filename = secure_filename(file.filename)

            dest_path = os.path.join(target_folder, filename)
            file.save(dest_path)
            uploaded_files.append(filename)

        missing = [f for f in required_files if f not in uploaded_files]
        if missing:
            return jsonify(
                {
                    "success": False,
                    "message": f"Missing required files: {', '.join(missing)}",
                }
            )

        folder_with_slash = target_folder if target_folder.endswith("/") else target_folder + "/"
        data_cache.clear(folder_with_slash)
        races_completed = get_races_completed(folder_with_slash, data_cache)

        driver_df = data_cache.load_csv(os.path.join(folder_with_slash, "driver_race_data.csv"))
        constructor_df = data_cache.load_csv(os.path.join(folder_with_slash, "constructor_race_data.csv"))
        drivers_list = sorted(driver_df["Driver"].astype(str).unique().tolist())
        constructors_list = sorted(constructor_df["Constructor"].astype(str).unique().tolist())

        if not update_default:
            sessions[session_id] = {
                "folder": folder_with_slash,
                "races_completed": races_completed,
                "drivers": drivers_list,
                "constructors": constructors_list,
            }

        return jsonify(
            {
                "success": True,
                "session_id": "default" if update_default else session_id,
                "races_completed": races_completed,
                "drivers": drivers_list,
                "constructors": constructors_list,
                "updated_default": update_default,
                "has_driver_mapping": has_driver_mapping() if update_default else False,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"Error uploading files: {e}"})


@app.route("/upload_driver_mapping", methods=["POST"])
def upload_driver_mapping():
    try:
        if "driver_mapping" not in request.files:
            return jsonify({"success": False, "message": "No driver mapping file provided"})

        file = request.files["driver_mapping"]
        if not file or not file.filename:
            return jsonify({"success": False, "message": "No file selected"})

        dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
        file.save(dest)

        data_cache.clear(app.config["DEFAULT_DATA_FOLDER"])
        mapping_df = data_cache.load_csv(dest)
        required_cols = ["driver_number", "driver_name", "team_name"]
        if not all(col in mapping_df.columns for col in required_cols):
            os.remove(dest)
            return jsonify(
                {
                    "success": False,
                    "message": f"driver_mapping.csv must contain columns: {', '.join(required_cols)}",
                }
            )
        if mapping_df.empty:
            os.remove(dest)
            return jsonify({"success": False, "message": "Driver mapping file is empty"})
        if not pd.api.types.is_numeric_dtype(mapping_df["driver_number"]):
            os.remove(dest)
            return jsonify({"success": False, "message": "driver_number must be numeric"})

        return jsonify(
            {
                "success": True,
                "message": f"Driver mapping uploaded ({len(mapping_df)} entries).",
                "driver_count": len(mapping_df),
            }
        )

    except Exception as e:
        dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
        if os.path.exists(dest):
            os.remove(dest)
        return jsonify({"success": False, "message": f"Error uploading driver mapping: {e}"})


@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")

        if session_id == "default":
            if not has_default_data():
                return jsonify({"success": False, "message": "No default data; upload files first."})
            data_folder = get_data_folder("default")
            default_info = load_default_data()
            races_completed = default_info["races_completed"]
        else:
            if session_id not in sessions:
                return jsonify({"success": False, "message": "Session not found; upload files first."})
            data_folder = get_data_folder(session_id)
            races_completed = sessions[session_id]["races_completed"]

        use_fp2 = bool(data.get("use_fp2_pace", False))
        raw_pw = data.get("pace_weight", None)
        try:
            pace_weight = float(raw_pw) if raw_pw is not None else 0.25
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "pace_weight must be a number if provided"})

        pace_modifier_type = data.get("pace_modifier_type") or "conservative"

        # Read calendar.csv and find the next race
        cal_path = os.path.join(data_folder, "calendar.csv")
        cal_df = pd.read_csv(cal_path, skipinitialspace=True)
        next_race = f"Race{races_completed + 1}"
        row = cal_df[cal_df["Race"] == next_race]

        # If user asked for FP2 but meeting_key is missing or invalid, disable FP2
        if use_fp2:
            if row.empty or "meeting_key" not in cal_df.columns or pd.isna(row.iloc[0]["meeting_key"]):
                use_fp2 = False
                meeting_key = None
                race_year = None
            else:
                meeting_key = int(row.iloc[0]["meeting_key"])
                race_year = int(row.iloc[0]["year"])
        else:
            meeting_key = None
            race_year = None

        config = {
            "base_path":            data_folder,
            "races_completed":      races_completed,
            "current_drivers":      data.get("current_drivers", []),
            "current_constructors": data.get("current_constructors", []),
            "remaining_budget":     float(data.get("remaining_budget", 0.0)),
            "step1_swaps":          int(data.get("step1_swaps", 0)),
            "step2_swaps":          int(data.get("step2_swaps", 0)),
            "weighting_scheme":     data.get("weighting_scheme", "trend_based"),
            "risk_tolerance":       data.get("risk_tolerance", "medium"),
            "multiplier":           int(data.get("multiplier", 1)),
            "top_n_candidates":     int(data.get("top_n_candidates", 10)),
            "use_ilp":              bool(data.get("use_ilp", False)),
            "use_parallel":         False,
            "use_fp2_pace":         use_fp2,
            "pace_weight":          pace_weight,
            "pace_modifier_type":   pace_modifier_type,
            "next_meeting_key":     meeting_key,
            "next_race_year":       race_year,
        }

        results = {"status": "running", "progress": []}

        if config["use_fp2_pace"]:
            results["progress"].append(f"Fetching FP2 pace data for meeting_key {meeting_key}...")
        results["progress"].append("Calculating VFM scores...")
        vfm_calc = F1VFMCalculator(config, data_cache)
        driver_vfm_df, constructor_vfm_df = vfm_calc.run()
        results["progress"].append("VFM calculation complete")

        # Check whether any Pace_Score column was actually set
        actual_fp2_applied = False
        if config["use_fp2_pace"] and "Pace_Score" in driver_vfm_df.columns:
            if driver_vfm_df["Pace_Score"].sum() > 0:
                actual_fp2_applied = True

        results["progress"].append("Calculating track affinities...")
        affinity_calc = F1TrackAffinityCalculator(config, data_cache)
        driver_aff_df, constructor_aff_df = affinity_calc.run()
        results["progress"].append("Track affinity calculation complete")

        results["progress"].append("Optimizing team selection...")
        optimizer = F1TeamOptimizer(config, data_cache)
        if not optimizer.load_data():
            return jsonify({"success": False, "message": "Error loading data for optimization"})
        best_dict, base_s1, base_s2 = optimizer.run_dual_step_optimization()

        step1 = best_dict.get("step1_result")
        step2 = best_dict.get("step2_result")

        resp = {
            "status": "complete",
            "success": True,
            "optimization": {
                "step1": {
                    "race":             optimizer.step1_race,
                    "circuit":          optimizer.step1_circuit,
                    "swaps":            step1["swaps"] if step1 else [],
                    "expected_points":  step1["points"] if step1 else base_s1[0],
                    "improvement":      (step1["points"] - base_s1[0]) if step1 else 0.0,
                    "boost_driver":     step1["boost_driver"] if step1 else base_s1[3],
                    "team": {
                        "drivers":      step1["drivers"] if step1 else config["current_drivers"],
                        "constructors": step1["constructors"] if step1 else config["current_constructors"],
                    },
                },
                "step2": {
                    "race":             optimizer.step2_race,
                    "circuit":          optimizer.step2_circuit,
                    "swaps":            step2["swaps"] if step2 else [],
                    "expected_points":  step2["points"] if step2 else base_s2[0],
                    "improvement":      (step2["points"] - base_s2[0]) if step2 else 0.0,
                    "boost_driver":     step2["boost_driver"] if step2 else base_s2[3],
                    "team": {
                        "drivers":      step2["drivers"] if step2 else config["current_drivers"],
                        "constructors": step2["constructors"] if step2 else config["current_constructors"],
                    },
                    "budget_used":      step2["cost"] if step2 else base_s2[2],
                    "budget_remaining": round(
                        optimizer.max_budget - (step2["cost"] if step2 else base_s2[2]), 2
                    ),
                },
                "summary": {
                    "total_improvement":  round(best_dict["final_points"] - base_s2[0], 2),
                    "patterns_evaluated": optimizer.performance_stats["patterns_evaluated"],
                    "optimization_time":  round(optimizer.performance_stats["optimization_time"], 2),
                    "step1_time":        round(optimizer.performance_stats["step1_time"], 2),
                    "step2_time":        round(optimizer.performance_stats["step2_time"], 2),
                },
            },
            "progress": results["progress"],
        }

        # Only include pace block if FP2 was actually applied
        if actual_fp2_applied:
            pace_info = {
                "meeting_key":       meeting_key,
                "year":              race_year,
                "pace_weight":       config["pace_weight"],
                "modifier_type":     config["pace_modifier_type"],
                "applied":           True,
                "pace_adjustments":  [],
            }
            for drv in config["current_drivers"]:
                row = optimizer.drivers_df[optimizer.drivers_df["Driver"] == drv]
                if not row.empty:
                    ps = row.iloc[0].get("Pace_Score", 0.0)
                    pm = row.iloc[0].get("Pace_Modifier", 1.0)
                    vfm_pre = row.iloc[0].get("VFM_Pre_Pace", row.iloc[0].get("VFM", 0.0))
                    vfm_post = row.iloc[0].get("VFM", 0.0)
                    if ps > 0:
                        pace_info["pace_adjustments"].append(
                            {
                                "driver":        drv,
                                "pace_score":    round(ps, 1),
                                "pace_modifier": round(pm, 3),
                                "vfm_original":  round(vfm_pre, 2),
                                "vfm_adjusted":  round(vfm_post, 2),
                            }
                        )
            resp["optimization"]["fp2_info"] = pace_info

        return jsonify(resp)

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error during optimization"})


@app.route("/statistics")
def statistics():
    return render_template("statistics.html")


@app.route("/api/statistics")
def get_statistics():
    try:
        data_folder = get_data_folder("default")
        if not data_folder:
            return jsonify({"success": False, "message": "No data; upload files first."})

        driver_race_df = data_cache.load_csv(os.path.join(data_folder, "driver_race_data.csv"))
        constructor_race_df = data_cache.load_csv(os.path.join(data_folder, "constructor_race_data.csv"))
        calendar_df = data_cache.load_csv(os.path.join(data_folder, "calendar.csv"))
        tracks_df = data_cache.load_csv(os.path.join(data_folder, "tracks.csv"))

        driver_aff_path = os.path.join(data_folder, "driver_affinity.csv")
        constructor_aff_path = os.path.join(data_folder, "constructor_affinity.csv")
        driver_char_aff = os.path.join(data_folder, "driver_characteristic_affinities.csv")
        constructor_char_aff = os.path.join(data_folder, "constructor_characteristic_affinities.csv")

        if not all(os.path.exists(p) for p in [driver_aff_path, constructor_aff_path, driver_char_aff, constructor_char_aff]):
            cfg = {
                "base_path":         data_folder,
                "races_completed":   get_races_completed(data_folder, data_cache),
                "weighting_scheme":  "trend_based",
                "use_fp2_pace":      False,
            }
            vfm_calc = F1VFMCalculator(cfg, data_cache)
            vfm_calc.run()
            aff_calc = F1TrackAffinityCalculator(cfg, data_cache)
            aff_calc.run()

        driver_vfm_df = data_cache.load_csv(os.path.join(data_folder, "driver_vfm.csv"))
        constructor_vfm_df = data_cache.load_csv(os.path.join(data_folder, "constructor_vfm.csv"))
        driver_aff_df = data_cache.load_csv(driver_aff_path)
        constructor_aff_df = data_cache.load_csv(constructor_aff_path)
        driver_char_df = data_cache.load_csv(driver_char_aff, index_col=0)
        constructor_char_df = data_cache.load_csv(constructor_char_aff, index_col=0)

        driver_stats = process_driver_statistics(
            driver_race_df, driver_vfm_df, driver_aff_df, driver_char_df, calendar_df, tracks_df
        )
        constructor_stats = process_constructor_statistics(
            constructor_race_df, constructor_vfm_df, constructor_aff_df, constructor_char_df, calendar_df, tracks_df
        )

        return jsonify(
            {
                "success": True,
                "drivers": driver_stats,
                "constructors": constructor_stats,
                "races_completed": get_races_completed(data_folder),
            }
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error calculating statistics"})


def process_driver_statistics(
    driver_race_df,
    driver_vfm_df,
    driver_aff_df,
    driver_char_df,
    calendar_df,
    tracks_df,
):
    stats_list = []
    race_cols = [c for c in driver_race_df.columns if c.startswith("Race")]

    for _, drv_row in driver_vfm_df.iterrows():
        name = str(drv_row["Driver"])
        race_data = driver_race_df[driver_race_df["Driver"] == name]
        if race_data.empty:
            continue

        points_arr = race_data[race_cols].values.flatten()
        valid_pts = [float(p) for p in points_arr if not np.isnan(p)]

        s = {
            "name":            name,
            "team":            str(drv_row.get("Team", "Unknown")),
            "cost":            str(drv_row["Cost"]),
            "cost_value":      float(re.sub(r"[^\d.]", "", str(drv_row["Cost"]))),
            "vfm":             float(drv_row["VFM"]),
            "trend":           str(drv_row.get("Performance_Trend", "Unknown")),
            "avg_points":      float(np.mean(valid_pts)) if valid_pts else 0.0,
            "total_points":    float(np.sum(valid_pts)) if valid_pts else 0.0,
            "races_completed": len(valid_pts),
            "consistency":     float(np.std(valid_pts)) if len(valid_pts) > 1 else 0.0,
            "best_race":       float(np.max(valid_pts)) if valid_pts else 0.0,
            "worst_race":      float(np.min(valid_pts)) if valid_pts else 0.0,
        }

        if len(valid_pts) >= 3:
            last3 = valid_pts[-3:]
            s["recent_form"] = float(np.mean(last3) - s["avg_points"])
        else:
            s["recent_form"] = 0.0

        if name in driver_char_df.index:
            char_vals = driver_char_df.loc[name]
            s["char_affinities"] = {
                "Corners":      float(char_vals.get("Corners", 0.0)),
                "Length":       float(char_vals.get("Length (km)", 0.0)),
                "Overtaking":   float(char_vals.get("Overtaking Opportunities_encoded", 0.0)),
                "Speed":        float(char_vals.get("Track Speed_encoded", 0.0)),
                "Temperature":  float(char_vals.get("Expected Temperatures_encoded", 0.0)),
            }
        else:
            s["char_affinities"] = {
                "Corners":      0.0,
                "Length":       0.0,
                "Overtaking":   0.0,
                "Speed":        0.0,
                "Temperature":  0.0,
            }

        perf_list = []
        for _, cal_row in calendar_df.iterrows():
            race_name = cal_row["Race"]
            if race_name in race_cols:
                idx = race_cols.index(race_name)
                if idx < len(points_arr) and not np.isnan(points_arr[idx]):
                    circuit = str(cal_row["Circuit"])
                    affinity_col = f"{name}_affinity"
                    aff = 0.0
                    if affinity_col in driver_aff_df.columns:
                        circ_row = driver_aff_df[driver_aff_df["Circuit"] == circuit]
                        if not circ_row.empty:
                            aff = float(circ_row.iloc[0][affinity_col])
                    perf_list.append({"circuit": circuit, "points": float(points_arr[idx]), "affinity": aff})

        perf_list.sort(key=lambda x: x["points"], reverse=True)
        s["best_tracks"] = perf_list[:3] if len(perf_list) >= 3 else perf_list
        s["worst_tracks"] = perf_list[-3:] if len(perf_list) >= 3 else []

        upc = []
        completed = len([r for r in race_cols if r in calendar_df["Race"].values])
        for i in range(completed + 1, min(completed + 4, len(race_cols) + 1)):
            rname = f"Race{i}"
            row = calendar_df[calendar_df["Race"] == rname]
            if not row.empty:
                circuit = str(row.iloc[0]["Circuit"])
                aff_col = f"{name}_affinity"
                aff_val = 0.0
                if aff_col in driver_aff_df.columns:
                    circ_row = driver_aff_df[driver_aff_df["Circuit"] == circuit]
                    if not circ_row.empty:
                        aff_val = float(circ_row.iloc[0][aff_col])
                upc.append({"race": rname, "circuit": circuit, "affinity": aff_val})

        s["upcoming_races"] = upc
        stats_list.append(s)

    stats_list.sort(key=lambda x: x["vfm"], reverse=True)
    for rank, entry in enumerate(stats_list, start=1):
        entry["vfm_rank"] = rank

    return stats_list


def process_constructor_statistics(
    constructor_race_df,
    constructor_vfm_df,
    constructor_aff_df,
    constructor_char_df,
    calendar_df,
    tracks_df,
):
    stats_list = []
    race_cols = [c for c in constructor_race_df.columns if c.startswith("Race")]

    for _, const_row in constructor_vfm_df.iterrows():
        name = str(const_row["Constructor"])
        race_data = constructor_race_df[constructor_race_df["Constructor"] == name]
        if race_data.empty:
            continue

        points_arr = race_data[race_cols].values.flatten()
        valid_pts = [float(p) for p in points_arr if not np.isnan(p)]

        s = {
            "name":            name,
            "cost":            str(const_row["Cost"]),
            "cost_value":      float(re.sub(r"[^\d.]", "", str(const_row["Cost"]))),
            "vfm":             float(const_row["VFM"]),
            "trend":           str(const_row.get("Performance_Trend", "Unknown")),
            "avg_points":      float(np.mean(valid_pts)) if valid_pts else 0.0,
            "total_points":    float(np.sum(valid_pts)) if valid_pts else 0.0,
            "races_completed": len(valid_pts),
            "consistency":     float(np.std(valid_pts)) if len(valid_pts) > 1 else 0.0,
            "best_race":       float(np.max(valid_pts)) if valid_pts else 0.0,
            "worst_race":      float(np.min(valid_pts)) if valid_pts else 0.0,
        }

        if len(valid_pts) >= 3:
            last3 = valid_pts[-3:]
            s["recent_form"] = float(np.mean(last3) - s["avg_points"])
        else:
            s["recent_form"] = 0.0

        if name in constructor_char_df.index:
            char_vals = constructor_char_df.loc[name]
            s["char_affinities"] = {
                "Corners":      float(char_vals.get("Corners", 0.0)),
                "Length":       float(char_vals.get("Length (km)", 0.0)),
                "Overtaking":   float(char_vals.get("Overtaking Opportunities_encoded", 0.0)),
                "Speed":        float(char_vals.get("Track Speed_encoded", 0.0)),
                "Temperature":  float(char_vals.get("Expected Temperatures_encoded", 0.0)),
            }
        else:
            s["char_affinities"] = {
                "Corners":      0.0,
                "Length":       0.0,
                "Overtaking":   0.0,
                "Speed":        0.0,
                "Temperature":  0.0,
            }

        perf_list = []
        for _, cal_row in calendar_df.iterrows():
            race_name = cal_row["Race"]
            if race_name in race_cols:
                idx = race_cols.index(race_name)
                if idx < len(points_arr) and not np.isnan(points_arr[idx]):
                    circuit = str(cal_row["Circuit"])
                    aff_col = f"{name}_affinity"
                    aff_val = 0.0
                    if aff_col in constructor_aff_df.columns:
                        circ_row = constructor_aff_df[constructor_aff_df["Circuit"] == circuit]
                        if not circ_row.empty:
                            aff_val = float(circ_row.iloc[0][aff_col])
                    perf_list.append({"circuit": circuit, "points": float(points_arr[idx]), "affinity": aff_val})

        perf_list.sort(key=lambda x: x["points"], reverse=True)
        s["best_tracks"] = perf_list[:3] if len(perf_list) >= 3 else perf_list
        s["worst_tracks"] = perf_list[-3:] if len(perf_list) >= 3 else []

        upc = []
        completed = len([r for r in race_cols if r in calendar_df["Race"].values])
        for i in range(completed + 1, min(completed + 4, len(race_cols) + 1)):
            rname = f"Race{i}"
            row = calendar_df[calendar_df["Race"] == rname]
            if not row.empty:
                circuit = str(row.iloc[0]["Circuit"])
                aff_col = f"{name}_affinity"
                aff_val = 0.0
                if aff_col in constructor_aff_df.columns:
                    circ_row = constructor_aff_df[constructor_aff_df["Circuit"] == circuit]
                    if not circ_row.empty:
                        aff_val = float(circ_row.iloc[0][aff_col])
                upc.append({"race": rname, "circuit": circuit, "affinity": aff_val})

        s["upcoming_races"] = upc
        stats_list.append(s)

    stats_list.sort(key=lambda x: x["vfm"], reverse=True)
    for rank, entry in enumerate(stats_list, start=1):
        entry["vfm_rank"] = rank

    return stats_list


@app.route("/api/export_statistics")
def export_statistics():
    try:
        stats_resp = get_statistics()
        stats_data = stats_resp.get_json()
        if not stats_data.get("success", False):
            return stats_resp

        driver_rows = []
        for drv in stats_data["drivers"]:
            driver_rows.append(
                {
                    "Driver":      drv["name"],
                    "Team":        drv["team"],
                    "Cost":        drv["cost"],
                    "VFM":         drv["vfm"],
                    "VFM_Rank":    drv["vfm_rank"],
                    "Avg_Points":  round(drv["avg_points"], 2),
                    "Total_Points":drv["total_points"],
                    "Consistency": round(drv["consistency"], 2),
                    "Recent_Form": round(drv["recent_form"], 2),
                    "Trend":       drv["trend"],
                }
            )
        df_drv = pd.DataFrame(driver_rows)

        const_rows = []
        for const in stats_data["constructors"]:
            const_rows.append(
                {
                    "Constructor": const["name"],
                    "Cost":        const["cost"],
                    "VFM":         const["vfm"],
                    "VFM_Rank":    const["vfm_rank"],
                    "Avg_Points":  round(const["avg_points"], 2),
                    "Total_Points":const["total_points"],
                    "Consistency": round(const["consistency"], 2),
                    "Recent_Form": round(const["recent_form"], 2),
                    "Trend":       const["trend"],
                }
            )
        df_const = pd.DataFrame(const_rows)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(app.config["RESULTS_FOLDER"], f"statistics_{ts}.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_drv.to_excel(writer, sheet_name="Drivers", index=False)
            df_const.to_excel(writer, sheet_name="Constructors", index=False)

        return send_file(
            out_path,
            as_attachment=True,
            download_name=f"f1_statistics_{ts}.xlsx",
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error exporting statistics"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
