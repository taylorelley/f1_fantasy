import pandas as pd
import pytest
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator


def test_load_driver_mapping_success(tmp_path):
    csv = "driver_number,driver_name,team_name\n1,A,TA\n2,B,TB\n"
    (tmp_path / "driver_mapping.csv").write_text(csv)
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "equal"}
    calc = F1VFMCalculator(config)
    df = calc._load_driver_number_mapping()
    assert list(df.columns) == ["driver_number", "driver_name", "team_name"]
    assert df.shape[0] == 2


def test_load_driver_mapping_missing(tmp_path):
    # presence of driver_race_data.csv triggers info message path
    (tmp_path / "driver_race_data.csv").write_text("Driver,Team\nA,TA\n")
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "equal"}
    calc = F1VFMCalculator(config)
    assert calc._load_driver_number_mapping() is None


def test_trend_based_vfm_classification(tmp_path):
    config = {
        "base_path": str(tmp_path) + "/",
        "weighting_scheme": "equal",
        "trend_slope_threshold": 0.0,
    }
    calc = F1VFMCalculator(config)
    df = pd.DataFrame({
        "Driver": ["A", "B"],
        "Race1": [1, 3],
        "Race2": [2, 2],
        "Race3": [3, 1],
    })
    res = calc._calculate_trend_based_vfm(df, "Driver", ["Race1", "Race2", "Race3"], 3)
    trend_a = res.loc[res["Driver"] == "A", "Performance_Trend"].iloc[0]
    trend_b = res.loc[res["Driver"] == "B", "Performance_Trend"].iloc[0]
    assert trend_a == "Improving"
    assert trend_b == "Declining"
    pts_a = res.loc[res["Driver"] == "A", "Weighted_Points"].iloc[0]
    pts_b = res.loc[res["Driver"] == "B", "Weighted_Points"].iloc[0]
    assert pts_a > pts_b


def test_prepare_merge_and_output(tmp_path):
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "equal"}
    calc = F1TrackAffinityCalculator(config)
    df = pd.DataFrame({"Driver": ["A"], "Team": ["TA"], "Race1": [10], "Race2": [20]})
    perf = calc._prepare_performance_data(df, ["Race1", "Race2"], "Driver")
    assert list(perf["Race"]) == ["Race1", "Race2"]
    cal = pd.DataFrame({"Race": ["Race1", "Race2"], "Grand Prix": ["GP1", "GP2"], "Circuit": ["C1", "C2"]})
    tracks = pd.DataFrame({"Grand Prix": ["GP1", "GP2"], "Circuit": ["C1", "C2"], "Corners": [10, 20]})
    merged = calc._merge_track_data(perf, cal, tracks)
    assert set(["Grand Prix", "Circuit"]).issubset(merged.columns)
    affinity = {"A": {"C1": 0.5, "C2": 0.3}}
    final = calc._create_final_output(tracks, affinity)
    assert "A_affinity" in final.columns


def test_pace_modifier_low_risk(tmp_path):
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "equal", "risk_tolerance": "low"}
    calc = F1VFMCalculator(config)
    mod = calc._calculate_pace_modifier(100)
    assert pytest.approx(mod, rel=1e-2) == 1.1


def test_pace_scores_all_equal(tmp_path):
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "equal"}
    calc = F1VFMCalculator(config)
    df = pd.DataFrame({"driver_number": [1, 2], "average_lap_time": [90.0, 90.0]})
    scored = calc._calculate_pace_scores(df)
    assert all(scored["pace_score"] == 100)
