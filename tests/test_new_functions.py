import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from f1_optimizer import get_expected_race_pace, get_races_completed, F1TeamOptimizer


def test_get_races_completed(tmp_path):
    csv = "Driver,Race1,Race2\nA,1,2\nB,3,4\n"
    (tmp_path / "driver_race_data.csv").write_text(csv)
    result = get_races_completed(str(tmp_path) + "/")
    assert result == 2


@patch("f1_optimizer.requests.get")
def test_expected_race_pace_basic(mock_get):
    data = [
        {"driver_number": 1, "lap_duration": 90.0},
        {"driver_number": 1, "lap_duration": 92.0},
        {"driver_number": 1, "lap_duration": 88.0},
        {"driver_number": 2, "lap_duration": 95.0},
        {"driver_number": 2, "lap_duration": 94.0},
        {"driver_number": 2, "lap_duration": 96.0},
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = data
    mock_get.return_value = mock_resp

    df = get_expected_race_pace(123)
    assert list(df.columns) == ["driver_number", "average_lap_time"]
    assert df.iloc[0]["driver_number"] == 1
    assert df.iloc[1]["driver_number"] == 2
    assert df.iloc[0]["average_lap_time"] < df.iloc[1]["average_lap_time"]


@patch("f1_optimizer.requests.get")
def test_expected_race_pace_sector_times(mock_get):
    data = [
        {"driver_number": 1, "duration_sector_1": 30.0, "duration_sector_2": 30.0, "duration_sector_3": 30.0},
        {"driver_number": 1, "duration_sector_1": 30.5, "duration_sector_2": 30.5, "duration_sector_3": 30.5},
        {"driver_number": 1, "duration_sector_1": 29.5, "duration_sector_2": 29.5, "duration_sector_3": 29.5},
        {"driver_number": 2, "duration_sector_1": 31.0, "duration_sector_2": 31.0, "duration_sector_3": 31.0},
        {"driver_number": 2, "duration_sector_1": 31.2, "duration_sector_2": 31.2, "duration_sector_3": 31.2},
        {"driver_number": 2, "duration_sector_1": 30.8, "duration_sector_2": 30.8, "duration_sector_3": 30.8},
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = data
    mock_get.return_value = mock_resp

    df = get_expected_race_pace(456)
    assert df.iloc[0]["driver_number"] == 1
    assert df.iloc[1]["driver_number"] == 2


def test_set_risk_weights(sample_data):
    sample_data["risk_tolerance"] = "low"
    opt = F1TeamOptimizer(sample_data)
    assert opt.affinity_weight == 0.5
    assert opt.vfm_weight == 1.5

    sample_data["risk_tolerance"] = "high"
    opt = F1TeamOptimizer(sample_data)
    assert opt.affinity_weight == 1.5
    assert opt.vfm_weight == 0.5

    sample_data["risk_tolerance"] = "medium"
    opt = F1TeamOptimizer(sample_data)
    assert opt.affinity_weight == 1.0
    assert opt.vfm_weight == 1.0
