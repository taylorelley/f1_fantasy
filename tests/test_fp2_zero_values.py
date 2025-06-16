import pandas as pd
from unittest.mock import patch, MagicMock
from f1_optimizer import get_expected_race_pace

@patch("f1_optimizer.requests.get")
def test_expected_race_pace_ignores_invalid(mock_get):
    data = [
        {"driver_number": 1, "lap_duration": 91.0},
        {"driver_number": 1, "lap_duration": None},
        {"driver_number": 1, "lap_duration": 92.0},
        {"driver_number": 1, "lap_duration": 93.0},
        {"driver_number": None, "lap_duration": 89.0},
        {"driver_number": 2, "lap_duration": 0},
        {"driver_number": 2, "lap_duration": 95.0},
        {"driver_number": 2, "lap_duration": 94.5},
        {"driver_number": 2, "lap_duration": 96.0},
    ]
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = data
    mock_get.return_value = mock_resp

    df = get_expected_race_pace(789)
    assert list(df["driver_number"]) == [1, 2]
    assert (df["average_lap_time"] > 0).all()
