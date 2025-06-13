import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import pytest

@pytest.fixture
def sample_data(tmp_path):
    base = tmp_path
    base_path = str(base) + "/"

    driver_csv = """Driver,Team,Cost,Race1,Race2,Race3
DriverA,Team1,$10M,10,8,9
DriverB,Team2,$8M,6,5,7
DriverC,Team3,$9M,8,9,6
DriverD,Team4,$7M,5,4,5
DriverE,Team5,$6M,4,3,4
"""
    constructor_csv = """Constructor,Cost,Race1,Race2,Race3
Team1,$20M,20,22,21
Team2,$15M,18,16,15
Team3,$17M,16,18,17
"""
    calendar_csv = """Race,Grand Prix,Circuit
Race1,GP1,Circuit1
Race2,GP2,Circuit2
Race3,GP3,Circuit3
Race4,GP4,Circuit4
"""
    tracks_csv = """Grand Prix,Circuit,Corners,Length (km),Overtaking Opportunities,Track Speed,Expected Temperatures
GP1,Circuit1,12,5.0,High,Medium,Hot
GP2,Circuit2,18,4.5,Medium,High,Warm
GP3,Circuit3,14,5.5,Low,Low,Cold
GP4,Circuit4,10,4.3,Medium,Medium,Warm
"""

    (base / "driver_race_data.csv").write_text(driver_csv)
    (base / "constructor_race_data.csv").write_text(constructor_csv)
    (base / "calendar.csv").write_text(calendar_csv)
    (base / "tracks.csv").write_text(tracks_csv)

    config = {
        "base_path": base_path,
        "races_completed": 1,
        "current_drivers": ["DriverA", "DriverB", "DriverC", "DriverD", "DriverE"],
        "current_constructors": ["Team1", "Team2"],
        "remaining_budget": 5.0,
        "step1_swaps": 2,
        "step2_swaps": 2,
        "weighting_scheme": "equal",
        "risk_tolerance": "medium",
        "multiplier": 2,
        "use_parallel": False,
        "use_fp2_pace": False,
        "top_n_candidates": 5,
        "use_ilp": False,
        "next_meeting_key": None,
        "next_race_year": None,
    }

    return config
