import pandas as pd
from f1_optimizer import (
    F1VFMCalculator,
    F1TrackAffinityCalculator,
    F1TeamOptimizer,
)


def build_optimizer(config):
    F1VFMCalculator(config).run()
    F1TrackAffinityCalculator(config).run()
    opt = F1TeamOptimizer(config)
    assert opt.load_data()
    return opt


def test_evaluate_swap_pattern_no_change(sample_data):
    opt = build_optimizer(sample_data)
    drivers = sample_data["current_drivers"]
    constructors = sample_data["current_constructors"]
    pattern = ((), ())
    available_drivers = drivers
    available_constructors = constructors

    expected = opt.evaluate_team(drivers, constructors, 1)
    result = opt.evaluate_swap_pattern(
        pattern, drivers, constructors, available_drivers, available_constructors, 1
    )
    assert result["points"] == expected[0]
    assert result["boost_driver"] in drivers
    assert opt.performance_stats["patterns_evaluated"] == 1


def test_calculate_track_affinity_simple():
    calc = F1TrackAffinityCalculator({"base_path": ""})
    char_aff = pd.DataFrame(
        {
            "Corners": [1.0],
            "Length (km)": [2.0],
            "Overtaking Opportunities_encoded": [1.0],
            "Track Speed_encoded": [1.0],
            "Expected Temperatures_encoded": [1.0],
        },
        index=["A"],
    )
    tracks = pd.DataFrame(
        {
            "Circuit": ["C1", "C2"],
            "Corners": [10, 27],
            "Length (km)": [3.337, 7.004],
            "Overtaking Opportunities_encoded": [0, 2],
            "Track Speed_encoded": [0, 2],
            "Expected Temperatures_encoded": [0, 2],
        }
    )
    result = calc._calculate_track_affinity(char_aff, tracks)
    assert result["A"]["C1"] == 0
    assert result["A"]["C2"] == 1

