import pandas as pd
import pytest
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer


def setup_optimizer(config):
    F1VFMCalculator(config).run()
    F1TrackAffinityCalculator(config).run()
    opt = F1TeamOptimizer(config)
    assert opt.load_data()
    return opt


def test_get_top_candidates_exclude(sample_data):
    opt = setup_optimizer(sample_data)
    # expected list sorted by Step 1_VFM excluding DriverC
    vals = opt.drivers_df.set_index("Driver")["Step 1_VFM"]
    expected = vals.drop("DriverC").sort_values(ascending=False).index[: opt.top_n].tolist()
    result = opt._get_top_candidates("driver", 1, ["DriverC"])
    assert result == expected
    assert "DriverC" not in result


def test_evaluate_team_budget_and_cache(sample_data):
    opt = setup_optimizer(sample_data)
    # reduce budget so current team exceeds max_budget
    opt.max_budget = opt.current_team_cost - 1
    team = sample_data["current_drivers"], sample_data["current_constructors"]
    first = opt.evaluate_team(*team, step=1)
    assert first[0] == -1 and first[1] == -1
    second = opt.evaluate_team(*team, step=1)
    assert second == first
    assert opt.performance_stats["cache_hits"] == 1
