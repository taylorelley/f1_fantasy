import pandas as pd
import pytest
from f1_optimizer import F1VFMCalculator


def test_weighted_vfm_linear_scheme(tmp_path):
    config = {"base_path": str(tmp_path) + "/", "weighting_scheme": "linear_decay"}
    calc = F1VFMCalculator(config)
    df = pd.DataFrame({"Driver": ["A"], "Race1": [1], "Race2": [2], "Race3": [3]})
    result = calc._calculate_weighted_vfm(df, ["Race1", "Race2", "Race3"], 3, None)
    weights = [(i + 1) / 3 for i in range(3)]
    expected = (1 * weights[0] + 2 * weights[1] + 3 * weights[2]) / sum(weights)
    assert pytest.approx(result.loc[0, "Weighted_Points"]) == expected
