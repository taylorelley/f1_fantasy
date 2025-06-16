import pandas as pd
import pytest
from f1_optimizer import F1VFMCalculator


def test_vfm_calculation(sample_data):
    calc = F1VFMCalculator(sample_data)
    driver_df, constructor_df = calc.run()

    # ensure correct number of rows
    assert len(driver_df) == 5
    assert len(constructor_df) == 3

    # VFM column should be numeric and not null
    assert driver_df['VFM'].notna().all()
    assert constructor_df['VFM'].notna().all()

def test_weighted_vfm_calculation(sample_data):
    calc = F1VFMCalculator(sample_data)
    df = pd.DataFrame({
        'Driver': ['Test'],
        'Race1': [1],
        'Race2': [2],
        'Race3': [3]
    })
    weights = [0.1, 0.3, 0.6]
    result = calc._calculate_weighted_vfm(df, ['Race1', 'Race2', 'Race3'], 3, weights)
    expected = (1*0.1 + 2*0.3 + 3*0.6) / sum(weights)
    assert pytest.approx(result.loc[0, 'Weighted_Points']) == expected

