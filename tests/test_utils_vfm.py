import pandas as pd
from f1_optimizer import F1VFMCalculator


def test_calculate_pace_scores(tmp_path):
    config = {'base_path': str(tmp_path) + '/', 'weighting_scheme': 'equal', 'outlier_stddev_factor': 0.1}
    calc = F1VFMCalculator(config)
    df = pd.DataFrame({'driver_number': [1, 2], 'average_lap_time': [90.0, 92.0]})
    scored = calc._calculate_pace_scores(df)
    assert 'pace_score' in scored.columns
    assert scored['pace_score'].max() == 100


def test_remove_outliers_basic(tmp_path):
    config = {'base_path': str(tmp_path) + '/', 'weighting_scheme': 'equal', 'outlier_stddev_factor': 0.1}
    calc = F1VFMCalculator(config)
    df = pd.DataFrame({
        'Driver': ['A', 'B'],
        'Race1': [10, 1000],
        'Race2': [12, 1100]
    })
    cleaned = calc._remove_outliers(df, 'Driver', ['Race1', 'Race2'])
    assert pd.isna(cleaned.loc[1, 'Race1'])
    assert pd.isna(cleaned.loc[1, 'Race2'])



