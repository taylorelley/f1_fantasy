import numpy as np
import pandas as pd
from f1_optimizer import F1TrackAffinityCalculator, F1VFMCalculator, F1TeamOptimizer


def test_characteristic_importance_bounds(sample_data):
    calc = F1TrackAffinityCalculator(sample_data)
    df = pd.DataFrame({
        'Corners': [10, 15, 20],
        'Length (km)': [4.0, 5.0, 6.0],
        'Overtaking Opportunities_encoded': [0, 1, 2],
        'Track Speed_encoded': [0, 1, 2],
        'Expected Temperatures_encoded': [0, 1, 2]
    })
    imp = calc._calculate_characteristic_importance(df)
    # all expected keys present and values within reasonable range
    assert set(imp.keys()) == {
        'Corners', 'Length (km)',
        'Overtaking Opportunities_encoded',
        'Track Speed_encoded',
        'Expected Temperatures_encoded'
    }
    assert all(0.4 < v <= 2.0 for v in imp.values())


def test_robust_corr_and_confidence(sample_data):
    calc = F1TrackAffinityCalculator(sample_data)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = calc._calculate_robust_correlation(x, y)
    assert np.isclose(corr, 1.0)
    weight = calc._calculate_confidence_weight(x, y)
    assert 0.5 < weight <= 1.0


def test_generate_swap_patterns(sample_data):
    F1VFMCalculator(sample_data).run()
    F1TrackAffinityCalculator(sample_data).run()
    opt = F1TeamOptimizer(sample_data)
    assert opt.load_data()
    patterns = opt.generate_swap_patterns(
        sample_data['current_drivers'],
        sample_data['current_constructors'],
        1,
        1
    )
    assert patterns[0] == ((), ())
    assert len(patterns) == 8
