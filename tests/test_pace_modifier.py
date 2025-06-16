import pandas as pd
import pytest
from f1_optimizer import F1VFMCalculator


def test_calculate_pace_modifier_high_risk(tmp_path):
    config = {
        'base_path': str(tmp_path) + '/',
        'weighting_scheme': 'equal',
        'risk_tolerance': 'high'
    }
    calc = F1VFMCalculator(config)
    mod = calc._calculate_pace_modifier(80)
    assert pytest.approx(mod, rel=1e-2) == 1.18


def test_apply_pace_modifiers(sample_data):
    # create simple mapping file so modifiers are applied
    mapping_csv = "driver_number,driver_name,team_name\n1,DriverA,Team1\n2,DriverB,Team2\n"
    open(sample_data['base_path'] + 'driver_mapping.csv', 'w').write(mapping_csv)
    calc = F1VFMCalculator(sample_data)
    race_df = pd.DataFrame({'Driver':['DriverA','DriverB'], 'VFM':[10.0,8.0]})
    pace_df = pd.DataFrame({'driver_number':[1,2],'average_lap_time':[90.0,92.0]})
    result = calc._apply_pace_modifiers(race_df, pace_df, 'driver')
    # DriverA is fastest so modifier should leave VFM mostly unchanged
    assert result.loc[0, 'Pace_Modifier'] >= 1.0
    # DriverB slower -> modifier < 1
    assert result.loc[1, 'Pace_Modifier'] < 1.0
