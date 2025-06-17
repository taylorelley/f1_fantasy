import pandas as pd
import pytest

from app import validate_simple_matrix


def test_validate_simple_matrix_valid():
    df = pd.DataFrame({
        "pace_weight": [0.2, 0.3],
        "pace_modifier_type": ["conservative", "aggressive"],
        "weighting_scheme": ["exp_decay", "trend_based"],
        "risk_tolerance": ["low", "high"],
    })
    assert validate_simple_matrix(df) is True


def test_validate_simple_matrix_invalid():
    df = pd.DataFrame({
        "pace_weight": ["bad"],
        "pace_modifier_type": ["unknown"],
        "weighting_scheme": ["invalid"],
        "risk_tolerance": ["extreme"],
    })
    with pytest.raises(ValueError):
        validate_simple_matrix(df)
