import pytest
from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer


def setup_optimizer(cfg):
    F1VFMCalculator(cfg).run()
    F1TrackAffinityCalculator(cfg).run()
    opt = F1TeamOptimizer(cfg)
    assert opt.load_data()
    return opt


def test_generate_patterns_respect_keep(sample_data):
    sample_data['keep_drivers'] = ['DriverA']
    sample_data['keep_constructors'] = ['Team1']
    opt = setup_optimizer(sample_data)
    patterns = opt.generate_swap_patterns(sample_data['current_drivers'], sample_data['current_constructors'], 2, 1)
    for out_d, out_c in patterns:
        assert 'DriverA' not in out_d
        assert 'Team1' not in out_c
