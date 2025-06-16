from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer


def test_team_optimizer(sample_data):
    F1VFMCalculator(sample_data).run()
    F1TrackAffinityCalculator(sample_data).run()

    optimizer = F1TeamOptimizer(sample_data)
    assert optimizer.load_data()

    best, base1, base2 = optimizer.run_dual_step_optimization()

    assert 'step1_result' in best
    assert 'step2_result' in best
    assert isinstance(best['final_points'], float)
    assert base1[0] > 0
    assert base2[0] > 0
