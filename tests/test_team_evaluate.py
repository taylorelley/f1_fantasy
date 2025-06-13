from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator, F1TeamOptimizer


def test_evaluate_team(sample_data):
    F1VFMCalculator(sample_data).run()
    F1TrackAffinityCalculator(sample_data).run()
    opt = F1TeamOptimizer(sample_data)
    assert opt.load_data()
    pts, ppm, cost, boost = opt.evaluate_team(
        sample_data['current_drivers'],
        sample_data['current_constructors'],
        1
    )
    assert pts > 0
    assert ppm > 0
    assert cost > 0
    assert boost in sample_data['current_drivers']

