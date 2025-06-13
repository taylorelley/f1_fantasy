from f1_optimizer import F1VFMCalculator, F1TrackAffinityCalculator


def test_track_affinity(sample_data):
    # first run VFM calculation because affinity requires vfm files
    F1VFMCalculator(sample_data).run()

    affinity_calc = F1TrackAffinityCalculator(sample_data)
    driver_aff, constructor_aff = affinity_calc.run()

    assert not driver_aff.empty
    assert not constructor_aff.empty
    # output should have Circuit column
    assert 'Circuit' in driver_aff.columns
    assert 'Circuit' in constructor_aff.columns
