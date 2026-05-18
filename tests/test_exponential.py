import numpy as np
import pytest
import TheHawkesPackage as THP


def test_stability_guard_raises():
    with pytest.raises(ValueError, match="alpha/beta"):
        THP.ExponentialHawkes(np.array([1.0, 5.0, 1.0]))


def test_stability_guard_boundary():
    # Exactly 1 is unstable
    with pytest.raises(ValueError):
        THP.ExponentialHawkes(np.array([1.0, 1.0, 1.0]))


def test_stable_simulation_runs():
    np.random.seed(0)
    G = THP.ExponentialHawkes(np.array([2.0, 0.5, 1.0]))
    G.simulate(50)
    assert len(G.Events) == 50


def test_reproducibility():
    np.random.seed(42)
    G1 = THP.ExponentialHawkes(np.array([1.0, 0.3, 1.0]))
    G1.simulate(30)
    np.random.seed(42)
    G2 = THP.ExponentialHawkes(np.array([1.0, 0.3, 1.0]))
    G2.simulate(30)
    np.testing.assert_array_equal(G1.Events, G2.Events)


def test_events_strictly_increasing():
    np.random.seed(1)
    G = THP.ExponentialHawkes(np.array([1.0, 0.4, 2.0]))
    G.simulate(100)
    assert np.all(np.diff(G.Events) > 0)


def test_simulate_alias():
    np.random.seed(5)
    G = THP.ExponentialHawkes(np.array([1.0, 0.5, 2.0]))
    G.simulate(10)
    assert len(G.Events) == 10


def test_intensity_over_interval_shape():
    np.random.seed(3)
    G = THP.ExponentialHawkes(np.array([1.0, 0.4, 2.0]))
    G.simulate(20)
    x = np.linspace(0, G.Events[-1], 50)
    times, intensities = G.intensity_over_interval(x)
    assert len(times) == len(intensities)
    assert np.all(intensities >= 0)
