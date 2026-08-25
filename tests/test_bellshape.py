import numpy as np
import pytest
import hawkes_package as THP


def triangular_kernel(x):
    Growing_region = (x > 0) & (x < 0.5)
    Decaying_region = (x > 0.5) & (x < 1)
    return 2 * x * Growing_region + (-2 * x + 2) * Decaying_region


def test_simulation_runs():
    np.random.seed(0)
    K = THP.BellShapeHawkes(triangular_kernel)
    K.simulate(50)
    assert len(K.Events) == 50


def test_events_strictly_increasing():
    np.random.seed(1)
    K = THP.BellShapeHawkes(triangular_kernel)
    K.simulate(100)
    assert np.all(np.diff(K.Events) > 0)


def test_vectorised_kernel_no_error():
    x = np.linspace(0, 1, 100)
    result = triangular_kernel(x)
    assert result.shape == (100,)
    assert np.all(result >= 0)


def test_propagate_typo_alias_works():
    np.random.seed(3)
    K = THP.BellShapeHawkes(triangular_kernel)
    K.propogate_by_amount(10)  # deprecated alias must still work
    assert len(K.Events) == 10


def test_propagate_correct_name_works():
    np.random.seed(4)
    K = THP.BellShapeHawkes(triangular_kernel)
    K.propagate_by_amount(10)
    assert len(K.Events) == 10


def test_intensity_non_negative():
    np.random.seed(5)
    K = THP.BellShapeHawkes(triangular_kernel)
    K.simulate(30)
    x = np.linspace(0, K.Events[-1], 100)
    _, z = K.intensity_over_interval(x)
    assert np.all(z >= 0)
