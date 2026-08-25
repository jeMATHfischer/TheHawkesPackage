"""Tests for BellShapeHawkes, whose kernel rises before it decays."""

import numpy as np
import pytest

from hawkes_package import BellShapeHawkes


def test_extremum_is_a_finite_float(triangular_kernel):
    """`ext` must be a plain float: under NumPy 2 a shape-(1,) array here made
    every comparison against it return an array instead of a bool."""
    p = BellShapeHawkes(triangular_kernel)
    assert isinstance(p.ext, float)
    assert np.isfinite(p.ext)
    assert p.ext == pytest.approx(0.5, abs=1e-3)  # triangular kernel peaks at 0.5


def test_simulation_runs(triangular_kernel):
    p = BellShapeHawkes(triangular_kernel, rng=0)
    p.simulate(50)
    assert len(p.Events) == 50


def test_events_strictly_increasing(triangular_kernel):
    p = BellShapeHawkes(triangular_kernel, rng=1)
    p.simulate(100)
    assert np.all(np.diff(p.Events) > 0)


def test_vectorised_kernel(triangular_kernel):
    x = np.linspace(0, 1, 100)
    result = triangular_kernel(x)
    assert result.shape == (100,)
    assert np.all(result >= 0)


def test_intensity_non_negative(triangular_kernel):
    p = BellShapeHawkes(triangular_kernel, rng=5)
    p.simulate(30)
    _, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 100))
    assert np.all(intensity >= 0)


def test_bound_is_inflated_while_the_kernel_rises(triangular_kernel):
    """Both branches of the bound must be reachable and correctly ordered.

    Before the kernel peaks the bound carries an extra peak's worth of
    headroom; after it, the bound collapses onto the plain intensity.
    """
    p = BellShapeHawkes(triangular_kernel, rng=2)
    p.simulate(10)
    t_last = float(p.Events[-1])

    rising = t_last + p.ext / 2  # still before the peak
    settled = t_last + p.ext * 2  # past the peak

    assert p._upper_bound(rising) > p._conditional_intensity(rising)
    assert p._upper_bound(settled) == pytest.approx(p._conditional_intensity(settled))


def test_bound_dominates_intensity_across_the_rise(triangular_kernel):
    """The inflated branch must actually dominate the rising intensity."""
    p = BellShapeHawkes(triangular_kernel, rng=3)
    p.simulate(10)
    t_last = float(p.Events[-1])
    bound = p._upper_bound(t_last)
    grid = np.linspace(t_last + 1e-9, t_last + p.ext, 60)
    assert np.all([p._conditional_intensity(t) <= bound + 1e-9 for t in grid])
