"""Tests for MonotoneKernelHawkes.

The historical bug this class carries scars from: the Ogata upper bound M(t)
excluded the contribution of the event at ``t = Events[-1]``, making
``M(t) < lambda(t + eps)``. Every candidate was then accepted unconditionally
and the output was a Poisson process, not a Hawkes one. The invariant itself is
checked in ``statistical/test_thinning_invariant.py``, which covers every class.
"""

import numpy as np
import pytest

from hawkes_package import MonotoneKernelHawkes

NONLINEARITIES = [
    pytest.param(lambda x: x + 2, id="affine"),
    pytest.param(lambda x: np.sqrt(x + 1), id="sqrt"),
    pytest.param(np.exp, id="exp"),
]


def test_event_count(exp_kernel):
    p = MonotoneKernelHawkes(exp_kernel, rng=42)
    p.simulate(100)
    assert len(p.Events) == 100


def test_default_nonlinearity_is_affine(exp_kernel):
    p = MonotoneKernelHawkes(exp_kernel, rng=0)
    assert p.nonlinearity(0.0) == pytest.approx(2.0)


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_nonlinearity_paths(nonlinearity):
    """Each nonlinearity must drive a terminating, well-ordered simulation."""
    p = MonotoneKernelHawkes(
        lambda x: np.exp(-10 * np.asarray(x, dtype=float)),
        nonlinearity=nonlinearity,
        rng=1,
    )
    p.simulate(40)
    assert len(p.Events) == 40
    assert np.all(np.diff(p.Events) > 0)


def test_intensity_applies_the_nonlinearity(exp_kernel):
    p = MonotoneKernelHawkes(exp_kernel, nonlinearity=lambda x: x + 2, rng=2)
    p.simulate(20)
    times, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 60))
    expected = np.array([2.0 + exp_kernel(t - p.Events[p.Events < t]).sum() for t in times])
    np.testing.assert_allclose(intensity, expected)


def test_intensity_non_negative(exp_kernel):
    p = MonotoneKernelHawkes(exp_kernel, rng=3)
    p.simulate(30)
    _, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 100))
    assert np.all(intensity >= 0)


def test_upper_bound_includes_the_most_recent_event(exp_kernel):
    """Regression for the Poisson-degeneracy bug.

    The bound at the last event must dominate the intensity just after it,
    which holds only if the bound counts that event.
    """
    p = MonotoneKernelHawkes(exp_kernel, rng=4)
    p.simulate(10)
    t_last = float(p.Events[-1])
    assert p._upper_bound(t_last) >= p._conditional_intensity(t_last + 1e-9) - 1e-12
    # strictly above the pre-jump intensity, i.e. the last event is counted
    assert p._upper_bound(t_last) > p._conditional_intensity(t_last)
