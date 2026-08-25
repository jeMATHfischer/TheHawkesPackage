"""Tests for the linear exponential-kernel Hawkes process."""

import numpy as np
import pytest

from hawkes_package import ExponentialHawkes


def test_stability_guard_raises():
    with pytest.raises(ValueError, match="alpha/beta"):
        ExponentialHawkes(np.array([1.0, 5.0, 1.0]))


def test_stability_guard_boundary():
    """alpha/beta of exactly 1 is already non-stationary."""
    with pytest.raises(ValueError, match="alpha/beta"):
        ExponentialHawkes(np.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("param", "match"),
    [
        (np.array([1.0, 0.5]), "exactly 3 entries"),
        (np.array([1.0, 0.5, 1.0, 2.0]), "exactly 3 entries"),
        (np.array([1.0, 0.5, 0.0]), "beta must be positive"),
        (np.array([1.0, 0.5, -1.0]), "beta must be positive"),
        (np.array([-1.0, 0.5, 2.0]), "non-negative"),
        (np.array([1.0, -0.5, 2.0]), "non-negative"),
    ],
)
def test_param_validation(param, match):
    with pytest.raises(ValueError, match=match):
        ExponentialHawkes(param)


def test_stable_simulation_runs():
    p = ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=0)
    p.simulate(50)
    assert len(p.Events) == 50


def test_reproducibility():
    a = ExponentialHawkes(np.array([1.0, 0.3, 1.0]), rng=42)
    a.simulate(30)
    b = ExponentialHawkes(np.array([1.0, 0.3, 1.0]), rng=42)
    b.simulate(30)
    np.testing.assert_array_equal(a.Events, b.Events)


def test_intensity_includes_baseline():
    """Regression: before 0.2.0 the baseline mu was omitted from the accessor.

    Before the first event no excitation has accumulated, so the intensity must
    equal mu exactly. The old implementation returned 0 there.
    """
    mu = 2.0
    p = ExponentialHawkes(np.array([mu, 0.5, 1.0]), rng=1)
    p.simulate(40)
    times, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 400))
    before_first = intensity[times < p.Events[0]]
    assert before_first.size > 0
    np.testing.assert_allclose(before_first, mu)


def test_intensity_never_below_baseline():
    mu = 1.5
    p = ExponentialHawkes(np.array([mu, 0.4, 2.0]), rng=2)
    p.simulate(40)
    _, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 300))
    assert np.all(intensity >= mu - 1e-12)


def test_intensity_matches_closed_form():
    """The accessor must agree with the analytic intensity, term for term."""
    mu, alpha, beta = 1.0, 0.4, 2.0
    p = ExponentialHawkes(np.array([mu, alpha, beta]), rng=3)
    p.simulate(15)
    times, intensity = p.intensity_over_interval(np.linspace(0, float(p.Events[-1]), 40))
    expected = np.array(
        [mu + alpha * np.exp(-beta * (t - p.Events[p.Events < t])).sum() for t in times]
    )
    np.testing.assert_allclose(intensity, expected)


def test_intensity_at_event_is_the_left_limit():
    """The value reported at an event time excludes that event.

    So the intensity jumps by exactly alpha as t crosses an event.
    """
    mu, alpha, beta = 1.0, 0.4, 2.0
    p = ExponentialHawkes(np.array([mu, alpha, beta]), rng=4)
    p.simulate(10)
    t_event = float(p.Events[3])

    before = p._conditional_intensity(t_event)
    after = p._conditional_intensity(t_event + 1e-9)
    assert after - before == pytest.approx(alpha, rel=1e-6)

    # the accessor reports that same pre-jump value on its grid
    times, intensity = p.intensity_over_interval(np.array([t_event]))
    idx = int(np.searchsorted(times, t_event))
    assert intensity[idx] == pytest.approx(before)


def test_decay_between_events():
    """With no intervening event the intensity decays monotonically."""
    p = ExponentialHawkes(np.array([1.0, 0.4, 2.0]), rng=5)
    p.simulate(30)
    gaps = np.diff(p.Events)
    i = int(np.argmax(gaps))
    lo, hi = float(p.Events[i]), float(p.Events[i + 1])
    grid = np.linspace(lo + 1e-6, hi - 1e-6, 50)
    vals = np.array([p._conditional_intensity(t) for t in grid])
    assert np.all(np.diff(vals) < 0)
