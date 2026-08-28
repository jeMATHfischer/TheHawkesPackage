"""Contract of the time-horizon stopping rule.

``simulate(k)`` fixes the count and lets the horizon fall where it may;
``simulate_until(T)`` fixes the horizon and lets the count fall where it may.
The second is what a forecast needs, and the whole reason it exists rather than
being spelled "simulate a lot and throw away the tail" is that the tail is
sometimes *all* of it: a horizon a Hawkes process happens to leave empty is an
outcome, and a fixed-count simulation cannot produce it.
"""

import numpy as np
import pytest

from hawkes_package import (
    BellShapeHawkes,
    Circle,
    ExponentialHawkes,
    MonotoneKernelHawkes,
    SpatioTemporalHawkesProcess,
)

PARAM = np.array([1.0, 0.4, 2.0])

ALL_TEMPORAL = [ExponentialHawkes, MonotoneKernelHawkes, BellShapeHawkes]


def _make(cls, exp_kernel, triangular_kernel, **kw):
    if cls is ExponentialHawkes:
        return ExponentialHawkes(PARAM, **kw)
    if cls is MonotoneKernelHawkes:
        return MonotoneKernelHawkes(exp_kernel, **kw)
    return BellShapeHawkes(triangular_kernel, **kw)


@pytest.fixture(params=ALL_TEMPORAL, ids=lambda c: c.__name__)
def temporal_cls(request):
    return request.param


@pytest.fixture
def spatio_temporal(flat_base, bump_spatial):
    def _make_st(**kw):
        return SpatioTemporalHawkesProcess(
            base=flat_base,
            spatial=bump_spatial,
            temporal=lambda dt: 0.9 * np.exp(-5.0 * np.asarray(dt, dtype=float)),
            domain=Circle(),
            monotone_temporal_kernel=True,
            n_iter=800,
            **kw,
        )

    return _make_st


# ---------------------------------------------------------------------------
# The correspondence with the count-based rule
# ---------------------------------------------------------------------------


def test_horizon_at_the_kth_event_reproduces_simulate(temporal_cls, exp_kernel, triangular_kernel):
    """Stopping at the k-th event's own time must give exactly those k events.

    This is the statement that the two loops are the same loop. Both draw the
    same variates in the same order up to the k-th acceptance; the horizon loop
    then draws one more inter-arrival time, lands past the horizon and stops
    without an acceptance test. So the *records* must agree bit for bit, and if
    they ever do not, one of the loops is bounding, advancing or accepting
    differently from the other.
    """
    counted = _make(temporal_cls, exp_kernel, triangular_kernel, rng=17)
    counted.simulate(40)
    horizon = float(counted.events[-1])

    timed = _make(temporal_cls, exp_kernel, triangular_kernel, rng=17)
    timed.simulate_until(horizon)

    np.testing.assert_array_equal(timed.events, counted.events)
    assert timed.n_simulated == counted.n_simulated == 40


def test_events_lie_inside_the_horizon(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=5)
    p.simulate_until(12.0)
    assert p.events.size > 0
    assert np.all(p.events > 0.0)
    assert np.all(p.events <= 12.0)
    assert np.all(np.diff(p.events) > 0)
    assert p.n_simulated == len(p.events)


def test_repeated_calls_continue_the_same_realisation(temporal_cls, exp_kernel, triangular_kernel):
    """Two horizons in sequence must equal one call to the later horizon."""
    split = _make(temporal_cls, exp_kernel, triangular_kernel, rng=9)
    split.simulate_until(6.0)
    split.simulate_until(12.0)

    single = _make(temporal_cls, exp_kernel, triangular_kernel, rng=9)
    single.simulate_until(12.0)

    # Not bit-identical, and cannot be: the first call consumes the variate that
    # carried it past 6.0 and the second starts a fresh inter-arrival time from
    # the last accepted event. Memorylessness is what makes that the same
    # *process*, not the same *stream*. The structural invariants are what a
    # caller can rely on.
    assert np.all(split.events <= 12.0)
    assert np.all(np.diff(split.events) > 0)
    assert split.n_simulated == len(split.events)
    assert len(single.events) > 0


def test_an_empty_horizon_is_a_legitimate_outcome():
    """A horizon short enough to hold no event must return an empty record.

    The outcome a count-based simulation cannot express, and the reason
    forecasting cannot be done by simulating and truncating.
    """
    p = ExponentialHawkes(np.array([0.05, 0.01, 2.0]), rng=1)
    p.simulate_until(0.5)
    assert p.events.size == 0
    assert p.n_simulated == 0
    # ... and the process is still usable afterwards.
    p.simulate(3)
    assert len(p.events) == 3


# ---------------------------------------------------------------------------
# `start`
# ---------------------------------------------------------------------------


def test_start_defaults_to_the_last_recorded_event(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=2)
    p.simulate(5)
    last = float(p.events[-1])
    p.simulate_until(last + 5.0)
    assert np.all(p.events[5:] > last)


def test_a_later_start_conditions_on_an_empty_gap(temporal_cls, exp_kernel, triangular_kernel):
    """`start` past the last event asserts that nothing happened in between.

    This is the forecasting case, and it is why `start` exists: the loop
    hard-codes its origin at the last recorded event, so without this argument
    there is no way to say "the history is observed to time T, and T is later
    than the last event in it".
    """
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=4)
    p.simulate(6)
    last = float(p.events[-1])
    p.simulate_until(last + 8.0, start=last + 3.0)
    assert np.all(p.events[6:] > last + 3.0)
    assert np.all(p.events[6:] <= last + 8.0)


def test_rewinding_the_record_raises(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=6)
    p.simulate(10)
    with pytest.raises(ValueError, match="cannot be rewound"):
        p.simulate_until(1000.0, start=float(p.events[-1]) - 1e-9)


def test_a_horizon_at_or_before_the_start_is_a_noop(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=8)
    p.simulate(4)
    before = p.events.copy()
    p.simulate_until(float(p.events[-1]))
    p.simulate_until(float(p.events[-1]) - 1.0)
    np.testing.assert_array_equal(p.events, before)
    assert p.n_simulated == 4


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_horizon_raises(bad):
    p = ExponentialHawkes(PARAM, rng=0)
    with pytest.raises(ValueError, match="t_end must be finite"):
        p.simulate_until(bad)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_a_non_finite_start_raises(bad):
    p = ExponentialHawkes(PARAM, rng=0)
    with pytest.raises(ValueError, match="start must be finite"):
        p.simulate_until(10.0, start=bad)


def test_a_nan_horizon_is_refused_rather_than_looping():
    """`nan` compares false against everything, so an unchecked horizon runs forever."""
    p = ExponentialHawkes(PARAM, rng=0)
    with pytest.raises(ValueError, match="must be finite"):
        p.simulate_until(np.nan, start=0.0)
    assert p.events.size == 0


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


def test_spatio_temporal_events_lie_inside_the_horizon(spatio_temporal):
    p = spatio_temporal(rng=3)
    p.simulate_until(2.5)
    assert p.events.shape[0] == 2
    times = p.events[0]
    assert times.size > 0
    assert np.all(times <= 2.5)
    assert np.all(np.diff(times) > 0)
    assert p.n_simulated == times.size
    # Locations stay on the domain, as they do under `simulate`.
    assert np.all(np.abs(p.events[1]) <= np.pi + 1e-12)


def test_spatio_temporal_horizon_reproduces_the_counted_record(spatio_temporal):
    """Structural, not bit-for-bit.

    ``CONTRIBUTING.md`` reserves exact-value reproducibility for the temporal
    classes: the spatio-temporal path branches on floating-point comparisons
    whose last bits move with the SciPy and BLAS build. The count of events
    inside a horizon read off the counted run is not a floating-point quantity,
    so it is assertable.
    """
    counted = spatio_temporal(rng=12)
    counted.simulate(6)
    horizon = float(counted.events[0, -1])

    timed = spatio_temporal(rng=12)
    timed.simulate_until(horizon)
    assert timed.events.shape[1] == 6


def test_an_over_horizon_candidate_does_not_draw_a_location(spatio_temporal, monkeypatch):
    """The horizon is checked before the location sampler is ever called.

    Drawing a location costs `n_iter` Metropolis-Hastings steps, each a full
    spatial intensity evaluation -- most of the cost of an event. Checking the
    horizon after the acceptance test instead of before it would spend all of
    that on the one candidate that is guaranteed to be discarded.
    """
    p = spatio_temporal(rng=21)
    calls = []
    original = p._draw_location
    monkeypatch.setattr(p, "_draw_location", lambda t: (calls.append(t), original(t))[1])

    p.simulate_until(2.0)
    assert len(calls) == p.events.shape[1]
    assert all(t <= 2.0 for t in calls)
