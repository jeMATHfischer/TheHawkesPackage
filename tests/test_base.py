"""Contract of the shared HawkesProcess / TemporalHawkesProcess machinery."""

import numpy as np
import pytest

from hawkes_package import (
    BellShapeHawkes,
    ExponentialHawkes,
    HawkesProcess,
    MonotoneKernelHawkes,
)

PARAM = np.array([1.0, 0.4, 2.0])


def _make(cls, exp_kernel, triangular_kernel, **kw):
    if cls is ExponentialHawkes:
        return ExponentialHawkes(PARAM, **kw)
    if cls is MonotoneKernelHawkes:
        return MonotoneKernelHawkes(exp_kernel, **kw)
    return BellShapeHawkes(triangular_kernel, **kw)


ALL_TEMPORAL = [ExponentialHawkes, MonotoneKernelHawkes, BellShapeHawkes]


@pytest.fixture(params=ALL_TEMPORAL, ids=lambda c: c.__name__)
def temporal_cls(request):
    return request.param


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        HawkesProcess()


def test_simulate_zero_is_a_noop(temporal_cls, exp_kernel, triangular_kernel):
    """simulate(0) must leave the process untouched and still usable."""
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=0)
    before = p.events.copy()
    p.simulate(0)
    np.testing.assert_array_equal(p.events, before)
    assert p.n_simulated == 0
    # ... and the process must still be usable afterwards.
    p.simulate(5)
    assert len(p.events) == 5


def test_simulate_negative_raises(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=0)
    with pytest.raises(ValueError, match="non-negative"):
        p.simulate(-1)


def test_sim_num_accumulates(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=1)
    p.simulate(5)
    assert p.n_simulated == 5
    p.simulate(7)
    assert p.n_simulated == 12
    assert len(p.events) == 12


def test_events_start_empty(temporal_cls, exp_kernel, triangular_kernel):
    """`Events` must contain only real events, at every moment.

    Before 0.2.0 a fictitious event sat at t=0 until the first simulate() call
    finished, contributing to every intensity sum in the meantime.
    """
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=2)
    assert p.events.size == 0
    p.simulate(3)
    assert len(p.events) == 3
    assert np.all(p.events > 0)
    p.simulate(3)
    assert len(p.events) == 6


def test_events_strictly_increasing(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=3)
    p.simulate(60)
    assert np.all(np.diff(p.events) > 0)


@pytest.mark.parametrize("seed", [None, 0, 12345])
def test_rng_accepts_none_int_and_generator(temporal_cls, exp_kernel, triangular_kernel, seed):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=seed)
    p.simulate(4)
    assert len(p.events) == 4


def test_rng_accepts_existing_generator(temporal_cls, exp_kernel, triangular_kernel):
    gen = np.random.default_rng(99)
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=gen)
    assert p.rng is gen  # default_rng passes a Generator straight through
    p.simulate(4)
    assert len(p.events) == 4


def test_same_seed_reproduces(temporal_cls, exp_kernel, triangular_kernel):
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    a.simulate(25)
    b.simulate(25)
    np.testing.assert_array_equal(a.events, b.events)


def test_different_seeds_differ(temporal_cls, exp_kernel, triangular_kernel):
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=8)
    a.simulate(25)
    b.simulate(25)
    assert not np.array_equal(a.events, b.events)


def test_global_numpy_seed_does_not_control_simulation(temporal_cls, exp_kernel, triangular_kernel):
    """Regression for the 0.2.0 breaking change: np.random.seed is irrelevant."""
    np.random.seed(4)
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=11)
    a.simulate(15)
    np.random.seed(999)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=11)
    b.simulate(15)
    np.testing.assert_array_equal(a.events, b.events)


def test_intensity_over_interval_shape_and_sign(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=5)
    p.simulate(20)
    grid = np.linspace(0, float(p.events[-1]), 50)
    times, intensity = p.intensity_over_interval(grid)
    assert times.shape == intensity.shape
    assert np.all(intensity >= 0)
    assert np.all(np.diff(times) > 0), "grid must be sorted and de-duplicated"
    # every event time appears in the grid
    assert np.isin(p.events, times).all()


def test_exploding_process_raises_instead_of_hanging():
    """An explosive nonlinearity must fail loudly, not spin forever.

    With phi = exp and a unit-mass kernel the intensity diverges: the
    inter-arrival time underflows to exactly zero and the thinning loop stops
    advancing, so before 0.2.0 `simulate` never returned.
    """
    p = MonotoneKernelHawkes(
        lambda x: np.exp(-10 * np.asarray(x, dtype=float)), nonlinearity=np.exp, rng=7
    )
    with pytest.raises(RuntimeError, match="exploding"):
        p.simulate(500)


def test_intensity_before_any_event_is_the_baseline():
    """With no history the intensity is exactly mu.

    Before 0.2.0 a phantom event at t=0 added an excitation term here, so the
    process the simulator drew from was not the documented one.
    """
    mu = 1.0
    p = ExponentialHawkes(np.array([mu, 0.5, 2.0]), rng=0)
    assert p.events.size == 0
    assert p._conditional_intensity(1.0) == pytest.approx(mu)
    assert p._upper_bound(1.0) == pytest.approx(mu)


def test_split_simulate_equals_one_call(temporal_cls, exp_kernel, triangular_kernel):
    """simulate(1); simulate(1) must be bit-identical to simulate(2).

    The docstring promises that repeated calls continue the same realisation.
    Before 0.2.0 the bootstrap event was deleted at the end of the first call,
    so the second started from a different history: over 1500 seeds the mean
    second gap was 28.17 against 22.58, KS p = 4.2e-06.
    """
    split = _make(temporal_cls, exp_kernel, triangular_kernel, rng=3)
    split.simulate(1)
    split.simulate(1)

    single = _make(temporal_cls, exp_kernel, triangular_kernel, rng=3)
    single.simulate(2)

    np.testing.assert_array_equal(split.events, single.events)
    assert split.n_simulated == single.n_simulated == 2


def test_simulate_rejects_a_fractional_count(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=0)
    with pytest.raises(ValueError, match="whole number"):
        p.simulate(2.7)


def test_state_is_consistent_after_a_failure():
    """A caught failure must leave Sim_num agreeing with the recorded events."""
    p = MonotoneKernelHawkes(
        lambda x: np.exp(-10 * np.asarray(x, dtype=float)), nonlinearity=np.exp, rng=7
    )
    with pytest.raises(RuntimeError, match="exploding"):
        p.simulate(500)
    assert p.n_simulated == len(p.events)
    assert np.all(p.events > 0), "no phantom event may survive a failure"
