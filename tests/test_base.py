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
    """simulate(0) must not consume the bootstrap event or the RNG."""
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=0)
    before = p.Events.copy()
    p.simulate(0)
    np.testing.assert_array_equal(p.Events, before)
    assert p.Sim_num == 0
    # ... and the process must still be usable afterwards.
    p.simulate(5)
    assert len(p.Events) == 5


def test_simulate_negative_raises(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=0)
    with pytest.raises(ValueError, match="non-negative"):
        p.simulate(-1)


def test_sim_num_accumulates(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=1)
    p.simulate(5)
    assert p.Sim_num == 5
    p.simulate(7)
    assert p.Sim_num == 12
    assert len(p.Events) == 12


def test_bootstrap_event_dropped_exactly_once(temporal_cls, exp_kernel, triangular_kernel):
    """The fictitious t=0 event must vanish after the first call and not again."""
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=2)
    assert p.Events.tolist() == [0.0]
    p.simulate(3)
    assert len(p.Events) == 3
    assert np.all(p.Events > 0)
    p.simulate(3)
    assert len(p.Events) == 6


def test_events_strictly_increasing(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=3)
    p.simulate(60)
    assert np.all(np.diff(p.Events) > 0)


@pytest.mark.parametrize("seed", [None, 0, 12345])
def test_rng_accepts_none_int_and_generator(temporal_cls, exp_kernel, triangular_kernel, seed):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=seed)
    p.simulate(4)
    assert len(p.Events) == 4


def test_rng_accepts_existing_generator(temporal_cls, exp_kernel, triangular_kernel):
    gen = np.random.default_rng(99)
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=gen)
    assert p.rng is gen  # default_rng passes a Generator straight through
    p.simulate(4)
    assert len(p.Events) == 4


def test_same_seed_reproduces(temporal_cls, exp_kernel, triangular_kernel):
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    a.simulate(25)
    b.simulate(25)
    np.testing.assert_array_equal(a.Events, b.Events)


def test_different_seeds_differ(temporal_cls, exp_kernel, triangular_kernel):
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=7)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=8)
    a.simulate(25)
    b.simulate(25)
    assert not np.array_equal(a.Events, b.Events)


def test_global_numpy_seed_does_not_control_simulation(temporal_cls, exp_kernel, triangular_kernel):
    """Regression for the 0.2.0 breaking change: np.random.seed is irrelevant."""
    np.random.seed(4)
    a = _make(temporal_cls, exp_kernel, triangular_kernel, rng=11)
    a.simulate(15)
    np.random.seed(999)
    b = _make(temporal_cls, exp_kernel, triangular_kernel, rng=11)
    b.simulate(15)
    np.testing.assert_array_equal(a.Events, b.Events)


def test_intensity_over_interval_shape_and_sign(temporal_cls, exp_kernel, triangular_kernel):
    p = _make(temporal_cls, exp_kernel, triangular_kernel, rng=5)
    p.simulate(20)
    grid = np.linspace(0, float(p.Events[-1]), 50)
    times, intensity = p.intensity_over_interval(grid)
    assert times.shape == intensity.shape
    assert np.all(intensity >= 0)
    assert np.all(np.diff(times) > 0), "grid must be sorted and de-duplicated"
    # every event time appears in the grid
    assert np.isin(p.Events, times).all()


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
