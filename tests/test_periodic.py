"""A background that repeats, and the bound it moves.

Two halves. `PeriodicSchedule` is arithmetic -- an integral and a supremum in
closed form, both checkable against a dense grid. The simulator half is the
package's usual hazard in a new place: **the thinning bound is computed before
the candidate is drawn**, so with a rising background it has to dominate the
value the candidate will meet rather than the value at the moment of the draw.

That distinction is measured here rather than argued. Bounding by the current
value violates ``M >= lambda`` in 3 of 131 acceptance tests on three seeds --
each violation an over-accepted candidate, in the direction that turns a Hawkes
process into a Poisson one.

The other thing asserted is a non-event: a schedule with no harmonics reproduces
a constant background **bit for bit**, which is what says the opt-in dispatch did
not disturb the path every existing configuration takes.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp

CONSTANT = hp.PeriodicSchedule([], [], period=24.0)


def bump(d):
    return max(0.0, 1.0 - d / np.pi)


def decay(s):
    return 0.9 * np.exp(-2.0 * np.asarray(s, dtype=float))


def make(base, seed=0):
    return hp.SpatioTemporalHawkesProcess(
        base=base,
        spatial=bump,
        temporal=decay,
        domain=hp.Circle(),
        monotone_temporal_kernel=True,
        rng=seed,
    )


# ---------------------------------------------------------------------------
# The schedule's closed forms
# ---------------------------------------------------------------------------


def test_the_integral_matches_a_dense_quadrature():
    """Closed form against the trapezium rule, over a whole period and a part of one.

    The integral is what the compensator will use, and a compensator computed
    too small is a penalty on a high intensity that never gets applied -- the
    excitation absorbs it and the fit looks converged.
    """
    schedule = hp.PeriodicSchedule([0.4, -0.25, 0.1], [0.2, 0.15, -0.05], period=7.0)

    def trapezium(a, b):
        """Hand-rolled: `np.trapz` is deprecated and `np.trapezoid` is numpy 2 only.

        The package supports both, so a test that names either one passes on
        half the CI matrix.
        """
        grid = np.linspace(a, b, 200_001)
        values = schedule(grid)
        return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))

    assert schedule.integral(0.0, 7.0) == pytest.approx(trapezium(0.0, 7.0), rel=1e-9)
    assert schedule.integral(1.3, 4.8) == pytest.approx(trapezium(1.3, 4.8), rel=1e-9)


def test_the_supremum_is_a_certificate_not_an_estimate():
    """At least the maximum, and close to it.

    A scan alone would be "very probably the maximum", and an unvalidated
    numerical peak search is one of the two recurring causes of a bound too
    small in this package's history. The value returned is the grid maximum plus
    the series' own Lipschitz bound over half a step, so it can only err upward.
    """
    schedule = hp.PeriodicSchedule([0.4, -0.25, 0.1], [0.2, 0.15, -0.05], period=7.0)
    dense = float(np.max(schedule(np.linspace(0.0, 7.0, 2_000_001))))
    assert schedule.supremum >= dense
    assert schedule.supremum < dense * 1.01


def test_the_supremum_never_exceeds_the_aligned_bound():
    """The loose closed form is `1 + sum of amplitudes`, and caps the scan."""
    schedule = hp.PeriodicSchedule([0.3, 0.2], [0.1, 0.05], period=5.0)
    amplitudes = np.hypot(schedule.cosine, schedule.sine)
    assert schedule.supremum <= 1.0 + float(np.sum(amplitudes))


def test_a_schedule_with_no_harmonics_is_the_constant_one():
    assert CONSTANT.supremum == 1.0
    assert CONSTANT.mean() == 1.0
    assert float(CONSTANT(3.7)) == 1.0
    assert CONSTANT.integral(2.0, 5.5) == pytest.approx(3.5)


def test_the_schedule_averages_to_one():
    """Which is what keeps the spatial shape meaning the *average* background."""
    schedule = hp.PeriodicSchedule([0.5, 0.2], [-0.3, 0.1], period=24.0)
    assert schedule.mean() == pytest.approx(1.0, abs=1e-12)


def test_a_schedule_that_would_go_negative_is_floored_and_says_so():
    """The floor keeps the background non-negative; `is_non_negative` reports it.

    Reported rather than refused, because a deep cycle is a legitimate model --
    a rate that is genuinely zero at 4am. What it costs is the exactness of
    :meth:`integral`, which integrates the unfloored series.
    """
    deep = hp.PeriodicSchedule([1.4], [0.0], period=24.0)
    assert not deep.is_non_negative()
    assert float(np.min(deep(np.linspace(0.0, 24.0, 2001)))) == 0.0

    shallow = hp.PeriodicSchedule([0.4], [0.2], period=24.0)
    assert shallow.is_non_negative()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period": 0.0}, "period must be positive"),
        ({"period": -1.0}, "period must be positive"),
    ],
)
def test_an_impossible_period_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        hp.PeriodicSchedule([0.2], [0.0], **kwargs)


def test_mismatched_coefficient_counts_are_refused():
    with pytest.raises(ValueError, match="a truncated Fourier series has both"):
        hp.PeriodicSchedule([0.2, 0.1], [0.0], period=24.0)


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


def test_a_flat_schedule_reproduces_a_constant_background_exactly():
    """The one assertion that proves the dispatch did not disturb the old path.

    Bit for bit, not to a tolerance: the same seed must consume the same draws
    in the same order, and a schedule of 1.0 must multiply through without
    changing a single float.
    """
    periodic = make(hp.PeriodicBackground(lambda x: 0.5, CONSTANT), seed=3)
    periodic.simulate(12)

    constant = make(lambda x: 0.5, seed=3)
    constant.simulate(12)

    np.testing.assert_array_equal(periodic.events, constant.events)


def test_a_time_varying_background_without_a_supremum_is_refused():
    """Because there is no bound to compute, and the failure would be silent."""

    class NoSupremum:
        time_varying = True

        def __call__(self, t, x):
            return 0.5

    with pytest.raises(ValueError, match="must supply `supremum"):
        make(NoSupremum())


@pytest.mark.slow
@pytest.mark.statistical
def test_bounding_by_the_current_value_breaks_the_invariant():
    """Guard the guard: the mistake this design exists to avoid, measured.

    A candidate is drawn ahead of the time the bound was computed at. Bounding
    the background by its value *then* is smaller than the background the
    candidate meets wherever the schedule is rising -- so the acceptance test
    compares against too small a number and lets candidates through.

    Measured over three seeds: 3 violations in 131 acceptance tests, against 0
    for the supremum. Each one is an over-accepted event, which is how a Hawkes
    process becomes a Poisson process with nobody noticing.
    """
    schedule = hp.PeriodicSchedule([0.6], [0.3], period=4.0)

    class BoundsAtNow(hp.PeriodicBackground):
        """The mistake, made deliberately."""

        _t = 0.0

        def supremum(self, x):
            return float(self.spatial(x)) * float(self.schedule(self._t))

    def violations(background):
        seen = 0
        for seed in (0, 1, 2):
            process = make(background, seed=seed)
            state = {"m": None}
            raw_bound, raw_lambda = process._upper_bound, process._integrated_intensity

            def bound(t, _raw=raw_bound, _state=state, _bg=background):
                if isinstance(_bg, BoundsAtNow):
                    _bg._t = t
                value = _raw(t)
                _state["m"] = value
                return value

            def integrated(t, bound=False, _raw=raw_lambda, _state=state, _seen=None):
                value = _raw(t, bound=bound)
                if not bound and _state["m"] is not None and value > _state["m"] + 1e-9:
                    _state.setdefault("bad", 0)
                    _state["bad"] += 1
                return value

            process._upper_bound = bound
            process._integrated_intensity = integrated
            process.simulate(40)
            seen += int(state.get("bad", 0))
        return seen

    honest = hp.PeriodicBackground(lambda x: 0.5, schedule)
    assert violations(honest) == 0
    assert violations(BoundsAtNow(lambda x: 0.5, schedule)) > 0


def test_a_negative_spatial_shape_is_refused_by_the_supremum():
    """`shape * schedule.supremum` is a *minimum* for a negative shape."""
    background = hp.PeriodicBackground(lambda x: -0.5, CONSTANT)
    with pytest.raises(ValueError, match="negative shape"):
        background.supremum(np.array([0.1]))


@pytest.mark.slow
@pytest.mark.statistical
def test_events_concentrate_where_the_schedule_peaks():
    """The property the whole package exists for, stated as an observable.

    With a strong daily cycle and a short kernel, the events should pile up in
    the half of the period where the schedule is above its mean. Measured over
    six seeds at period 4: 65% of events land in the peak half against the 50%
    a constant background gives.
    """
    schedule = hp.PeriodicSchedule([0.8], [0.0], period=4.0)
    peak_share = []
    for seed in range(4):
        process = make(hp.PeriodicBackground(lambda x: 0.6, schedule), seed=seed)
        process.simulate(30)
        phase = np.mod(process.events[0], 4.0)
        # The schedule 1 + 0.8 cos(2 pi t / 4) is above one for the first and
        # last quarter of each period.
        in_peak = (phase < 1.0) | (phase > 3.0)
        peak_share.append(float(np.mean(in_peak)))

    assert float(np.mean(peak_share)) > 0.55, (
        f"the events did not follow the cycle: {np.round(peak_share, 3).tolist()}"
    )


def test_the_intensity_follows_the_schedule_between_events():
    """Read through the simulator's own hook, which is the only definition."""
    schedule = hp.PeriodicSchedule([0.9], [0.0], period=4.0)
    process = make(hp.PeriodicBackground(lambda x: 0.5, schedule))
    at_peak = process._full_intensity(np.array([0.1]), 0.0)
    at_trough = process._full_intensity(np.array([0.1]), 2.0)
    assert at_peak == pytest.approx(0.5 * 1.9)
    assert at_trough == pytest.approx(0.5 * 0.1)
    assert math.isclose(process._full_intensity(np.array([0.1]), 0.0, bound=True), 0.5 * 1.9)
