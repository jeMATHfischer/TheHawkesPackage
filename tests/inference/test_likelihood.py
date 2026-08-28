"""The observed history, and the three log-likelihood implementations.

Two kinds of check run here. The first is that ``History`` refuses data whose
defects would be invisible -- a tied event time costs a ``log lambda`` term with
nothing raised, a guessed window costs the ``-int lambda`` over the empty tail.
The second is that the implementations agree: the exponential closed form
against the quadrature that goes through the intensity hook, and the cached
spatio-temporal path against the hooks it is an optimisation of.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    SpatioTemporalLogLikelihood,
    TemporalLogLikelihood,
    bell_shape_model,
    exponential_model,
    monotone_model,
    spatio_temporal_model,
)
from hawkes_package.inference.likelihood import _bind_history

TIMES = np.array([0.31, 0.47, 1.05, 2.30, 2.88, 4.51, 5.60, 7.15, 9.44])


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_history_records_the_window_and_the_events(history):
    assert history.n_events == 13
    assert history.start == 0.0
    assert history.end == 10.0
    assert history.duration == 10.0
    assert history.ndim == 0


def test_tied_event_times_are_refused():
    """Each of two simultaneous events vanishes from the other's intensity.

    The likelihood then loses a ``log lambda`` term and nothing raises -- the
    fit simply comes back with less excitation than the data implies.
    """
    with pytest.raises(ValueError, match="strictly increasing"):
        History(np.array([1.0, 2.0, 2.0, 3.0]), None, 0.0, end=4.0)


def test_unsorted_times_are_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        History(np.array([1.0, 3.0, 2.0]), None, 0.0, end=4.0)


def test_events_outside_the_window_are_refused():
    with pytest.raises(ValueError, match=r"\(start, end\]"):
        History(np.array([1.0, 5.0]), None, 0.0, end=4.0)
    with pytest.raises(ValueError, match=r"\(start, end\]"):
        History(np.array([0.0, 1.0]), None, 0.0, end=4.0)


def test_a_reversed_window_is_refused():
    with pytest.raises(ValueError, match="before start"):
        History(np.array([]), None, 4.0, end=1.0)


def test_end_has_no_default():
    """Keyword-required and undefaulted, deliberately.

    Defaulting it to the last event time silently switches the model between
    "observed on [0, T]" and "stopped at the n-th event". They differ by the
    compensator over the empty tail, which is the information that nothing
    happened after the last event.
    """
    with pytest.raises(TypeError):
        History.from_events(TIMES)  # type: ignore[call-arg]


def test_from_simulation_uses_the_last_event_as_the_window():
    """Correct for `simulate`'s output, and the docstring says why."""
    process = hp.ExponentialHawkes(np.array([2.0, 0.4, 1.0]), rng=1)
    process.simulate(30)
    history = History.from_simulation(process)
    assert history.end == float(process.events[-1])
    assert history.n_events == 30


def test_from_simulation_handles_an_empty_record():
    process = hp.ExponentialHawkes(np.array([2.0, 0.4, 1.0]), rng=1)
    history = History.from_simulation(process)
    assert history.n_events == 0
    assert history.end == history.start == 0.0


def test_upto_truncates_both_the_events_and_the_window(history):
    cut = history.upto(3.0)
    assert np.all(cut.times <= 3.0)
    assert cut.end == 3.0
    assert cut.start == history.start


def test_upto_outside_the_window_is_refused(history):
    with pytest.raises(ValueError, match="leaves the observation window"):
        history.upto(11.0)


def test_from_events_round_trips_a_spatio_temporal_record(spatial_history):
    record = spatial_history.as_process_events()
    assert record.shape == (2, spatial_history.n_events)
    rebuilt = History.from_events(record, start=0.0, end=spatial_history.end)
    np.testing.assert_array_equal(rebuilt.times, spatial_history.times)
    np.testing.assert_array_equal(rebuilt.points, spatial_history.points)


def test_points_must_have_one_column_per_event():
    with pytest.raises(ValueError, match="one column per event"):
        History(TIMES, np.zeros((1, 3)), 0.0, end=10.0)


def test_binding_a_history_keeps_the_process_coherent(history, exp_model, theta):
    """The record and `n_simulated` must not drift apart."""
    process = exp_model(theta, rng=0)
    _bind_history(process, history)
    np.testing.assert_array_equal(process.events, history.times)
    assert process.n_simulated == history.n_events


def test_binding_copies_rather_than_aliases(history, exp_model, theta):
    process = exp_model(theta, rng=0)
    _bind_history(process, history)
    process.simulate(1)
    assert history.n_events == 13, "the observed history must not grow"


# ---------------------------------------------------------------------------
# Temporal likelihoods
# ---------------------------------------------------------------------------


def test_the_two_temporal_implementations_agree(history, exp_model, theta):
    """The closed form and the quadrature through the hook, to 1e-10.

    The exponential path never builds a process; the generic one evaluates
    ``_conditional_intensity``, which is what the Ogata loop thins against. That
    they agree is what makes the fast path an optimisation rather than a second
    model.
    """
    exact = ExponentialLogLikelihood(exp_model).total(theta, history)
    quadrature = TemporalLogLikelihood(exp_model).total(theta, history)
    assert exact == pytest.approx(quadrature, rel=1e-10)


def test_the_likelihood_matches_the_formula_written_out(history, exp_model, theta):
    """Against the definition, not against another implementation."""
    mu, alpha, beta = (float(v) for v in theta)
    times = history.times
    log_sum = sum(
        math.log(mu + alpha * np.sum(np.exp(-beta * (t - times[times < t])))) for t in times
    )
    compensator = mu * history.end + (alpha / beta) * np.sum(
        1.0 - np.exp(-beta * (history.end - times))
    )
    expected = log_sum - compensator
    assert ExponentialLogLikelihood(exp_model).total(theta, history) == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("blocks", [1, 2, 3, 7])
def test_extending_in_blocks_equals_one_shot(history, exp_model, theta, blocks):
    """`extend` many times must equal `total` once, for both temporal paths."""
    for likelihood in (ExponentialLogLikelihood(exp_model), TemporalLogLikelihood(exp_model)):
        state = likelihood.initial_state(history.start)
        boundaries = np.linspace(history.start, history.end, blocks + 1)[1:]
        for upto in boundaries:
            state, _ = likelihood.extend(state, theta, history, float(upto))
        assert state.log_lik == pytest.approx(likelihood.total(theta, history), rel=1e-10)
        assert state.n_events == history.n_events


def test_the_increment_is_what_the_state_gained(history, exp_model, theta):
    likelihood = ExponentialLogLikelihood(exp_model)
    state = likelihood.initial_state(history.start)
    running = 0.0
    for upto in (2.0, 5.0, 10.0):
        state, increment = likelihood.extend(state, theta, history, upto)
        running += increment
        assert state.log_lik == pytest.approx(running, rel=1e-12)


def test_the_empty_tail_is_accounted_for(exp_model, theta):
    """Two windows differing only after the last event give different answers.

    By exactly the compensator over the gap -- the information that nothing
    happened there. If they came out equal, `end` would be decorative.
    """
    likelihood = ExponentialLogLikelihood(exp_model)
    short = History(TIMES, None, 0.0, end=float(TIMES[-1]))
    long = History(TIMES, None, 0.0, end=float(TIMES[-1]) + 5.0)
    difference = likelihood.total(theta, short) - likelihood.total(theta, long)
    gap = likelihood.compensator(theta, long, np.array([long.end])) - likelihood.compensator(
        theta, long, np.array([short.end])
    )
    assert difference == pytest.approx(float(gap[0]), rel=1e-10)
    # At least the background rate over the five-unit gap: the tail cannot cost
    # less than a Poisson process would.
    assert difference > 5.0 * theta[0] * 0.9


def test_extending_backwards_is_refused(history, exp_model, theta):
    likelihood = ExponentialLogLikelihood(exp_model)
    state, _ = likelihood.extend(likelihood.initial_state(history.start), theta, history, 5.0)
    with pytest.raises(ValueError, match="backwards"):
        likelihood.extend(state, theta, history, 3.0)


def test_extending_past_the_window_is_refused(history, exp_model, theta):
    likelihood = ExponentialLogLikelihood(exp_model)
    with pytest.raises(ValueError, match="past the observation window"):
        likelihood.total(theta, history, upto=99.0)


def test_the_compensator_is_non_decreasing(history, exp_model, theta):
    for likelihood in (ExponentialLogLikelihood(exp_model), TemporalLogLikelihood(exp_model)):
        values = likelihood.compensator(theta, history, history.times)
        assert np.all(np.diff(values) >= 0)
        assert values[0] > 0


def test_the_two_compensators_agree(history, exp_model, theta):
    exact = ExponentialLogLikelihood(exp_model).compensator(theta, history, history.times)
    quadrature = TemporalLogLikelihood(exp_model).compensator(theta, history, history.times)
    np.testing.assert_allclose(exact, quadrature, rtol=1e-10)


def test_unsorted_query_times_are_refused(history, exp_model, theta):
    for likelihood in (ExponentialLogLikelihood(exp_model), TemporalLogLikelihood(exp_model)):
        with pytest.raises(ValueError, match="sorted"):
            likelihood.compensator(theta, history, np.array([3.0, 1.0]))


def test_the_exponential_shortcut_refuses_a_model_it_does_not_implement():
    with pytest.raises(ValueError, match="closed form"):
        ExponentialLogLikelihood(monotone_model())


def test_the_temporal_likelihood_refuses_a_spatial_model():
    with pytest.raises(ValueError, match="Use SpatioTemporalLogLikelihood"):
        TemporalLogLikelihood(spatio_temporal_model(hp.Circle(), n_quad=16))


def test_the_spatial_likelihood_refuses_a_temporal_model():
    with pytest.raises(ValueError, match="Use TemporalLogLikelihood"):
        SpatioTemporalLogLikelihood(exponential_model())


def test_a_history_of_no_events_is_pure_compensator(exp_model, theta):
    empty = History(np.array([]), None, 0.0, end=4.0)
    mu = theta[0]
    value = ExponentialLogLikelihood(exp_model).total(theta, empty)
    assert value == pytest.approx(-mu * 4.0, rel=1e-12)


def test_a_bell_shaped_model_is_evaluable_through_the_hook():
    """The generic path must work for every temporal class, not only the linear one."""
    model = bell_shape_model()
    theta = np.array([1.0, 0.4, 2.5, 2.0])
    process = model(theta, rng=5)
    process.simulate(40)
    history = History.from_simulation(process)
    value = TemporalLogLikelihood(model).total(theta, history)
    assert np.isfinite(value)


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


def test_cached_and_hooks_agree(spatial_history, st_model_1d, st_theta):
    """The cached rearrangement against the definition, to 1e-10.

    ``homogeneous=False`` on purpose: a single mean spatial mass agrees only to
    the spread across events, which is the quadrature error and not zero.
    """
    cached = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    hooks = SpatioTemporalLogLikelihood(st_model_1d, backend="hooks")
    assert cached.total(st_theta, spatial_history) == pytest.approx(
        hooks.total(st_theta, spatial_history), rel=1e-10
    )
    assert cached.backend_used == "cached"
    assert hooks.backend_used == "hooks"


def test_the_homogeneous_shortcut_agrees_on_a_homogeneous_domain(
    spatial_history, st_model_1d, st_theta
):
    """A circle really is homogeneous, so one mean mass loses nothing measurable."""
    single = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=True)
    per_event = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    assert single.total(st_theta, spatial_history) == pytest.approx(
        per_event.total(st_theta, spatial_history), rel=1e-6
    )
    assert single.homogeneous_used
    assert single.spatial_spread < 1e-6


def test_cached_and_hooks_compensators_agree(spatial_history, st_model_1d, st_theta):
    cached = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    hooks = SpatioTemporalLogLikelihood(st_model_1d, backend="hooks")
    query = spatial_history.times
    np.testing.assert_allclose(
        cached.compensator(st_theta, spatial_history, query),
        hooks.compensator(st_theta, spatial_history, query),
        rtol=1e-10,
    )


def test_the_cached_backend_refuses_a_signed_spatial_kernel(spatial_history, st_theta):
    """The precondition of the whole rearrangement, enforced rather than assumed.

    The process floors the intensity *after* summing, so the separability
    identity holds only where the pre-floor integrand is non-negative at every
    node. Below zero the cached form over-counts, the compensator comes out too
    small, and the excitation is biased upward -- silently.
    """

    class SignedSpatial:
        """An excitatory core with an inhibitory ring, as a family."""

        @property
        def spec(self):
            from hawkes_package.inference import Parameter, ParameterSpec

            return ParameterSpec((Parameter("sigma"),))

        def build(self, theta):
            sigma = float(np.asarray(theta).ravel()[0])
            return lambda d: (
                2.0 * np.exp(-((np.asarray(d) / sigma) ** 2))
                - 0.9 * np.exp(-(((np.asarray(d) - 2.0) / sigma) ** 2))
            )

        def mass(self, theta):
            return np.ones(np.atleast_2d(theta).shape[0])

        def min_scale(self, theta):
            return np.atleast_2d(theta)[:, 0]

    model = spatio_temporal_model(hp.Circle(), spatial=SignedSpatial(), n_quad=32)
    likelihood = SpatioTemporalLogLikelihood(model, backend="cached")
    with pytest.raises(ValueError, match="non-negative at every quadrature node"):
        likelihood.total(st_theta, spatial_history)


def test_auto_falls_back_to_the_hooks_and_says_so(spatial_history, st_theta):
    """Never silent about which path ran."""

    class NegativeBase:
        @property
        def spec(self):
            from hawkes_package.inference import Parameter, ParameterSpec

            return ParameterSpec((Parameter("mu", lower=-5.0),))

        def build(self, theta):
            mu = float(np.asarray(theta).ravel()[0])
            return lambda x: mu

        def at(self, theta, points):
            mu = float(np.asarray(theta).ravel()[0])
            return np.full(np.asarray(points).shape[0], mu)

    model = spatio_temporal_model(hp.Circle(), base=NegativeBase(), n_quad=32)
    likelihood = SpatioTemporalLogLikelihood(model, backend="auto")
    theta = np.array([-0.2, 0.6, 1.5, 0.6])
    with pytest.warns(UserWarning, match="fell back to the hooks"):
        likelihood.total(theta, spatial_history)
    assert likelihood.backend_used == "hooks"


def test_a_spatial_likelihood_refuses_a_history_without_locations(st_model_1d, st_theta, history):
    likelihood = SpatioTemporalLogLikelihood(st_model_1d, backend="cached")
    with pytest.raises(ValueError, match="event locations"):
        likelihood.total(st_theta, history)


@pytest.mark.parametrize("blocks", [2, 4])
def test_spatio_temporal_blocks_equal_one_shot(spatial_history, st_model_1d, st_theta, blocks):
    likelihood = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    state = likelihood.initial_state(spatial_history.start)
    for upto in np.linspace(spatial_history.start, spatial_history.end, blocks + 1)[1:]:
        state, _ = likelihood.extend(state, st_theta, spatial_history, float(upto))
    assert state.log_lik == pytest.approx(likelihood.total(st_theta, spatial_history), rel=1e-10)


def test_a_reused_likelihood_refuses_a_different_history_of_the_same_length(
    spatial_history, st_model_1d, st_theta
):
    """The prefix check has to run on every reuse, not only on a length change.

    `extend_geometry` is where the check lives, and `geometry_for` used to call it
    only when the event *count* differed. So a second history of the same length
    got the distance tensors built for the first one, and the log-likelihood came
    back for data nobody had passed -- with nothing raised.
    """
    likelihood = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    likelihood.total(st_theta, spatial_history)

    shifted = History(
        spatial_history.times + 0.01,
        spatial_history.points,
        spatial_history.start,
        end=spatial_history.end,
    )
    with pytest.raises(ValueError, match="does not extend"):
        likelihood.total(st_theta, shifted)


def test_a_reused_likelihood_still_accepts_the_history_it_was_built_for(
    spatial_history, st_model_1d, st_theta
):
    """Guard the fix's cost: the unconditional call must be free, not a rebuild."""
    likelihood = SpatioTemporalLogLikelihood(st_model_1d, backend="cached", homogeneous=False)
    first = likelihood.total(st_theta, spatial_history)
    cache = likelihood.geometry_for(spatial_history)
    assert likelihood.total(st_theta, spatial_history) == first
    assert likelihood.geometry_for(spatial_history) is cache


@pytest.mark.parametrize("backend", ["nope", ""])
def test_an_unknown_backend_is_refused(st_model_1d, backend):
    with pytest.raises(ValueError, match="backend must be"):
        SpatioTemporalLogLikelihood(st_model_1d, backend=backend)


def test_the_spatial_spread_is_the_quadrature_error_on_a_masked_domain():
    """A hexagon does not fill its bounding box, so `S_i` really does vary.

    The variation *is* the quadrature error -- the rule resolves the boundary
    only to the width of a panel -- which makes `spatial_spread` a free accuracy
    diagnostic rather than only a switch. Measured at 16 nodes per axis: 2.4e-3.
    Above `rtol` the likelihood falls back to a mass per event, which is exact,
    and says so.
    """
    domain = hp.FundamentalDomain.hexagon(1.0)
    model = spatio_temporal_model(domain, n_quad=16)
    rng = np.random.default_rng(0)
    points = np.column_stack([domain.wrap(domain.sample_uniform(rng)) for _ in range(8)])
    history = History(np.arange(1.0, 9.0), points, 0.0, end=10.0)
    theta = np.array([0.4, 0.5, 1.5, 0.5])

    tolerant = SpatioTemporalLogLikelihood(model, backend="cached", rtol=1e-2)
    tolerant.total(theta, history)
    assert tolerant.homogeneous_used
    assert 1e-4 < tolerant.spatial_spread < 1e-2

    strict = SpatioTemporalLogLikelihood(model, backend="cached", rtol=1e-6)
    with pytest.warns(UserWarning, match="spatial mass varies"):
        fallback = strict.total(theta, history)
    assert not strict.homogeneous_used

    # The fallback is exact: it agrees with the hooks to rounding, while the
    # single-mass shortcut agrees only to the spread.
    hooks = SpatioTemporalLogLikelihood(model, backend="hooks").total(theta, history)
    assert fallback == pytest.approx(hooks, rel=1e-12)


def test_a_kernel_that_dips_negative_only_in_a_narrow_band_is_still_caught():
    """The precondition is about the kernel, not about the distances sampled.

    Under `homogeneous` the spatial masses are evaluated at a handful of events'
    distances, so a kernel that goes negative only in a narrow band could pass
    while breaking the separability identity everywhere that band falls. The
    check therefore probes the whole range of distances the geometry holds.
    """
    from hawkes_package.inference import Parameter, ParameterSpec

    class NarrowDip:
        """Gaussian, with an inhibitory notch 0.05 wide at distance 2."""

        @property
        def spec(self):
            return ParameterSpec((Parameter("sigma"),))

        def build(self, theta):
            sigma = float(np.asarray(theta).ravel()[0])
            return lambda d: (
                np.exp(-((np.asarray(d) / sigma) ** 2))
                - 0.4 * np.exp(-(((np.asarray(d) - 2.0) / 0.05) ** 2))
            )

        def mass(self, theta):
            return np.ones(np.atleast_2d(theta).shape[0])

        def min_scale(self, theta):
            return np.atleast_2d(theta)[:, 0]

    model = spatio_temporal_model(hp.Circle(), spatial=NarrowDip(), n_quad=32)
    rng = np.random.default_rng(1)
    points = rng.uniform(-np.pi, np.pi, size=(1, 6))
    history = History(np.arange(1.0, 7.0), points, 0.0, end=8.0)

    likelihood = SpatioTemporalLogLikelihood(model, backend="cached")
    with pytest.raises(ValueError, match="non-negative at every quadrature node"):
        likelihood.total(np.array([0.5, 0.5, 1.5, 0.5]), history)
