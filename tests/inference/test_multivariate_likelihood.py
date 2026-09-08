"""The multivariate log-likelihood, and the term that makes it not the temporal one.

The sum runs over the intensity of the type each event carries; the integral
runs over the total across types. Using the total in both places is the one
mistake available here, and it is quantified below rather than described: it
adds 265 nats over 400 events, always in the same direction.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    History,
    MultivariateLogLikelihood,
    TemporalLogLikelihood,
    exponential_model,
    multivariate_model,
)

THETA = np.array([0.4, 0.2, 0.3, 0.1, 0.5, 0.2, 2.0])


@pytest.fixture
def model():
    return multivariate_model(2)


@pytest.fixture
def history(model):
    process = model(THETA, rng=3)
    process.simulate(200)
    return History.from_multivariate_events(
        process.events, n_types=2, end=float(process.events[0, -1])
    )


@pytest.mark.parametrize(
    "theta", [np.array([1.0, 0.5, 2.0]), np.array([0.8, 0.3, 1.5]), np.array([1.5, 0.9, 3.0])]
)
def test_one_type_equals_the_univariate_likelihood_exactly(theta):
    """At one type this must *be* the temporal likelihood, not merely agree with it.

    Measured as exactly zero relative difference, not to a tolerance: both
    compute ``mu + alpha * sum(exp(...))`` with the amplitude factored out of the
    sum, and both integrate through the same panels. A tolerance here would hide
    a real divergence behind a number nobody chose.
    """
    reference = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=5)
    reference.simulate(150)
    end = float(reference.events[-1])

    univariate = History.from_events(reference.events, end=end)
    multivariate = History.from_events(reference.events, end=end, types=np.zeros(150), n_types=1)

    expected = TemporalLogLikelihood(exponential_model()).total(theta, univariate)
    got = MultivariateLogLikelihood(multivariate_model(1)).total(theta, multivariate)
    assert got == expected


def test_the_log_sum_uses_the_component_not_the_total(model, history):
    """The failure this class exists to prevent, measured.

    Reading the total where the accepted component belongs adds
    ``log(Lambda / lambda_c)`` at every event, which is strictly positive.
    Measured at +265 nats over 400 events -- so the fit would report far more
    excitation than the data supports, and report it converged.
    """
    process = model(THETA)
    process.events = history.as_process_events()

    correct = sum(
        math.log(process._component_intensities(float(t))[k])
        for t, k in zip(history.times, history.types, strict=True)
    )
    inflated = sum(math.log(process._conditional_intensity(float(t))) for t in history.times)

    assert inflated > correct
    assert inflated - correct > 50.0, "the two must be far apart, or this test proves nothing"

    likelihood = MultivariateLogLikelihood(model)
    compensator = likelihood.compensator(THETA, history, np.array([history.end]))[0]
    assert likelihood.total(THETA, history) == pytest.approx(correct - compensator, rel=1e-9)


@pytest.mark.parametrize("blocks", [1, 2, 3, 7])
def test_extending_in_blocks_equals_one_shot(model, history, blocks):
    """Blocking moves the panel edges, so this is quadrature-close, not exact.

    Measured over six seeds: worst relative difference 1.2e-8. The threshold is
    an order of magnitude above that.
    """
    likelihood = MultivariateLogLikelihood(model)
    state = likelihood.initial_state(history.start)
    for upto in np.linspace(history.start, history.end, blocks + 1)[1:]:
        state, _ = likelihood.extend(state, THETA, history, float(upto))

    assert state.log_lik == pytest.approx(likelihood.total(THETA, history), rel=1e-7)
    assert state.n_events == history.n_events


def test_the_compensator_is_non_decreasing(model, history):
    likelihood = MultivariateLogLikelihood(model)
    values = likelihood.compensator(THETA, history, np.linspace(history.start, history.end, 200))
    assert np.all(np.diff(values) >= 0.0)
    assert values[0] >= 0.0


def test_the_compensator_refuses_unsorted_times(model, history):
    likelihood = MultivariateLogLikelihood(model)
    with pytest.raises(ValueError, match="sorted"):
        likelihood.compensator(THETA, history, np.array([2.0, 1.0]))


def test_an_untyped_history_is_refused(model):
    """Every event would land in component 0 -- a different model, silently."""
    likelihood = MultivariateLogLikelihood(model)
    untyped = History(np.array([0.4, 1.1, 2.9]), None, 0.0, end=4.0)
    with pytest.raises(ValueError, match="carrying event types"):
        likelihood.total(THETA, untyped)


def test_a_type_count_mismatch_is_refused(model):
    """Dropping or inventing a component takes its whole row and column with it."""
    likelihood = MultivariateLogLikelihood(model)
    wrong = History(np.array([0.4, 1.1, 2.9]), None, 0.0, 4.0, np.array([0, 1, 0]), 3)
    with pytest.raises(ValueError, match="n_types=3 but the model has 2"):
        likelihood.total(THETA, wrong)


def test_a_univariate_model_is_refused():
    likelihood_type = MultivariateLogLikelihood
    with pytest.raises(ValueError, match="is for multivariate models"):
        likelihood_type(exponential_model())


def test_a_zero_component_intensity_gives_minus_infinity(model, history, monkeypatch):
    """A likelihood of zero, not an error: the parameter cannot have produced this.

    Unreachable through the model as it stands, and deliberately so -- every
    coordinate lives on an *open* positive interval, so ``mu_i > 0`` and the
    component intensity is bounded below by it. The branch is what stops a
    future background family that can reach zero, or a nonlinearity that
    saturates to it, from turning "impossible data" into a ``math domain
    error`` from inside a rejuvenation move. So it is exercised directly rather
    than through a parameter that cannot be constructed.
    """
    likelihood = MultivariateLogLikelihood(model)
    build = likelihood._process

    def zeroed(theta, hist, upto):
        process = build(theta, hist, upto)
        process._component_intensities = lambda t: np.zeros(2)
        return process

    monkeypatch.setattr(likelihood, "_process", zeroed)
    assert likelihood.total(THETA, history) == -math.inf


def test_the_increment_is_what_the_state_gained(model, history):
    likelihood = MultivariateLogLikelihood(model)
    state = likelihood.initial_state(history.start)
    running = 0.0
    for upto in np.linspace(history.start, history.end, 4)[1:]:
        state, increment = likelihood.extend(state, THETA, history, float(upto))
        running += increment
        assert state.log_lik == pytest.approx(running, rel=1e-12)


# ---------------------------------------------------------------------------
# The O(n d) closed form
# ---------------------------------------------------------------------------


def exact(model):
    from hawkes_package.inference import MultivariateExponentialLogLikelihood

    return MultivariateExponentialLogLikelihood(model)


@pytest.mark.parametrize("n_types", [1, 2, 3])
def test_the_recursion_agrees_with_the_general_path(n_types):
    """Two independent routes to one number, over a sweep of parameters.

    The general path integrates the total intensity by quadrature; this one
    writes the integral down. So where they differ, the closed form is the
    accurate one and the gap measures the quadrature -- which is why the
    threshold is set from the measurement rather than from what the recursion
    deserves. Worst relative difference over twenty parameter vectors: 4.6e-6 at
    one type, 5.6e-9 at two, 4.8e-11 at three. The one-type case is the hardest
    integrand of the three, not the loosest arithmetic: a single component
    carrying all the excitation spikes more sharply between events.
    """
    model = multivariate_model(n_types)
    d = n_types
    truth = np.concatenate([np.full(d, 0.4), np.full(d * d, 0.9 / d), [2.0]])

    process = model(truth, rng=1)
    process.simulate(300)
    history = History.from_multivariate_events(
        process.events, n_types=d, end=float(process.events[0, -1])
    )

    general, closed = MultivariateLogLikelihood(model), exact(model)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        theta = truth * rng.uniform(0.7, 1.3, size=truth.size)
        if not bool(model.support(theta)):
            continue
        checked += 1
        assert closed.total(theta, history) == pytest.approx(
            general.total(theta, history), rel=1e-5
        )
    assert checked >= 10, "the sweep should stay inside the support most of the time"


@pytest.mark.parametrize(
    "theta", [np.array([1.0, 0.5, 2.0]), np.array([0.8, 0.3, 1.5]), np.array([1.5, 0.9, 3.0])]
)
def test_one_type_equals_the_univariate_closed_form_exactly(theta):
    """Measured at exactly zero relative difference, log-likelihood and compensator.

    The scalar recursion and the matrix one do the same arithmetic in the same
    order at ``d = 1``, so this is an equality rather than a tolerance.
    """
    from hawkes_package.inference import ExponentialLogLikelihood

    reference = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=5)
    reference.simulate(300)
    end = float(reference.events[-1])
    univariate = History.from_events(reference.events, end=end)
    multivariate = History.from_events(reference.events, end=end, types=np.zeros(300), n_types=1)

    scalar = ExponentialLogLikelihood(exponential_model())
    matrix = exact(multivariate_model(1))
    assert matrix.total(theta, multivariate) == scalar.total(theta, univariate)

    query = np.linspace(0.0, end, 200)
    np.testing.assert_array_equal(
        matrix.compensator(theta, multivariate, query),
        scalar.compensator(theta, univariate, query),
    )


def test_the_carry_is_one_float_per_type(model, history):
    """``O(n d)``, not ``O(n d**2)`` -- the carry is indexed by source only.

    That is what the single shared decay rate buys. Per-pair rates would need
    one entry per ordered pair and the recursion would cost ``d**2`` per step.
    """
    likelihood = exact(model)
    state = likelihood.initial_state(history.start)
    assert len(state.carry) == 2

    state, _ = likelihood.extend(state, THETA, history, history.end)
    assert len(state.carry) == 2
    assert all(value >= 0.0 for value in state.carry)


@pytest.mark.parametrize("blocks", [1, 2, 3, 7])
def test_the_recursion_extends_in_blocks(model, history, blocks):
    """Blocking must be exact here: the recursion carries its own state."""
    likelihood = exact(model)
    state = likelihood.initial_state(history.start)
    for upto in np.linspace(history.start, history.end, blocks + 1)[1:]:
        state, _ = likelihood.extend(state, THETA, history, float(upto))
    assert state.log_lik == pytest.approx(likelihood.total(THETA, history), rel=1e-12)
    assert state.n_events == history.n_events


def test_the_recursion_compensator_matches_the_general_one(model, history):
    likelihood, general = exact(model), MultivariateLogLikelihood(model)
    query = np.linspace(history.start, history.end, 60)
    np.testing.assert_allclose(
        likelihood.compensator(THETA, history, query),
        general.compensator(THETA, history, query),
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (exponential_model, "implements the closed form for multivariate_model"),
        (lambda: multivariate_model(2), None),
    ],
)
def test_the_closed_form_refuses_a_model_it_does_not_implement(factory, match):
    from hawkes_package.inference import MultivariateExponentialLogLikelihood

    model = factory()
    if match is None:
        assert MultivariateExponentialLogLikelihood(model) is not None
        return
    with pytest.raises(ValueError, match=match):
        MultivariateExponentialLogLikelihood(model)


def test_the_closed_form_refuses_a_saturating_nonlinearity():
    """It is the *linear* closed form; softplus is not that model."""
    from hawkes_package.inference import (
        MultivariateExponentialLogLikelihood,
        SoftPlusNonlinearity,
    )

    model = multivariate_model(2, nonlinearity=SoftPlusNonlinearity())
    with pytest.raises(ValueError, match=r"is the \*linear\* closed form"):
        MultivariateExponentialLogLikelihood(model)


@pytest.mark.parametrize(
    "factory",
    [MultivariateLogLikelihood, lambda m: exact(m)],
    ids=["general", "closed-form"],
)
def test_a_history_carrying_locations_is_refused(model, factory):
    """Space and event type do not combine, and the boundary is stated here.

    `History.from_multivariate_events` parses a ``(ndim + 2, n)`` record happily
    -- the layout is well defined and costs nothing to describe -- but nothing in
    the package produces one and no likelihood consumes one. Without this guard
    it would be turned away three frames deeper by `_EventBuffer.replace`,
    complaining about a row count rather than about the thing that is missing.
    """
    times = np.array([0.4, 1.1, 2.9])
    points = np.array([[0.1, 0.2, 0.3]])
    types = np.array([0.0, 1.0, 0.0])
    record = np.vstack([times[None, :], points, types[None, :]])
    history = History.from_multivariate_events(record, n_types=2, end=4.0)

    with pytest.raises(ValueError, match="temporal only"):
        factory(model).total(THETA, history)
