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
