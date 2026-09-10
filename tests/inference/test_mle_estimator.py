"""`HawkesMLE`, the sklearn-shaped sibling of `HawkesEstimator`.

A sibling rather than a `method="mle"` mode, and the reason is what these tests
mostly check: the two classes must refuse the same inputs in the same words while
offering *different* things. A single class would carry a `diagnostics_` with no
cloud behind it, and those diagnostics exist to be read.

The shared refusals are the interesting half. Both estimators reject a target,
demand an observation window with a bare array, and refuse to evaluate an
intensity past the window -- and since 1.0.0 they do that through the same
module-level helpers, so the explanations cannot drift apart.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior,
    ExponentialLogLikelihood,
    HawkesMLE,
    History,
    IndependentPrior,
    LogNormal,
    exponential_model,
    fit_smc,
)

TRUTH = np.array([1.0, 0.5, 2.0])
START = [1.0, 0.3, 1.0]


@pytest.fixture(scope="module")
def history():
    process = hp.ExponentialHawkes(TRUTH, rng=0)
    process.simulate(300)
    return History.from_simulation(process)


@pytest.fixture(scope="module")
def fitted(history):
    return HawkesMLE("exponential", start=START).fit(history)


def test_it_fits_and_reports_the_estimate(fitted):
    assert fitted.theta_.shape == (3,)
    assert fitted.fit_.converged
    assert set(fitted.fit_.named()) == {"mu", "alpha", "beta"}


def test_a_target_is_refused_rather_than_ignored(history):
    """The same refusal `HawkesEstimator` makes, through the same helper."""
    with pytest.raises(ValueError, match="y"):
        HawkesMLE("exponential", start=START).fit(history, y=np.zeros(3))


def test_an_array_without_a_window_is_refused(history):
    """`end=times[-1]` is the tempting default and it biases `mu` upward."""
    with pytest.raises(ValueError, match="end="):
        HawkesMLE("exponential", start=START).fit(history.times)


def test_an_array_with_a_window_is_accepted(history):
    estimator = HawkesMLE("exponential", start=START).fit(history.times, end=history.end)
    assert estimator.theta_.shape == (3,)


def test_predict_is_the_plug_in_and_says_so(fitted, history):
    """At the estimate, where `HawkesEstimator.predict` averages over a posterior.

    Not a detail: the intensity is convex in the decay rate, so by Jensen the
    plug-in sits systematically below the posterior mean wherever the posterior
    has width. Both are correct answers to different questions, and this class
    answers the one with a point estimate in it.
    """
    values = fitted.predict(history.times[:20])
    assert values.shape == (20,)
    assert np.all(values > 0.0)

    process = fitted.model_(fitted.theta_)
    process.events = history.times
    expected = [float(process._conditional_intensity(float(t))) for t in history.times[:20]]
    np.testing.assert_allclose(values, expected, rtol=1e-12)


def test_predict_refuses_times_outside_the_window(fitted, history):
    """Where the intensity from the observed record answers a different question."""
    with pytest.raises(ValueError, match="observation window"):
        fitted.predict(np.array([history.end + 1.0]))


def test_predict_before_a_fit_is_refused():
    with pytest.raises(ValueError, match="needs a fitted estimator"):
        HawkesMLE("exponential", start=START).predict(np.array([1.0]))


def test_score_is_a_later_block(fitted, history):
    """Scoring the data a fit was made on measures its appetite, not its quality."""
    process = hp.ExponentialHawkes(TRUTH, rng=1)
    process.events = history.times
    process.n_simulated = history.times.size
    process.simulate_until(history.end + 20.0, start=history.end)
    fresh = process.events[process.events > history.end]

    value = fitted.score(fresh, end=history.end + 20.0)
    assert np.isfinite(value)


def test_score_refuses_events_inside_the_fitted_window(fitted, history):
    with pytest.raises(ValueError, match="after the fitted window"):
        fitted.score(history.times[:5], end=history.end + 1.0)


def test_score_refuses_a_window_that_does_not_advance(fitted, history):
    with pytest.raises(ValueError, match="does not advance"):
        fitted.score(np.empty(0), end=history.end)


def test_get_params_round_trips(fitted):
    """The scikit-learn contract: every constructor argument, by name."""
    params = fitted.get_params()
    rebuilt = HawkesMLE(**params)
    assert rebuilt.get_params() == params


def test_set_params_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="no parameter"):
        HawkesMLE("exponential", start=START).set_params(n_particles=64)


def test_the_repr_shows_the_estimate_once_there_is_one(fitted):
    assert "fitted" in repr(fitted)
    assert "mu=" in repr(fitted)
    assert "fitted" not in repr(HawkesMLE("exponential", start=START))


def test_the_profile_interval_is_reachable_from_the_estimator(fitted):
    lower, upper = fitted.profile_interval_("mu")
    assert lower < fitted.theta_[0] < upper


def test_the_warm_start_proposal_is_reachable_from_the_estimator(fitted):
    prior = fitted.warm_start_proposal_(width=0.4)
    draws = prior.sample(64, np.random.default_rng(0))
    assert draws.shape == (64, 3)


@pytest.mark.statistical
def test_the_warm_start_does_not_move_the_posterior(history, fitted):
    """The distinction between a proposal and a prior, measured.

    Handed to ``proposal=`` the warm distribution is corrected for by weight, so
    the target stays the posterior under the vague prior. Handed to ``prior=``
    it *is* the model, and a tight distribution centred on the maximum pulls the
    answer there: the four-block mean moves from (0.95, 0.21, 1.52) to
    (1.07, 0.30, 2.39) and the log evidence rises by about two nats, because
    that is a different model and it fits better.

    The log evidence is what this asserts on. It is the quantity that changes
    when the target changes, and it does not move when only the starting point
    does.
    """
    model = exponential_model()
    vague = ConstrainedPrior(
        IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )
    likelihood = ExponentialLogLikelihood(model)
    warm = ConstrainedPrior(fitted.warm_start_proposal_(width=0.5), model.support)

    cold = fit_smc(likelihood, vague, history, blocks=4, n_particles=128, rng=0)
    as_proposal = fit_smc(
        likelihood, vague, history, blocks=4, n_particles=128, rng=0, proposal=warm
    )
    as_prior = fit_smc(likelihood, warm, history, blocks=4, n_particles=128, rng=0)

    evidence = cold.diagnostics.log_evidence
    assert as_proposal.diagnostics.log_evidence == pytest.approx(evidence, abs=1.0), (
        "using the warm distribution as a proposal changed the evidence, so it "
        "changed the target it was supposed to leave alone"
    )
    assert as_prior.diagnostics.log_evidence > evidence + 1.0, (
        "using it as a prior should be a visibly different model; if this stops "
        "being true the warning in warm_start_proposal is over-stated"
    )


@pytest.mark.statistical
def test_the_warm_start_arrives_sooner(history, fitted):
    """What the proposal buys, in the only place it can buy anything: early blocks.

    Against an eight-block, 512-particle reference on the same data, the warm
    cloud's mean after **one** block is 0.284 away where the cold one is 0.506,
    and after two blocks 0.073 against 0.438. By four blocks the cold chain has
    caught up and the warm one is carrying a little importance-weight variance
    instead -- so this asserts on the early blocks, which is where the claim is.
    """
    model = exponential_model()
    vague = ConstrainedPrior(
        IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )
    likelihood = ExponentialLogLikelihood(model)
    warm = ConstrainedPrior(fitted.warm_start_proposal_(width=0.5), model.support)

    reference = fit_smc(likelihood, vague, history, blocks=8, n_particles=512, rng=7)
    target = reference.cloud.mean()

    def gap(**kwargs):
        smc = fit_smc(likelihood, vague, history, blocks=2, n_particles=128, rng=0, **kwargs)
        return float(np.linalg.norm(smc.cloud.mean() - target))

    assert gap(proposal=warm) < 0.5 * gap()
