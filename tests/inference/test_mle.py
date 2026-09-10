"""Maximum likelihood, and the three ways it goes quietly wrong.

The package's position is that the sequential machinery is the recommended path.
This exists so the library can be *compared* with others on the terms a reviewer
will use, and so a cloud can be started at the mode.

What is checked here, in order of how badly it would hurt to get wrong:

* **the optimum is the optimum** -- against an independent grid search on the
  exact exponential likelihood, which is the test that proves the objective and
  the transform are wired together correctly;
* **the Jacobian is absent**, so the answer is a maximum likelihood estimate and
  not a MAP estimate under a flat prior on the unconstrained scale, which is a
  different number with the same name;
* **failure is loud** -- a non-converged optimiser warns rather than returning
  its last iterate, and an optimum outside the support is refused.
"""

import math
import warnings

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    TemporalLogLikelihood,
    exponential_model,
    fit_mle,
    monotone_model,
    profile_interval,
    warm_start_proposal,
)
from hawkes_package.inference import _compensator as compensator

TRUTH = np.array([1.0, 0.5, 2.0])
START = np.array([1.0, 0.3, 1.0])


@pytest.fixture(scope="module")
def data():
    """400 events from the exponential model, and its exact likelihood."""
    process = hp.ExponentialHawkes(TRUTH, rng=0)
    process.simulate(400)
    history = History.from_simulation(process)
    model = exponential_model()
    return model, ExponentialLogLikelihood(model), history


def test_the_optimum_agrees_with_a_grid_search(data):
    """The test that proves the objective and the transform are wired up.

    A coarse grid is an independent optimum: no transform, no optimiser, no
    support handling. On this history it puts the maximum at
    ``(1.05, 0.25, 1.60)`` with a log-likelihood of -312.3656, and `fit_mle`
    finds -312.3526 at ``(1.035, 0.254, 1.571)`` -- inside the grid's own spacing
    and very slightly higher, which is the only direction a finer search may go.
    """
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)

    best, argmax = -math.inf, None
    for mu in np.linspace(0.6, 1.6, 21):
        for alpha in np.linspace(0.05, 1.2, 24):
            for beta in np.linspace(0.4, 4.0, 25):
                if alpha >= beta:
                    continue
                value = likelihood.total(np.array([mu, alpha, beta]), history)
                if value > best:
                    best, argmax = value, np.array([mu, alpha, beta])

    assert fit.log_likelihood >= best, "the optimiser found a worse point than a coarse grid"
    assert fit.log_likelihood - best < 0.1, "the two optima are not the same optimum"
    np.testing.assert_allclose(fit.theta, argmax, rtol=0.15)


def test_the_maximum_beats_the_truth_on_this_sample(data):
    """A maximum likelihood estimate maximises the *likelihood*, not the error.

    Worth asserting because it looks like a failure: on 400 events the estimate
    is (1.035, 0.254, 1.571) against a truth of (1.0, 0.5, 2.0), with a branching
    ratio of 0.162 against 0.25. The estimate is nonetheless correct -- it has
    the higher log-likelihood, and `alpha` and `beta` trade off along a ridge
    that 400 events do not resolve.
    """
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    assert fit.log_likelihood > likelihood.total(TRUTH, history)


def test_the_jacobian_is_not_included(data):
    """Otherwise this would be a MAP estimate under a flat prior on the wrong scale.

    The distinction is invisible in the result and visible in the arithmetic, so
    it is checked by arithmetic: at the reported optimum the *likelihood* is
    maximal, while the likelihood-plus-Jacobian is not -- the two disagree, which
    is what says the correction was left out on purpose.
    """
    model, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    spec = model.spec

    def with_jacobian(theta):
        z = spec.to_unconstrained(np.atleast_2d(theta))
        return float(likelihood.total(theta, history)) + float(
            np.asarray(spec.log_abs_det_jacobian(z)).ravel()[0]
        )

    # Step along `beta`, the coordinate whose transform has the steepest
    # log-derivative here, and show the two objectives prefer different points.
    grid = fit.theta[2] * np.linspace(0.6, 1.6, 41)
    plain, penalised = [], []
    for beta in grid:
        theta = fit.theta.copy()
        theta[2] = beta
        plain.append(likelihood.total(theta, history))
        penalised.append(with_jacobian(theta))

    assert grid[int(np.argmax(plain))] == pytest.approx(fit.theta[2], rel=0.05)
    assert grid[int(np.argmax(penalised))] != pytest.approx(fit.theta[2], rel=0.05)


def test_a_start_outside_the_support_is_refused(data):
    """Rather than becoming an infinity the optimiser cannot escape from."""
    _, likelihood, history = data
    with pytest.raises(ValueError, match="outside the model's support"):
        fit_mle(likelihood, history, np.array([1.0, 3.0, 1.0]))


def test_a_start_of_the_wrong_length_is_refused(data):
    _, likelihood, history = data
    with pytest.raises(ValueError, match="3"):
        fit_mle(likelihood, history, np.array([1.0, 0.3]))


def test_not_converging_warns_rather_than_returning_quietly(data):
    """Silently returning the last iterate is how a failed fit gets published.

    The suite runs with ``filterwarnings = ["error"]``, so this warning is a
    failure in any careful downstream suite too -- which is the point of making
    it a warning rather than a flag nobody reads.
    """
    _, likelihood, history = data
    with pytest.warns(UserWarning, match="did not converge"):
        fit = fit_mle(likelihood, history, START, max_iterations=3)
    assert not fit.converged
    assert "iterations" in fit.message.lower() or "maximum" in fit.message.lower()


def test_the_resolution_check_runs_at_the_optimum(monkeypatch):
    """An unconstrained maximiser has no prior to protect it from a bad compensator.

    The check is a once-per-likelihood affair by design, so left alone it fires
    at whatever parameters were evaluated first -- the *starting* ones, whose
    integrand is not the one the answer was read off. `fit_mle` resets it and
    evaluates once more at the optimum, so it runs exactly twice.
    """
    process = hp.MonotoneKernelHawkes(
        lambda s: 0.5 * np.exp(-1.0 * np.asarray(s, dtype=float)), rng=1
    )
    process.simulate(150)
    history = History.from_simulation(process)
    model = monotone_model()
    likelihood = TemporalLogLikelihood(model, check=True)

    calls = []
    original = compensator.check_resolution

    def spy(fn, edges, **kwargs):
        calls.append(len(np.asarray(edges)))
        return original(fn, edges, **kwargs)

    monkeypatch.setattr(compensator, "check_resolution", spy)
    fit_mle(likelihood, history, np.array([0.5, 0.4, 1.0]))
    assert len(calls) == 2, f"the check ran {len(calls)} time(s), not once per fit end"


def test_the_recheck_can_be_switched_off(monkeypatch):
    """It doubles a quadrature pass, so a caller fitting in a loop may decline it."""
    process = hp.MonotoneKernelHawkes(
        lambda s: 0.5 * np.exp(-1.0 * np.asarray(s, dtype=float)), rng=1
    )
    process.simulate(150)
    history = History.from_simulation(process)
    likelihood = TemporalLogLikelihood(monotone_model(), check=True)

    calls = []
    original = compensator.check_resolution
    monkeypatch.setattr(
        compensator,
        "check_resolution",
        lambda fn, edges, **kw: (calls.append(1), original(fn, edges, **kw))[1],
    )
    fit_mle(likelihood, history, np.array([0.5, 0.4, 1.0]), recheck_resolution=False)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Profile intervals
# ---------------------------------------------------------------------------


def test_the_profile_interval_brackets_the_estimate(data):
    """And covers the truth, which at this sample size it must for all three."""
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    for index, name in enumerate(fit.spec.names):
        lower, upper = profile_interval(likelihood, history, fit, name)
        assert lower < fit.theta[index] < upper, f"{name}: the interval excludes the estimate"
        assert lower <= TRUTH[index] <= upper, f"{name}: the interval misses the truth"


def test_an_unbounded_coordinate_reports_the_search_boundary(data):
    """Which is information, not a failure.

    `alpha`'s lower profile at 400 events drops only 0.85 by the time `alpha` has
    fallen a hundredfold -- `beta` follows it down and the branching ratio, which
    is what the data pins, barely moves. An interval that stopped at some
    plausible-looking number instead would be inventing a bound the data does not
    supply.
    """
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    lower, _ = profile_interval(likelihood, history, fit, "alpha", span=0.99)
    assert lower == pytest.approx(fit.theta[1] * 0.01, rel=1e-6)


def test_a_wider_level_gives_a_wider_interval(data):
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    narrow = profile_interval(likelihood, history, fit, "mu", level=0.90)
    wide = profile_interval(likelihood, history, fit, "mu", level=0.99)
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


def test_an_unknown_level_is_refused(data):
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    with pytest.raises(ValueError, match="level must be one of"):
        profile_interval(likelihood, history, fit, "mu", level=0.5)


# ---------------------------------------------------------------------------
# Warm starting
# ---------------------------------------------------------------------------


def test_the_warm_start_proposal_is_centred_on_the_mode(data):
    """On the *unconstrained* scale, which is where the sampler's proposals live."""
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    prior = warm_start_proposal(fit, width=0.3)

    draws = prior.sample(4000, np.random.default_rng(0))
    assert draws.shape == (4000, 3)
    # The median rather than the mean: the transform is nonlinear, so the mode
    # maps to the median of the draws and not to their average.
    np.testing.assert_allclose(np.median(draws, axis=0), fit.theta, rtol=0.05)


def test_the_warm_start_proposal_has_a_finite_density_at_the_mode(data):
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    prior = warm_start_proposal(fit)
    density = prior.log_pdf(np.atleast_2d(fit.theta))
    assert density.shape == (1,)
    assert np.isfinite(density).all()


def test_warm_starting_from_a_failed_fit_warns(data):
    """The centre would be the optimiser's last iterate, which is not a maximum."""
    _, likelihood, history = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_mle(likelihood, history, START, max_iterations=3)
    with pytest.warns(UserWarning, match="did not converge"):
        warm_start_proposal(fit)


def test_the_fit_reports_itself_by_name(data):
    _, likelihood, history = data
    fit = fit_mle(likelihood, history, START)
    assert set(fit.named()) == {"mu", "alpha", "beta"}
    assert "mu=" in repr(fit)
    assert "converged" in repr(fit)
