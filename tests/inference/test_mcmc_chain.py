"""The adaptive Metropolis chain used for reference posteriors.

Checked against targets whose answer is known in closed form, because its whole
job is to be an *independent* second opinion on the sequential sampler. A chain
verified only against that sampler would verify nothing.
"""

import math

import numpy as np
import pytest

from hawkes_package.inference import ExponentialLogLikelihood, batch_posterior, metropolis_chain


def gaussian_target(mean, covariance):
    inverse = np.linalg.inv(covariance)

    def log_density(z):
        centred = np.asarray(z, dtype=float) - mean
        return float(-0.5 * centred @ inverse @ centred)

    return log_density


def test_the_chain_recovers_a_correlated_gaussian():
    """Mean and covariance within four standard errors of the truth."""
    mean = np.array([1.5, -2.0])
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    result = metropolis_chain(
        gaussian_target(mean, covariance),
        np.zeros(2),
        n_samples=20_000,
        burn_in=5_000,
        rng=17,
    )
    assert result.samples.shape == (20_000, 2)
    # The chain is autocorrelated, so the effective sample size is well below
    # the nominal one; a factor of ten is generous and still tight enough that a
    # chain sampling the wrong distribution fails.
    effective = result.samples.shape[0] / 10
    tolerance = 4 * np.sqrt(np.diag(covariance) / effective)
    assert np.all(np.abs(result.samples.mean(axis=0) - mean) < tolerance)
    empirical = np.cov(result.samples.T)
    np.testing.assert_allclose(empirical, covariance, rtol=0.25, atol=0.15)


def test_the_acceptance_rate_lands_in_the_usable_band():
    """Outside [0.1, 0.6] the samples are far more correlated than their count."""
    mean = np.array([0.0, 0.0, 0.0])
    covariance = np.diag([1.0, 100.0, 0.01])
    result = metropolis_chain(
        gaussian_target(mean, covariance), np.zeros(3), n_samples=8000, burn_in=4000, rng=3
    )
    assert 0.1 <= result.acceptance <= 0.6


def test_the_adaptation_fits_a_badly_scaled_target():
    """A scalar step cannot serve coordinates whose scales differ by 10^4.

    The proposal is shaped from the burn-in's own scatter for exactly this: a
    Hawkes posterior over (mu, alpha, beta) routinely spans that much.
    """
    covariance = np.diag([1e-4, 1.0, 1e4])
    result = metropolis_chain(
        gaussian_target(np.zeros(3), covariance),
        np.zeros(3),
        n_samples=12_000,
        burn_in=8_000,
        rng=5,
    )
    spread = result.samples.std(axis=0)
    np.testing.assert_allclose(spread, np.sqrt(np.diag(covariance)), rtol=0.4)


def test_thinning_keeps_the_requested_count():
    result = metropolis_chain(
        gaussian_target(np.zeros(2), np.eye(2)),
        np.zeros(2),
        n_samples=500,
        burn_in=500,
        thin=3,
        rng=1,
        warn_acceptance=False,
    )
    assert result.samples.shape == (500, 2)


def test_a_starting_point_outside_the_support_is_refused():
    """A chain started at -inf accepts its first proposal whatever it is."""

    def log_density(z):
        return -math.inf if z[0] < 0 else -0.5 * float(z @ z)

    with pytest.raises(ValueError, match="starting point"):
        metropolis_chain(log_density, np.array([-1.0]), n_samples=100, rng=0)


def test_a_nan_target_is_named_rather_than_stalling():
    """`nan` compares false, so it would be rejected forever with nothing amiss."""
    calls = {"n": 0}

    def log_density(z):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else math.nan

    with pytest.raises(ValueError, match="nan"):
        metropolis_chain(log_density, np.zeros(1), n_samples=50, rng=0)


def test_a_crawling_chain_warns():
    """Guard the guard: a proposal far too small must be reported, not hidden."""
    with pytest.warns(UserWarning, match="acceptance rate"):
        metropolis_chain(
            gaussian_target(np.zeros(2), np.eye(2)),
            np.zeros(2),
            n_samples=400,
            burn_in=0,
            scale=1e-6,
            rng=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "at least 1"),
        ({"thin": 0}, "at least 1"),
        ({"burn_in": -1}, "non-negative"),
    ],
)
def test_bad_chain_arguments_are_refused(kwargs, match):
    kwargs.setdefault("n_samples", 10)
    with pytest.raises(ValueError, match=match):
        metropolis_chain(gaussian_target(np.zeros(1), np.eye(1)), np.zeros(1), rng=0, **kwargs)


# ---------------------------------------------------------------------------
# The batch posterior
# ---------------------------------------------------------------------------


def test_batch_posterior_returns_constrained_samples(history, exp_model, exp_prior):
    result = batch_posterior(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        history,
        n_samples=800,
        burn_in=800,
        rng=0,
        warn_acceptance=False,
    )
    assert result.samples.shape == (800, 3)
    assert np.all(result.samples > 0), "the samples come back on the model's own scale"
    assert np.all(exp_model.support(result.samples))


def test_batch_posterior_carries_the_jacobian(history, exp_model, exp_prior):
    """Without it the two samplers disagree by a factor of theta per coordinate.

    Checked directly: the target the chain is handed must equal the prior plus
    the likelihood plus the log Jacobian, and the recorded density lets that be
    read back off a returned sample.
    """
    likelihood = ExponentialLogLikelihood(exp_model)
    result = batch_posterior(
        likelihood,
        exp_prior,
        history,
        n_samples=200,
        burn_in=400,
        rng=1,
        warn_acceptance=False,
    )
    theta = result.samples[-1]
    z = exp_model.spec.to_unconstrained(theta)
    expected = (
        float(np.atleast_1d(exp_prior.log_pdf(theta))[0])
        + likelihood.total(theta, history)
        + float(exp_model.spec.log_abs_det_jacobian(z))
    )
    assert result.log_density[-1] == pytest.approx(expected, rel=1e-9)


def test_an_explicit_start_is_honoured(history, exp_model, exp_prior):
    result = batch_posterior(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        history,
        n_samples=100,
        burn_in=100,
        initial=np.array([2.0, 0.5, 1.0]),
        rng=2,
        warn_acceptance=False,
    )
    assert result.samples.shape == (100, 3)
