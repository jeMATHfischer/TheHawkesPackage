"""Tests for the random-walk Metropolis-Hastings sampler."""

import numpy as np
import pytest

from hawkes_package.mcmc import mcmc_sampler


def gaussian_density(x, mu=0.0, sigma=1.0):
    return float(np.exp(-0.5 * ((x[0] - mu) / sigma) ** 2))


def test_returns_point_in_domain():
    space = np.array([[-5.0, 5.0]])
    sample = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=0)
    assert sample.shape == (1,)
    assert -5.0 <= float(sample[0]) <= 5.0


def test_returns_bare_array_without_diagnostics():
    sample = mcmc_sampler(gaussian_density, np.array([[-5.0, 5.0]]), n_iter=200, seed=0)
    assert isinstance(sample, np.ndarray)


def test_acceptance_rate_reasonable():
    sample, rate = mcmc_sampler(
        gaussian_density,
        np.array([[-5.0, 5.0]]),
        n_iter=2000,
        burn_in=500,
        seed=1,
        return_diagnostics=True,
    )
    assert sample.shape == (1,)
    assert 0.1 < rate < 0.9, f"Acceptance rate {rate} out of expected range"


@pytest.mark.statistical
def test_sample_near_true_mean():
    """Over many independent chains the mean of the last states should be near 0."""
    space = np.array([[-10.0, 10.0]])
    samples = [
        float(
            mcmc_sampler(
                gaussian_density, space, n_iter=3000, burn_in=1000, proposal_std=1.5, seed=s
            )[0]
        )
        for s in range(50)
    ]
    mean = float(np.mean(samples))
    assert abs(mean) < 0.5, f"Sample mean {mean:.3f} too far from true mean 0"


def test_seed_reproducibility():
    space = np.array([[-5.0, 5.0]])
    a = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=7)
    b = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=7)
    np.testing.assert_array_equal(a, b)


def test_accepts_a_generator_and_advances_it():
    """Passing a Generator shares one stream, so consecutive draws differ."""
    gen = np.random.default_rng(3)
    space = np.array([[-5.0, 5.0]])
    a = mcmc_sampler(gaussian_density, space, n_iter=300, burn_in=50, seed=gen)
    b = mcmc_sampler(gaussian_density, space, n_iter=300, burn_in=50, seed=gen)
    assert not np.array_equal(a, b), "the generator's state must advance"


def test_two_dimensional_target():
    def density(x):
        return float(np.exp(-0.5 * (x[0] ** 2 + x[1] ** 2)))

    space = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    sample = mcmc_sampler(density, space, n_iter=1000, burn_in=200, seed=2)
    assert sample.shape == (2,)


def test_initialisation_retries_until_density_is_positive():
    """The chain must start somewhere with positive density, not just anywhere."""
    calls = {"n": 0}

    def narrow(x):
        # Non-zero only on a thin slice, so the uniform initial draw usually misses.
        calls["n"] += 1
        return 1.0 if -0.1 < x[0] < 0.1 else 0.0

    sample = mcmc_sampler(narrow, np.array([[-50.0, 50.0]]), n_iter=200, burn_in=50, seed=4)
    assert calls["n"] > 1, "expected the initialisation loop to retry"
    assert abs(float(sample[0])) < 0.1
