import numpy as np
import pytest
from TheHawkesPackage.MCMC_sampler import mcmc_sampler


def gaussian_density(x, mu=0.0, sigma=1.0):
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def test_returns_point_in_domain():
    space = np.array([[-5.0, 5.0]])
    sample = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=0)
    assert -5.0 <= float(sample) <= 5.0


def test_acceptance_rate_reasonable():
    space = np.array([[-5.0, 5.0]])
    sample, rate = mcmc_sampler(gaussian_density, space, n_iter=2000, burn_in=500,
                                seed=1, return_diagnostics=True)
    assert 0.1 < rate < 0.9, f"Acceptance rate {rate} out of expected range"


def test_sample_near_true_mean():
    """Over many independent chains, the mean of last samples should be near 0."""
    rng = np.random.default_rng(42)
    space = np.array([[-10.0, 10.0]])
    samples = []
    for seed in range(50):
        s = mcmc_sampler(gaussian_density, space, n_iter=3000, burn_in=1000,
                         proposal_std=1.5, seed=int(seed))
        samples.append(float(s))
    mean = np.mean(samples)
    assert abs(mean) < 0.5, f"Sample mean {mean:.3f} too far from true mean 0"


def test_seed_reproducibility():
    space = np.array([[-5.0, 5.0]])
    s1 = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=7)
    s2 = mcmc_sampler(gaussian_density, space, n_iter=500, burn_in=100, seed=7)
    np.testing.assert_array_equal(s1, s2)


def test_2d_target():
    def bivariate(x):
        return float(np.exp(-0.5 * (x[0]**2 + x[1]**2)))

    space = np.array([[-4.0, 4.0], [-4.0, 4.0]])
    sample = mcmc_sampler(bivariate, space, n_iter=1000, burn_in=200, seed=3)
    assert sample.shape == (2,)
