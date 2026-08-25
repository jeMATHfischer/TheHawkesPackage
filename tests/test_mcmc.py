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
    sample = mcmc_sampler(gaussian_density, np.array([[-5.0, 5.0]]), n_iter=200, burn_in=50, seed=0)
    assert isinstance(sample, np.ndarray)


def test_burn_in_must_be_shorter_than_the_chain():
    """burn_in >= n_iter used to return acceptance_rate = 0.0 and adapt forever."""
    with pytest.raises(ValueError, match="burn_in"):
        mcmc_sampler(gaussian_density, np.array([[-5.0, 5.0]]), n_iter=100, burn_in=500)


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


# ---------------------------------------------------------------------------
# The chain must stay inside the domain
# ---------------------------------------------------------------------------


def test_chain_never_leaves_the_domain():
    """Regression: the proposal was an unbounded random walk.

    Out-of-domain proposals were never rejected and `space` bounded only the
    initial draw, so with a non-periodic target the chain wandered dozens of
    periods away and the wrapped marginal was indistinguishable from uniform.
    """
    space = np.array([[-2.0, 3.0]])
    seen = []

    def density(x):
        seen.append(float(x[0]))
        return 1.0 + 0.5 * np.sin(x[0])

    for seed in range(20):
        sample = mcmc_sampler(density, space, n_iter=300, burn_in=50, seed=seed)
        assert space[0, 0] <= float(sample[0]) <= space[0, 1]

    assert seen, "density was never evaluated"
    assert min(seen) >= space[0, 0]
    assert max(seen) <= space[0, 1]


def test_transform_folds_instead_of_rejecting():
    """A periodic domain may fold, which mixes better across the seam."""
    period = 2 * np.pi
    space = np.array([[-np.pi, np.pi]])

    def periodize(x):
        return (np.asarray(x, dtype=float) + np.pi) % period - np.pi

    sample = mcmc_sampler(
        lambda x: 1.0, space, n_iter=400, burn_in=100, seed=0, transform=periodize
    )
    assert -np.pi <= float(sample[0]) <= np.pi


def test_proposal_scale_follows_the_domain_width():
    """A fixed scalar std cannot serve a domain three orders of magnitude wide."""
    wide = np.array([[-500.0, 500.0]])
    _, rate = mcmc_sampler(
        lambda x: float(np.exp(-0.5 * (x[0] / 100.0) ** 2)),
        wide,
        n_iter=3000,
        burn_in=1000,
        seed=0,
        return_diagnostics=True,
    )
    assert 0.05 < rate < 0.95, f"acceptance rate {rate} suggests a badly scaled proposal"


def test_anisotropic_domain_gets_a_per_axis_scale():
    """A scalar std is wrong on both axes of a strongly anisotropic box."""
    space = np.array([[-1.5, 1.5], [-50.0, 50.0]])
    _, rate = mcmc_sampler(
        lambda x: float(np.exp(-0.5 * (x[0] ** 2 + (x[1] / 20.0) ** 2))),
        space,
        n_iter=3000,
        burn_in=1000,
        seed=1,
        return_diagnostics=True,
    )
    assert 0.05 < rate < 0.95


# ---------------------------------------------------------------------------
# Initialisation and the acceptance test
# ---------------------------------------------------------------------------


def _narrow(x):
    """Positive only on a thin slice, so a uniform search usually misses."""
    return 1.0 if -0.1 < x[0] < 0.1 else 0.0


def test_initialisation_failure_raises():
    """It used to fall through, then divide by a zero density.

    With a Python-float density 2 of 30 seeds raised ZeroDivisionError; with a
    NumPy float 0/0 gave nan, and min(1.0, nan) is 1.0 in Python, so the chain
    accepted every proposal and silently returned a point outside the support.
    """
    with pytest.raises(RuntimeError, match="x0="):
        mcmc_sampler(
            _narrow, np.array([[-50.0, 50.0]]), n_iter=200, burn_in=50, seed=0, max_init_tries=5
        )


@pytest.mark.parametrize("seed", range(30))
def test_concentrated_target_never_returns_a_point_outside_its_support(seed):
    sample = mcmc_sampler(
        _narrow, np.array([[-50.0, 50.0]]), n_iter=200, burn_in=50, seed=seed, x0=np.array([0.0])
    )
    assert abs(float(sample[0])) < 0.1


def test_explicit_start_outside_the_support_raises():
    with pytest.raises(RuntimeError, match="not positive and finite"):
        mcmc_sampler(_narrow, np.array([[-50.0, 50.0]]), seed=0, x0=np.array([40.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_non_finite_or_zero_density_is_never_accepted(bad):
    """A nan ratio used to be treated as certain acceptance."""
    calls = {"n": 0}

    def density(x):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else bad

    sample = mcmc_sampler(
        density, np.array([[-1.0, 1.0]]), n_iter=50, burn_in=10, seed=0, x0=np.array([0.0])
    )
    assert float(sample[0]) == 0.0, "the chain must not move to a bad-density point"


def test_space_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"shape \(ndim, 2\)"):
        mcmc_sampler(gaussian_density, np.array([-5.0, 5.0]), seed=0)
