"""How a drifting parameter is allowed to move between blocks.

Each kernel has one defining property, and the properties are what distinguish
them in use -- not a preference. Liu-West preserves the cloud's mean and
variance *exactly*, which is what makes it well behaved on a parameter that is
merely uncertain and what makes it unable to follow a large jump once the cloud
has contracted: its jitter is proportional to a variance that has already gone
small. A plain random walk adds variance whether or not the data supports it,
and so can re-expand.

Both are checked here against their algebra rather than against an outcome, so
that a change to either is caught where it happens instead of three tests later
in a statistical battery.
"""

import numpy as np
import pytest

from hawkes_package.inference import LiuWest, RandomWalkDrift, Static
from hawkes_package.inference.evolution import _cholesky_with_floor, default_scale


@pytest.fixture
def cloud(rng):
    """A weighted, correlated cloud on the unconstrained scale."""
    mean = np.array([1.0, -2.0])
    factor = np.array([[0.8, 0.0], [0.5, 1.2]])
    z = mean + rng.normal(size=(4000, 2)) @ factor.T
    log_weights = np.log(rng.dirichlet(np.ones(4000)))
    return z, log_weights


def weighted_moments(z, log_weights):
    weights = np.exp(log_weights)
    mean = np.average(z, axis=0, weights=weights)
    centred = z - mean
    return mean, (centred * weights[:, None]).T @ centred


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


def test_static_is_the_identity(cloud, rng):
    z, log_weights = cloud
    moved = Static().propagate(z, log_weights, rng)
    np.testing.assert_array_equal(moved, z)
    assert Static().static is True
    assert repr(Static()) == "Static()"


# ---------------------------------------------------------------------------
# Liu-West
# ---------------------------------------------------------------------------


def test_liu_west_preserves_the_weighted_mean_and_variance(cloud, rng):
    r"""Its defining property, and the reason it exists.

    A plain jitter inflates the cloud's variance at every block, so a parameter
    that is genuinely constant looks more uncertain the longer it is watched.
    The shrinkage cancels that exactly: with ``a = (3d-1)/2d`` and
    ``h^2 = 1 - a^2``, ``E[z'] = zbar`` and ``Var[z'] = V``.
    """
    z, log_weights = cloud
    mean, covariance = weighted_moments(z, log_weights)
    moved = LiuWest(0.95).propagate(z, log_weights, rng)
    # The jitter is applied with equal weight per particle, so the moved cloud's
    # *unweighted* moments target the weighted ones of the cloud it came from.
    np.testing.assert_allclose(moved.mean(axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.cov(moved.T, bias=True), covariance, atol=0.1)


def test_the_liu_west_coefficients_are_the_published_ones():
    kernel = LiuWest(0.95)
    assert kernel.shrinkage == pytest.approx((3 * 0.95 - 1) / (2 * 0.95))
    assert kernel.shrinkage == pytest.approx(0.973684, rel=1e-5)
    assert kernel.jitter_variance == pytest.approx(1 - kernel.shrinkage**2)
    assert kernel.jitter_variance == pytest.approx(0.051939, rel=1e-4)
    assert kernel.static is False


def test_a_discount_at_one_does_not_move_the_cloud(cloud, rng):
    """delta = 1 is Static with a resample: no shrinkage and no jitter."""
    z, log_weights = cloud
    kernel = LiuWest(1.0)
    assert kernel.shrinkage == 1.0
    assert kernel.jitter_variance == 0.0
    np.testing.assert_allclose(kernel.propagate(z, log_weights, rng), z, atol=1e-12)


@pytest.mark.parametrize("delta", [0.5, 0.4, 0.0, -1.0, 1.5])
def test_a_discount_outside_its_range_is_refused(delta):
    """Below 0.5 the shrinkage turns negative and the kernel reflects the cloud."""
    with pytest.raises(ValueError, match=r"\(0.5, 1\]"):
        LiuWest(delta)


def test_liu_west_survives_a_collapsed_cloud(rng):
    """The cloud that most needs a jitter is the one whose covariance is singular."""
    z = np.tile(np.array([0.5, -0.5]), (32, 1))
    log_weights = np.full(32, -np.log(32))
    moved = LiuWest(0.95).propagate(z, log_weights, rng)
    assert np.all(np.isfinite(moved))
    # Nothing to move along: a collapsed cloud has no direction left.
    np.testing.assert_allclose(moved, z, atol=1e-9)


# ---------------------------------------------------------------------------
# RandomWalkDrift
# ---------------------------------------------------------------------------


def test_a_random_walk_adds_the_variance_it_says_it_does(cloud, rng):
    """Measured on the displacement, not on the difference of two variances.

    The cloud's own variance is an order of magnitude larger than what the
    jitter adds, so subtracting one from the other measures mostly noise -- 25%
    of it at four thousand particles. The displacement *is* the jitter, and its
    spread is the number the kernel promises.
    """
    z, log_weights = cloud
    kernel = RandomWalkDrift(0.3)
    displacement = kernel.propagate(z, log_weights, rng) - z
    np.testing.assert_allclose(displacement.std(axis=0), 0.3, rtol=0.05)
    np.testing.assert_allclose(displacement.mean(axis=0), 0.0, atol=0.02)
    assert kernel.static is False


def test_a_random_walk_can_re_expand_a_collapsed_cloud(rng):
    """The property Liu-West does not have, and the reason both are shipped.

    An absolute jitter moves a cloud that has contracted to a point; a jitter
    proportional to the cloud's own variance cannot. That is what decides which
    kernel can follow a parameter that jumps.
    """
    z = np.tile(np.array([0.5, -0.5]), (256, 1))
    log_weights = np.full(256, -np.log(256))
    walked = RandomWalkDrift(0.2).propagate(z, log_weights, rng)
    shrunk = LiuWest(0.95).propagate(z, log_weights, rng)
    assert walked.std(axis=0).min() > 0.15
    np.testing.assert_allclose(shrunk.std(axis=0), 0.0, atol=1e-9)


def test_a_per_coordinate_scale_is_honoured(rng):
    z = np.zeros((4000, 2))
    log_weights = np.full(4000, -np.log(4000))
    moved = RandomWalkDrift(np.array([0.1, 0.5])).propagate(z, log_weights, rng)
    np.testing.assert_allclose(moved.std(axis=0), [0.1, 0.5], rtol=0.1)


@pytest.mark.parametrize("scale", [0.0, -0.1, np.inf, np.nan])
def test_a_bad_random_walk_scale_is_refused(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        RandomWalkDrift(scale)


def test_the_kernels_repr_readably():
    assert repr(LiuWest(0.9)) == "LiuWest(0.9)"
    assert "0.05" in repr(RandomWalkDrift(0.05))


# ---------------------------------------------------------------------------
# Shared numerics
# ---------------------------------------------------------------------------


def test_the_cholesky_floor_factorises_a_singular_matrix():
    """The factorisation is needed exactly when the cloud makes it fail."""
    singular = np.outer([1.0, 2.0], [1.0, 2.0])
    factor = _cholesky_with_floor(singular)
    assert np.all(np.isfinite(factor))
    np.testing.assert_allclose(factor @ factor.T, singular, atol=1e-6)


def test_the_cholesky_floor_returns_a_negligible_factor_for_a_zero_matrix():
    """Every particle identical in every coordinate: no direction left to move in.

    The factor is not exactly zero -- the escalating ridge succeeds on its first
    attempt at around 1e-318, whose square root is 1e-159 -- but it is zero to
    every purpose, and returning it rather than raising keeps a collapsed cloud
    usable.
    """
    factor = _cholesky_with_floor(np.zeros((2, 2)))
    assert np.all(np.isfinite(factor))
    np.testing.assert_allclose(factor, 0.0, atol=1e-100)


def test_the_default_scale_is_the_published_constant():
    assert default_scale(1) == pytest.approx(2.38)
    assert default_scale(4) == pytest.approx(1.19)
