"""Shared fixtures.

All randomness flows through explicit, seeded generators passed as ``rng=``.
Nothing here touches :func:`numpy.random.seed`, so ``pytest-randomly`` shuffling
test order and reseeding the globals between tests actively proves that no
hidden dependency on the global stream survives.
"""

import numpy as np
import pytest

from hawkes_package import Circle, Torus2D

SEED = 20260825


@pytest.fixture
def rng():
    """A fresh, deterministically seeded generator."""
    return np.random.default_rng(SEED)


@pytest.fixture
def exp_kernel():
    """Monotone-decreasing temporal kernel, kappa(t) = 0.9 exp(-2t)."""

    def kernel(dt):
        return 0.9 * np.exp(-2.0 * np.asarray(dt, dtype=float))

    return kernel


@pytest.fixture
def triangular_kernel():
    """Bell-shaped (triangular) kernel supported on (0, 1), peaking at 0.5."""

    def kernel(x):
        x = np.asarray(x, dtype=float)
        rise = (x > 0) & (x < 0.5)
        fall = (x >= 0.5) & (x < 1.0)
        return 2 * x * rise + (-2 * x + 2) * fall

    return kernel


@pytest.fixture
def flat_base():
    """Constant background intensity mu(x) = 0.5."""
    return lambda x: 0.5


@pytest.fixture
def bump_spatial():
    """Isotropic spatial kernel decaying linearly to zero at distance pi."""
    return lambda d: max(0.0, 1.0 - d / np.pi)


@pytest.fixture(
    params=[Circle(), Circle(radius=2.0), Torus2D(), Torus2D(L1=3.0, L2=5.0)],
    ids=["circle", "circle-r2", "torus", "torus-3x5"],
)
def domain(request):
    """Every concrete SpatialDomain, for contract tests."""
    return request.param


@pytest.fixture
def make_st_process(flat_base, exp_kernel, bump_spatial):
    """Factory building a SpatioTemporalHawkesProcess on a given domain."""
    from hawkes_package import SpatioTemporalHawkesProcess

    def _make(domain, *, seed=SEED, spatial=None, base=None, monotone=True):
        return SpatioTemporalHawkesProcess(
            base or flat_base,
            spatial or bump_spatial,
            exp_kernel,
            domain=domain,
            monotone_temporal_kernel=monotone,
            rng=seed,
        )

    return _make


@pytest.fixture
def legacy_kernels():
    """The (base, spatial, temporal) triple the original notebook used."""

    def base(x):
        return 0.5

    def spatial(s):
        b = np.pi
        s = np.asarray(s, dtype=float)
        inside = ((s + b / 2) >= 0) & ((s + b / 2) <= b)
        return (504 / (5 * np.pi**4) * s**4 - 146 / (5 * np.pi**2) * s**2 + 1) * inside

    def temporal(dt):
        a, b = 0.9, 2.0
        dt = np.asarray(dt, dtype=float)
        rise = (dt < b / 2) & (dt > 0)
        decay = (dt >= b / 2) & (dt < b)
        return 2 * a / b * dt * rise + ((-2 * a / b) * dt + 2 * a) * decay

    return base, spatial, temporal
