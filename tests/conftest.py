"""Shared fixtures.

All randomness flows through explicit, seeded generators passed as ``rng=``.
Nothing here touches :func:`numpy.random.seed`, so ``pytest-randomly`` shuffling
test order and reseeding the globals between tests actively proves that no
hidden dependency on the global stream survives.
"""

import numpy as np
import pytest

from hawkes_package import Circle, FundamentalDomain, Sphere, Torus2D

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
    params=[
        Circle(),
        Circle(radius=2.0),
        Torus2D(),
        Torus2D(width=3.0, height=5.0),
        FundamentalDomain.rectangle(3.0, 5.0),
        FundamentalDomain.hexagon(1.0),
        FundamentalDomain.hexagon(2.5),
        FundamentalDomain.klein_bottle(3.0, 5.0),
        Sphere(),
        Sphere(radius=2.0),
        FundamentalDomain.projective_plane(),
        FundamentalDomain.genus(2),
        FundamentalDomain.crosscaps(3),
    ],
    ids=[
        "circle",
        "circle-r2",
        "torus",
        "torus-3x5",
        "fd-rect-3x5",
        "fd-hex",
        "fd-hex-2.5",
        "fd-klein",
        "sphere",
        "sphere-r2",
        "fd-rp2",
        "fd-genus2",
        "fd-crosscaps3",
    ],
)
def domain(request):
    """Every concrete SpatialDomain, for contract tests.

    Deliberately one of each *kind*, because each breaks a different assumption
    the contract used to be able to make.

    * The hexagons do not fill their bounding box.
    * The Klein bottle is non-orientable, so its deck group contains a
      reflection and ``wrap`` is not a translation.
    * The spheres and the projective plane carry a non-flat ``volume_element``,
      so the chart measure and the surface measure differ; the projective plane
      also has a chart whose bounding box is not its boundary's.
    * The hyperbolic pair have a bounding box that reaches *outside* their model
      space, and a geodesic diameter several times the width of their chart.
    """
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
def delayed_bump_kernel():
    """Unimodal kernel that is exactly zero near lag 0 and peaks at lag 2.5.

    A local search started at 0 cannot see this peak: the kernel is flat at the
    start point, so the search returns 0, the peak value collapses to 0, and the
    bell-shaped bound silently degrades to the monotone one while the kernel is
    still rising. Before the fix that violated M >= lambda in 46% of steps.

    Total mass 0.5, so with the default phi(x) = x + 2 the process is stable.
    """

    def kernel(s):
        s = np.asarray(s, dtype=float)
        return np.maximum(0.0, 1.0 - np.abs(s - 2.5) * 2.0) * ((s > 2.0) & (s < 3.0))

    return kernel


@pytest.fixture
def signed_spatial():
    """Spatial kernel with an excitatory core and an inhibitory ring."""
    return lambda d: 2.0 * np.exp(-4.0 * d**2) - 0.8 * np.exp(-2.0 * (d - 2.0) ** 2)
