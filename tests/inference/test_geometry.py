"""The parameter-free distance cache.

Its whole justification is that it computes the same numbers the simulator
would, only once. So the tests are about *agreement*, on every domain the
project ships, and about the two ways it can silently stop agreeing: a history
that has moved underneath it, and a periodised kernel whose image set it derives
differently from `make_periodic`.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import ConstantBase, ExponentialKernel, GaussianSpatial
from hawkes_package.inference._geometry import (
    build_geometry,
    extend_geometry,
    quadrature_for,
)
from hawkes_package.inference.models import SpatialComponents
from hawkes_package.spatio_temporal.kernels import image_distance_fn, make_periodic


def components_for(domain, *, n_quad=16, n_images=None):
    ndim = int(np.asarray(domain.bounds).shape[0])
    return SpatialComponents(
        domain=domain,
        base=ConstantBase(),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(ndim),
        n_quad=n_quad,
        n_images=n_images,
    )


def sample_points(domain, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([domain.wrap(domain.sample_uniform(rng)) for _ in range(n)])


# ---------------------------------------------------------------------------
# The images match the simulator's
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain",
    [hp.Circle(), hp.Circle(radius=2.0), hp.Torus2D(), hp.FundamentalDomain.hexagon(1.0)],
    ids=["circle", "circle-r2", "torus", "hexagon"],
)
def test_the_periodised_kernel_is_the_sum_over_the_cached_distances(domain):
    """The single reason `image_distance_fn` was factored out of `make_periodic`.

    If the cache summed over a different image set, the likelihood would be
    fitting a different process from the one the simulator draws -- and the
    difference would show up as a plausible posterior, not as an error.
    """

    def kernel(d):
        return np.exp(-((np.asarray(d, dtype=float)) ** 2))

    periodised = make_periodic(kernel, domain, 3)
    distances = image_distance_fn(domain, 3)
    rng = np.random.default_rng(1)
    for _ in range(25):
        x = domain.wrap(domain.sample_uniform(rng))
        y = domain.wrap(domain.sample_uniform(rng))
        assert periodised(x, y) == pytest.approx(
            float(sum(kernel(d) for d in distances(x, y))), rel=1e-14
        )


def test_an_unperiodised_cache_holds_exactly_the_geodesic_distance():
    domain = hp.Circle()
    points = sample_points(domain, 4)
    cache = build_geometry(components_for(domain), np.arange(1.0, 5.0), points)
    assert cache.node_event.shape[2] == 1
    expected = domain.distance(cache.nodes[0], points[:, 0])
    assert cache.node_event[0, 0, 0] == pytest.approx(expected, rel=1e-14)


def test_a_periodised_cache_holds_one_entry_per_image():
    domain = hp.Circle()
    points = sample_points(domain, 3)
    cache = build_geometry(components_for(domain, n_images=2), np.arange(1.0, 4.0), points)
    assert cache.node_event.shape[2] == 5  # 2 * n_images + 1
    assert cache.event_event.shape == (3, 3, 5)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_the_cache_uses_the_process_s_own_quadrature_rule():
    """Masked by `contains` and reweighted by `volume_element`, as the process is.

    A rule over the bounding box instead would differ from the simulator's by the
    ratio of the box's area to the domain's -- 1.30 on a hexagon.
    """
    domain = hp.FundamentalDomain.hexagon(1.0)
    rule = quadrature_for(components_for(domain, n_quad=16))
    assert np.all(rule.weights > 0)
    assert float(rule.weights.sum()) == pytest.approx(domain.volume, rel=0.05)
    assert all(domain.contains(node) for node in rule.nodes)


def test_event_to_itself_is_zero_distance():
    domain = hp.Circle()
    points = sample_points(domain, 5)
    cache = build_geometry(components_for(domain), np.arange(1.0, 6.0), points)
    np.testing.assert_allclose(np.diagonal(cache.event_event[:, :, 0]), 0.0, atol=1e-12)


def test_the_cache_reports_its_own_size():
    domain = hp.Circle()
    points = sample_points(domain, 6)
    cache = build_geometry(components_for(domain), np.arange(1.0, 7.0), points)
    assert cache.n_events == 6
    assert cache.n_nodes == cache.weights.size
    assert cache.nbytes == cache.node_event.nbytes + cache.event_event.nbytes


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_images", [None, 2])
def test_extending_equals_rebuilding(n_images):
    """The online loop grows the history block by block.

    Rebuilding each time would make the geometry cost quadratic in the number of
    blocks; extending must produce the same tensors it would have rebuilt.
    """
    domain = hp.Circle()
    parts = components_for(domain, n_images=n_images)
    points = sample_points(domain, 9)
    times = np.arange(1.0, 10.0)

    grown = build_geometry(parts, times[:4], points[:, :4])
    grown = extend_geometry(grown, parts, times[:7], points[:, :7])
    grown = extend_geometry(grown, parts, times, points)
    fresh = build_geometry(parts, times, points)

    np.testing.assert_allclose(grown.node_event, fresh.node_event, rtol=0, atol=0)
    np.testing.assert_allclose(grown.event_event, fresh.event_event, rtol=0, atol=0)
    np.testing.assert_array_equal(grown.times, fresh.times)


def test_extending_by_nothing_returns_the_same_cache():
    domain = hp.Circle()
    parts = components_for(domain)
    points = sample_points(domain, 4)
    times = np.arange(1.0, 5.0)
    cache = build_geometry(parts, times, points)
    assert extend_geometry(cache, parts, times, points) is cache


def test_a_history_that_is_not_an_extension_is_refused():
    """Rebuilding silently would hide a caller changing data under a running fit."""
    domain = hp.Circle()
    parts = components_for(domain)
    points = sample_points(domain, 5)
    times = np.arange(1.0, 6.0)
    cache = build_geometry(parts, times[:3], points[:, :3])
    altered = times.copy()
    altered[0] = 0.5
    with pytest.raises(ValueError, match="does not extend"):
        extend_geometry(cache, parts, altered, points)


def test_matches_recognises_a_prefix():
    domain = hp.Circle()
    points = sample_points(domain, 4)
    times = np.arange(1.0, 5.0)
    cache = build_geometry(components_for(domain), times[:2], points[:, :2])
    assert cache.matches(times)
    assert cache.matches(times[:2])
    assert not cache.matches(times[1:])


def test_an_oversized_cache_is_refused_with_the_three_numbers():
    domain = hp.Torus2D()
    points = sample_points(domain, 20)
    with pytest.raises(MemoryError, match="nodes x events x images"):
        build_geometry(
            components_for(domain, n_quad=32, n_images=3),
            np.arange(1.0, 21.0),
            points,
            max_bytes=1024,
        )
