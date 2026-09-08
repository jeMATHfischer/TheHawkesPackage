"""The first domain here with a real boundary.

Every other domain in the package is a closed surface -- periodic, like `Circle`
and `Torus2D`, or a quotient, like `Sphere` and `FundamentalDomain`. `Rectangle`
is neither, and the contract battery in `test_domains.py` (which it joins
through the `domain` fixture) checks that it still satisfies everything a domain
must. This file checks the things that are true *because* it has an edge.
"""

import numpy as np
import pytest
from scipy import stats

import hawkes_package as hp


def test_it_declares_a_boundary_and_no_periodicity():
    """Both flags, and they are not each other's negation.

    `Sphere` and `FundamentalDomain` are also `periodic = False` and have no
    boundary at all -- the first is not a quotient, the second is glued by
    isometries that are not translations. A correction keyed off `periodic`
    would rescale their intensity for nothing, which is why `has_boundary` is
    its own flag.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    assert rectangle.has_boundary is True
    assert rectangle.periodic is False

    for closed in (hp.Sphere(), hp.FundamentalDomain.rectangle(4.0, 3.0), hp.Torus2D()):
        assert closed.has_boundary is False


def test_the_same_box_glued_up_is_a_different_domain():
    """`Rectangle(4, 3)` and `FundamentalDomain.rectangle(4, 3)` share a box only.

    The torus wraps, so its diameter is the half-diagonal and opposite edges are
    adjacent. The rectangle does not, so its diameter is the full diagonal and
    the corners are the farthest pair. Confusing the two is the easiest mistake
    available here, and it is silent: both have the same `volume` and `bounds`.
    """
    bounded = hp.Rectangle(4.0, 3.0)
    glued = hp.FundamentalDomain.rectangle(4.0, 3.0)

    assert bounded.volume == pytest.approx(glued.volume)
    np.testing.assert_allclose(bounded.bounds, glued.bounds)

    corner_a = np.array([-2.0, -1.5])
    corner_b = np.array([2.0, 1.5])
    assert bounded.distance(corner_a, corner_b) == pytest.approx(5.0)
    # On the torus the two corners are the *same point*.
    assert glued.distance(corner_a, corner_b) == pytest.approx(0.0, abs=1e-9)


def test_the_diameter_is_the_full_diagonal():
    """Not the half-diagonal the base class defaults to.

    That default is right for a domain with every axis periodic, where the
    farthest two points are half a period apart. Nothing wraps here.
    """
    assert hp.Rectangle(4.0, 3.0).max_distance == pytest.approx(5.0)
    assert hp.Rectangle(2.0).max_distance == pytest.approx(2.0)


def test_distance_is_plain_euclidean():
    rectangle = hp.Rectangle(4.0, 3.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rectangle.sample_uniform(rng), rectangle.sample_uniform(rng)
        assert rectangle.distance(x, y) == pytest.approx(float(np.linalg.norm(x - y)))


def test_wrap_clips_rather_than_folding():
    """A clipping map is not reversible, which is why `periodic` is False.

    Folding an MCMC proposal through it would pile every out-of-bounds draw onto
    the boundary instead of rejecting it, and the chain would target a density
    with atoms on the edge.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    np.testing.assert_allclose(rectangle.wrap(np.array([9.0, -9.0])), [2.0, -1.5])
    np.testing.assert_allclose(rectangle.wrap(np.array([1.0, 1.0])), [1.0, 1.0])
    # Idempotent, as `wrap` must be for any domain.
    once = rectangle.wrap(np.array([9.0, -9.0]))
    np.testing.assert_array_equal(rectangle.wrap(once), once)


def test_contains_answers_honestly_outside_the_box():
    """The inherited default returns True unconditionally.

    Sound for the quadrature, which only ever asks about nodes drawn from
    `bounds`, but not what the method says -- and a bounded domain is the first
    one where a caller might reasonably ask about a point outside.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    assert rectangle.contains(np.array([0.0, 0.0]))
    assert rectangle.contains(np.array([2.0, 1.5])), "the closed boundary is inside"
    assert not rectangle.contains(np.array([2.01, 0.0]))
    assert not rectangle.contains(np.array([0.0, -1.51]))


def test_it_has_no_deck_group():
    """No translations to sum over, so `make_periodic` falls back to `distance`."""
    assert hp.Rectangle(4.0, 3.0).orbit(np.array([0.0, 0.0])) is None


@pytest.mark.statistical
def test_sample_uniform_is_uniform():
    """Per axis, against the analytic uniform, by KS."""
    rectangle = hp.Rectangle(4.0, 3.0, origin=[1.0, -2.0])
    rng = np.random.default_rng(3)
    points = np.array([rectangle.sample_uniform(rng) for _ in range(4000)])

    assert np.all(points >= rectangle.lower)
    assert np.all(points <= rectangle.upper)
    for axis, (low, high) in enumerate(rectangle.bounds):
        result = stats.kstest(points[:, axis], stats.uniform(low, high - low).cdf)
        assert result.pvalue > 1e-3, f"axis {axis} is not uniform (p={result.pvalue:.4g})"


def test_an_origin_shifts_the_box_without_changing_its_size():
    shifted = hp.Rectangle(4.0, 3.0, origin=[10.0, 10.0])
    np.testing.assert_allclose(shifted.bounds, [[10.0, 14.0], [10.0, 13.0]])
    assert shifted.volume == pytest.approx(12.0)
    assert shifted.contains(np.array([12.0, 11.0]))
    assert not shifted.contains(np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    ("sides", "kwargs", "match"),
    [
        ((), {}, "at least one side"),
        ((0.0,), {}, "finite and positive"),
        ((-1.0,), {}, "finite and positive"),
        ((np.inf,), {}, "finite and positive"),
        ((4.0, 3.0), {"origin": [1.0]}, "one entry per side"),
        ((4.0,), {"origin": [np.nan]}, "origin must be finite"),
    ],
)
def test_construction_is_validated(sides, kwargs, match):
    with pytest.raises(ValueError, match=match):
        hp.Rectangle(*sides, **kwargs)


def test_a_process_simulates_on_it(flat_base, exp_kernel, bump_spatial):
    """It drops into the existing machinery with no correction yet."""
    process = hp.SpatioTemporalHawkesProcess(
        base=flat_base,
        spatial=bump_spatial,
        temporal=exp_kernel,
        domain=hp.Rectangle(4.0, 3.0),
        monotone_temporal_kernel=True,
        rng=0,
    )
    process.simulate(8)

    assert process.events.shape == (3, 8)
    locations = process.events[1:]
    assert np.all(locations[0] >= -2.0)
    assert np.all(locations[0] <= 2.0)
    assert np.all(locations[1] >= -1.5)
    assert np.all(locations[1] <= 1.5)


# ---------------------------------------------------------------------------
# The general case
# ---------------------------------------------------------------------------

#: The 3-4-5 triangle: area 6 in a box of area 12, so the mask drops half the
#: nodes, and its hypotenuse is diagonal.
TRIANGLE = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]


def test_the_polygon_area_is_the_shoelace_area():
    assert hp.Polygon(TRIANGLE).volume == pytest.approx(6.0)
    square = hp.Polygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert square.volume == pytest.approx(4.0)


def test_the_winding_direction_does_not_matter():
    """Either ordering describes the same polygon, so both must be accepted."""
    forward = hp.Polygon(TRIANGLE)
    backward = hp.Polygon(TRIANGLE[::-1])
    assert forward.volume == pytest.approx(backward.volume)
    probe = np.array([0.5, 0.5])
    assert forward.contains(probe) == backward.contains(probe)


def test_contains_agrees_with_a_monte_carlo_area():
    """The predicate and the declared area have to be the same polygon.

    `volume` from the shoelace formula and `contains` from the half-planes are
    two independent statements about the same shape; if they disagreed, the
    quadrature would measure one and the process would be told the other.
    """
    triangle = hp.Polygon(TRIANGLE)
    rng = np.random.default_rng(0)
    lower, upper = triangle.bounds[:, 0], triangle.bounds[:, 1]
    draws = lower + rng.uniform(size=(20000, 2)) * (upper - lower)
    inside = np.mean([triangle.contains(point) for point in draws])
    box_area = float(np.prod(upper - lower))
    assert inside * box_area == pytest.approx(triangle.volume, rel=0.02)


def test_a_non_convex_polygon_is_refused():
    """`contains` is a conjunction of half-planes, which needs convexity.

    A re-entrant corner would make it admit points outside the polygon, and the
    quadrature would then integrate over a shape nobody described.
    """
    with pytest.raises(ValueError, match="not convex"):
        hp.Polygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]])


@pytest.mark.parametrize(
    ("vertices", "match"),
    [
        ([[0.0, 0.0], [1.0, 1.0]], "at least three"),
        ([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "zero area"),
        ([[0.0, 0.0], [1.0, 0.0], [np.nan, 1.0]], "finite"),
    ],
)
def test_polygon_construction_is_validated(vertices, match):
    with pytest.raises(ValueError, match=match):
        hp.Polygon(vertices)


def test_the_interior_point_is_inside_by_construction():
    """Convexity puts a mean of the vertices inside, with nothing to check.

    The base class returns the centre of the bounding box, which is a guarantee
    only for a domain that fills its box. In practice the box centre lands
    inside a convex polygon too -- it did on every one of 2024 random convex
    polygons, and no counterexample was found -- but the centroid needs no such
    search to believe, and this point is what probes whether the quadrature
    resolves the spatial kernel.
    """
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(300):
        corners = rng.uniform(0.0, 10.0, size=(int(rng.integers(3, 6)), 2))
        centre = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
        try:
            polygon = hp.Polygon(corners[np.argsort(angles)])
        except ValueError:
            continue  # collinear or non-convex after sorting
        checked += 1
        assert polygon.contains(polygon.interior_point)
    assert checked > 100, "the sweep should build a useful number of polygons"


def test_the_polygon_diameter_is_attained_at_two_vertices():
    assert hp.Polygon(TRIANGLE).max_distance == pytest.approx(5.0)


def test_wrap_returns_the_nearest_point_of_the_polygon():
    triangle = hp.Polygon(TRIANGLE)
    inside = np.array([0.5, 0.5])
    np.testing.assert_allclose(triangle.wrap(inside), inside)

    projected = triangle.wrap(np.array([5.0, 5.0]))
    assert triangle.contains(projected)
    np.testing.assert_allclose(triangle.wrap(projected), projected, atol=1e-12)


@pytest.mark.statistical
def test_polygon_sample_uniform_is_uniform():
    """By area: the share of draws in a sub-triangle must match its share of area."""
    triangle = hp.Polygon(TRIANGLE)
    rng = np.random.default_rng(5)
    points = np.array([triangle.sample_uniform(rng) for _ in range(4000)])
    assert all(triangle.contains(point) for point in points)

    # The half of the triangle with x below 2 has 3/4 of its area, since a
    # triangle's area scales with the square of its linear size.
    share = float(np.mean(points[:, 0] < 2.0))
    assert share == pytest.approx(0.75, abs=0.03)


def test_the_polygon_asks_for_more_quadrature_nodes():
    """A diagonal boundary cuts every panel, so the area error falls like 1/n.

    Measured on this triangle: 3.4% at the flat default of 32, 0.85% at 128 --
    and the area error is exactly the factor the simulated event rate is wrong
    by. 128 is the first value under the 1% the process warns at.
    """
    assert hp.Polygon(TRIANGLE).nodes_per_axis == 128
    assert hp.Rectangle(4.0, 3.0).nodes_per_axis == 32, "an axis-aligned box needs no extra"


def test_a_process_simulates_on_a_polygon():
    process = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 0.5,
        spatial=lambda d: max(0.0, 1.0 - d / np.pi),
        temporal=lambda dt: 0.9 * np.exp(-2.0 * np.asarray(dt, dtype=float)),
        domain=hp.Polygon(TRIANGLE),
        monotone_temporal_kernel=True,
        rng=0,
    )
    process.simulate(3)

    assert process.events.shape == (3, 3)
    assert process._renormalise is True, "a polygon has a boundary to correct for"
    for point in process.events[1:].T:
        assert process.domain.contains(point)
