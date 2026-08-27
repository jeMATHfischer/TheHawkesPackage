"""The three constant-curvature model spaces, checked against closed forms.

`FundamentalDomain` is only as trustworthy as the geometry underneath it, and
the geometry is the layer with no independent implementation to compare
against — `Torus2D` covers the flat case and nothing covers the other two. So
everything here is checked against a formula that was derived rather than
implemented: great-circle distance against the arc, hyperbolic distance against
``2 artanh``, polygon area against the angle excess, and the classification of
isometries against motions written out by hand.
"""

import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from hawkes_package.spatio_temporal._model import (
    EuclideanPlane,
    HyperbolicPlane,
    SphericalPlane,
    isometry_between,
)

MODELS = {
    "euclidean": EuclideanPlane(),
    "sphere": SphericalPlane(),
    "sphere-r3": SphericalPlane(radius=3.0),
    "hyperbolic": HyperbolicPlane(),
}

#: A chart point comfortably inside each model space's chart.
INTERIOR = {
    "euclidean": np.array([0.7, -1.3]),
    "sphere": np.array([1.1, 0.4]),
    "sphere-r3": np.array([1.1, 0.4]),
    "hyperbolic": np.array([0.21, -0.35]),
}


@pytest.fixture(params=sorted(MODELS), ids=sorted(MODELS))
def model(request):
    return MODELS[request.param]


@pytest.fixture
def interior(request, model):
    return INTERIOR[request.node.callspec.params["model"]]


def scatter(model, rng, count):
    """Random chart points inside the model space."""
    if isinstance(model, HyperbolicPlane):
        radius = np.sqrt(rng.uniform(0.0, 0.36, size=count))
        angle = rng.uniform(-np.pi, np.pi, size=count)
        return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    if isinstance(model, SphericalPlane):
        return np.column_stack(
            [np.arccos(rng.uniform(-1, 1, size=count)), rng.uniform(-np.pi, np.pi, size=count)]
        )
    return rng.uniform(-3, 3, size=(count, 2))


# ---------------------------------------------------------------------------
# Chart and metric
# ---------------------------------------------------------------------------


def test_chart_and_lift_are_inverse(model, rng):
    for point in scatter(model, rng, 40):
        np.testing.assert_allclose(model.chart(model.lift(point)), point, atol=1e-12)


def test_distance_is_a_metric(model, rng):
    points = scatter(model, rng, 20)
    for x in points:
        assert model.distance(x, x) == pytest.approx(0.0, abs=1e-12)
    for x, y in zip(points, points[::-1], strict=True):
        assert model.distance(x, y) == pytest.approx(model.distance(y, x), abs=1e-12)
    shifted = (np.roll(points, 1, axis=0), np.roll(points, 2, axis=0))
    for x, y, z in zip(points, *shifted, strict=True):
        assert model.distance(x, z) <= model.distance(x, y) + model.distance(y, z) + 1e-9


def test_batched_distance_matches_the_scalar_one(model, rng):
    points = scatter(model, rng, 25)
    here = points[0]
    np.testing.assert_allclose(
        model.distances(here, points),
        [model.distance(here, other) for other in points],
        atol=1e-12,
    )


def test_ambient_distance_matches_the_chart_one(model, rng):
    points = scatter(model, rng, 25)
    here = points[0]
    np.testing.assert_allclose(
        model.ambient_distances(model.lift(here), model.lift_many(points)),
        [model.distance(here, other) for other in points],
        atol=1e-12,
    )


def test_distance_stays_accurate_near_zero(model, interior):
    """``arccos`` and ``arccosh`` lose half their digits at coincidence.

    Not academic: the quotient metric's inner loop compares a point against
    images of itself, so the *nearly zero* case is the one it evaluates most.
    Travelling a known arc length and measuring it back is the check with no
    linearisation in it — at an arc of ``1e-8`` the naive ``arccosh(-<p,q>)``
    has nothing left to work with, since ``cosh`` differs from one by ``5e-17``
    there, below the resolution of a double at one.
    """
    base = model.lift(interior)
    tangent = model.tangent(interior, interior + np.array([1e-3, 0.0]))
    for arc in (1e-2, 1e-4, 1e-6, 1e-8):
        far = model.chart(model._advance(base, tangent, arc))
        assert model.distance(interior, far) == pytest.approx(arc, rel=1e-5)


def test_a_geodesic_is_the_shortest_path(model, rng):
    """Interpolating along a geodesic must add up to the direct distance."""
    for _ in range(10):
        a, b = scatter(model, rng, 2)
        steps = np.linspace(0.0, 1.0, 9)
        path = model._interpolate(a, b, steps)
        total = sum(model.distance(p, q) for p, q in itertools.pairwise(path))
        assert total == pytest.approx(model.distance(a, b), rel=1e-9)


def test_midpoint_is_equidistant(model, rng):
    for _ in range(10):
        a, b = scatter(model, rng, 2)
        middle = model.midpoint(a, b)
        assert model.distance(a, middle) == pytest.approx(model.distance(middle, b), rel=1e-9)


# ---------------------------------------------------------------------------
# Closed forms, one geometry at a time
# ---------------------------------------------------------------------------


def test_spherical_distance_is_the_arc():
    sphere = SphericalPlane(radius=2.0)
    assert sphere.distance([0.0, 0.0], [np.pi, 0.0]) == pytest.approx(2.0 * np.pi)
    assert sphere.distance([np.pi / 2, 0.0], [np.pi / 2, np.pi / 2]) == pytest.approx(np.pi)
    assert sphere.distance([np.pi / 2, 0.0], [np.pi / 2, 0.3]) == pytest.approx(0.6)


def test_hyperbolic_distance_from_the_origin_is_two_artanh():
    plane = HyperbolicPlane()
    for radius in (0.1, 0.5, 0.9, 0.99):
        assert plane.distance([0.0, 0.0], [radius, 0.0]) == pytest.approx(2 * np.arctanh(radius))


def test_spherical_area_element_integrates_to_the_sphere():
    """``R^2 sin(theta)`` over the chart box must be the sphere's own area.

    The contract every :class:`~hawkes_package.SpatialDomain` owes -- the
    measure and the declared volume agreeing -- written out for the one chart
    where they are not trivially equal.
    """
    sphere = SphericalPlane(radius=1.5)
    over_colatitude, _ = quad(lambda t: sphere.volume_element([t, 0.0]), 0.0, np.pi)
    assert 2 * np.pi * over_colatitude == pytest.approx(4 * np.pi * sphere.radius**2, rel=1e-9)


def test_hyperbolic_volume_element_sup_is_at_the_outermost_vertex():
    plane = HyperbolicPlane()
    vertices = np.array([[0.5, 0.0], [0.0, 0.3], [-0.2, -0.2]])
    assert plane.hull_volume_element_sup(vertices) == pytest.approx(4.0 / (1 - 0.25) ** 2)


def test_hyperbolic_chart_rejects_points_outside_the_disc():
    plane = HyperbolicPlane()
    assert plane.in_chart(np.array([0.9, 0.3]))
    assert not plane.in_chart(np.array([0.9, 0.5]))
    with pytest.raises(ValueError, match=r"covers \|z\| < 1"):
        plane.lift(np.array([1.2, 0.0]))
    with pytest.raises(ValueError, match="not compactly contained"):
        plane.hull_volume_element_sup(np.array([[0.99999999, 0.9], [0.0, 0.0], [0.1, 0.1]]))


def test_flat_chart_covers_everything():
    assert EuclideanPlane().in_chart(np.array([1e6, -1e6]))
    assert SphericalPlane().in_chart(np.array([7.0, 100.0]))


# ---------------------------------------------------------------------------
# Polygon area: Gauss-Bonnet on a single cell
# ---------------------------------------------------------------------------


def test_euclidean_area_is_the_shoelace():
    plane = EuclideanPlane()
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    assert plane.polygon_area(square) == pytest.approx(6.0)
    assert plane.polygon_area(square[::-1]) == pytest.approx(6.0)


def test_spherical_area_is_the_angle_excess():
    """An octant is an eighth of the sphere and has three right angles."""
    sphere = SphericalPlane(radius=2.0)
    octant = np.array([[0.0, 0.0], [np.pi / 2, 0.0], [np.pi / 2, np.pi / 2]])
    assert sphere.polygon_area(octant) == pytest.approx(4 * np.pi * 4.0 / 8)


def test_hyperbolic_area_is_the_angle_defect():
    """An ideal-ish triangle: the smaller the angles, the closer to ``pi``."""
    plane = HyperbolicPlane()
    triangle = np.array([[0.99, 0.0], [-0.495, 0.857], [-0.495, -0.857]])
    assert plane.polygon_area(triangle) == pytest.approx(np.pi, rel=2e-2)
    assert plane.polygon_area(triangle) < np.pi


def test_regular_hyperbolic_polygon_area_matches_gauss_bonnet():
    """The genus-2 octagon: interior angle ``2 pi / 8``, so area ``4 pi``."""
    plane = HyperbolicPlane()
    angles = 2 * np.pi * np.arange(8) / 8
    radius = float(np.tanh(np.arccosh(1 / np.tan(np.pi / 8) ** 2) / 2))
    octagon = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    for index in range(8):
        interior = plane.angle(octagon[index - 1], octagon[index], octagon[(index + 1) % 8])
        assert interior == pytest.approx(2 * np.pi / 8, rel=1e-9)
    assert plane.polygon_area(octagon) == pytest.approx(4 * np.pi, rel=1e-9)


# ---------------------------------------------------------------------------
# Isometries
# ---------------------------------------------------------------------------


def test_isometry_between_matches_the_correspondence_it_was_given(model, rng):
    """Point, bearing and a sign determine an isometry; that is the whole builder."""
    for _ in range(8):
        source, towards, target, aiming = scatter(model, rng, 4)
        for orientation in (1.0, -1.0):
            built = isometry_between(model, (source, towards), (target, aiming), orientation)
            assert model.is_isometry(built)
            np.testing.assert_allclose(
                model.apply(built, source), model.chart(model.lift(target)), atol=1e-9
            )
            span = model.distance(source, towards)
            landed = model.apply(built, towards)
            assert model.distance(target, landed) == pytest.approx(span, rel=1e-9)


def test_isometries_preserve_distance(model, rng):
    points = scatter(model, rng, 12)
    built = isometry_between(model, (points[0], points[1]), (points[2], points[3]), 1.0)
    for x, y in zip(points, points[::-1], strict=True):
        moved = model.distance(model.apply(built, x), model.apply(built, y))
        assert moved == pytest.approx(model.distance(x, y), abs=1e-9)


def test_a_shear_is_not_an_isometry():
    assert not EuclideanPlane().is_isometry(
        np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert not SphericalPlane().is_isometry(np.diag([2.0, 1.0, 1.0]))
    assert not HyperbolicPlane().is_isometry(np.diag([2.0, 1.0, 1.0]))


def test_hyperbolic_isometry_must_preserve_the_upper_sheet():
    """``-I`` preserves the Minkowski form and swaps the two sheets."""
    assert not HyperbolicPlane().is_isometry(-np.eye(3))


@pytest.mark.parametrize(
    ("matrix", "kind", "free"),
    [
        (np.eye(3), "identity", False),
        (np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "translation", True),
        (np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), "rotation", False),
        (np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "reflection", False),
        (
            np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
            "glide reflection",
            True,
        ),
    ],
)
def test_euclidean_motions_are_classified(matrix, kind, free):
    """The distinction the whole non-orientable programme turns on.

    A glide reflection and a pure reflection have the same linear part and the
    same determinant. One is a legal side pairing and the other quotients to a
    mirror line, and the only thing separating them is whether the translation
    has a component along the reflection axis.
    """
    motion = EuclideanPlane().classify(matrix)
    assert (motion.kind, motion.free) == (kind, free)


def test_spherical_motions_are_classified():
    sphere = SphericalPlane()
    turn = 0.7
    rotation = np.array(
        [[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]]
    )
    assert sphere.classify(np.eye(3)).kind == "identity"
    assert sphere.classify(rotation) == ("rotation", 1.0, False)
    assert sphere.classify(np.diag([1.0, 1.0, -1.0])) == ("reflection", -1.0, False)
    assert sphere.classify(-np.eye(3)) == ("antipodal map", -1.0, True)
    assert sphere.classify(rotation @ np.diag([1.0, 1.0, -1.0])).free


def test_hyperbolic_motions_are_classified():
    plane = HyperbolicPlane()
    shift = 0.9
    translation = np.array(
        [
            [np.cosh(shift), 0.0, np.sinh(shift)],
            [0.0, 1.0, 0.0],
            [np.sinh(shift), 0.0, np.cosh(shift)],
        ]
    )
    reflection = np.diag([1.0, -1.0, 1.0])
    turn = 0.6
    elliptic = np.array(
        [[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]]
    )
    assert plane.classify(np.eye(3)).kind == "identity"
    assert plane.classify(translation) == ("translation", 1.0, True)
    assert plane.classify(reflection) == ("reflection", -1.0, False)
    assert plane.classify(translation @ reflection) == ("glide reflection", -1.0, True)
    assert plane.classify(elliptic) == ("elliptic rotation", 1.0, False)


def test_a_reflections_fixed_subspace_is_found_even_from_a_lightlike_basis():
    """Guard the guard: the eigenspace test must look at the *space*.

    A reflection's ``+1`` eigenspace is two-dimensional and can be handed back
    spanned by two lightlike vectors, each of Minkowski norm zero, whose span
    nonetheless contains the timelike direction that makes the motion unfree.
    Testing individual eigenvectors would call this free and admit a mirror line
    as a side pairing.
    """
    plane = HyperbolicPlane()
    # Conjugate the axis reflection so its eigenvectors are not axis-aligned.
    tilt = 0.8
    boost = np.array(
        [[np.cosh(tilt), 0.0, np.sinh(tilt)], [0.0, 1.0, 0.0], [np.sinh(tilt), 0.0, np.cosh(tilt)]]
    )
    tilted = boost @ np.diag([1.0, -1.0, 1.0]) @ np.linalg.inv(boost)
    assert plane.classify(tilted).free is False


def test_spherical_bearing_between_antipodes_is_undefined():
    with pytest.raises(ValueError, match="coincident or antipodal"):
        SphericalPlane().tangent(np.array([0.0, 0.0]), np.array([np.pi, 0.0]))


def test_a_bearing_to_the_same_point_is_undefined(model, interior):
    with pytest.raises(ValueError, match="bearing"):
        model.tangent(interior, interior)


def test_spherical_centroid_of_a_great_circle_is_undefined():
    """Which is why the projective plane's hemisphere is given its pole explicitly."""
    equator = np.array([[np.pi / 2, k * np.pi / 2] for k in (0, 1, 2, -1)])
    with pytest.raises(ValueError, match="centre of the sphere"):
        SphericalPlane().centroid(equator)
