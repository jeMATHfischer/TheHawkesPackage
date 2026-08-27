"""Tests for spatial domains, including the contract every SpatialDomain owes."""

import numpy as np
import pytest

from hawkes_package import Circle, FundamentalDomain, SpatialDomain, Torus2D
from hawkes_package.spatio_temporal import _integration


class Interval(SpatialDomain):
    """A minimal non-periodic domain, proving third parties can implement the ABC.

    Also exercises the generic fallback branch of ``make_periodic``.
    """

    def __init__(self, lo=0.0, hi=1.0):
        self.lo, self.hi = float(lo), float(hi)

    def distance(self, x, y):
        return abs(float(np.ravel(x)[0]) - float(np.ravel(y)[0]))

    def wrap(self, x):
        return np.clip(np.asarray(x, dtype=float), self.lo, self.hi)

    def sample_uniform(self, rng):
        return rng.uniform(self.lo, self.hi, size=(1,))

    @property
    def volume(self):
        return self.hi - self.lo

    @property
    def bounds(self):
        return np.array([[self.lo, self.hi]])


# ---------------------------------------------------------------------------
# The ABC itself
# ---------------------------------------------------------------------------


def test_spatial_domain_is_abstract():
    with pytest.raises(TypeError):
        SpatialDomain()


def test_incomplete_subclass_cannot_be_instantiated():
    class Partial(SpatialDomain):
        def distance(self, x, y):
            return 0.0

    with pytest.raises(TypeError):
        Partial()


def test_third_party_domain_satisfies_the_contract():
    d = Interval(0.0, 2.0)
    assert d.volume == pytest.approx(2.0)
    assert d.bounds.shape == (1, 2)
    assert d.distance(np.array([0.5]), np.array([1.5])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Contract obeyed by every concrete domain (parametrized via the `domain` fixture)
# ---------------------------------------------------------------------------


def off_domain(domain, rng):
    """A point outside `domain` that `wrap` is obliged to bring back inside.

    A far-away lift wherever the chart is the whole plane, and an image of an
    interior point under the deck group otherwise. The distinction is not
    pedantry: the Poincare disc chart covers ``|z| < 1``, so a draw from
    ``(-10, 10)^2`` is not a set of hyperbolic points at all and `wrap` rightly
    refuses it. Deck-group images are the case that matters anyway -- they are
    what a lift coming out of `orbit` or `make_periodic` actually looks like.
    """
    ndim = domain.bounds.shape[0]
    images = domain.orbit(domain.sample_uniform(rng), 2)
    if images is None:
        return rng.uniform(-10, 10, size=ndim)
    return np.asarray(images[int(rng.integers(len(images)))], dtype=float)


def test_bounds_shape(domain):
    bounds = domain.bounds
    assert bounds.ndim == 2
    assert bounds.shape[1] == 2
    assert np.all(bounds[:, 0] < bounds[:, 1])


def test_volume_matches_quadrature(domain):
    """`volume` must be what the simulator's own integration rule measures.

    The generalisation of the old ``volume == prod(bounds widths)``: since 0.3.0
    a domain may be a proper subset of its bounding box, and the rule is masked
    by `contains` and weighted by `volume_element` accordingly. For a
    box-filling domain nothing is masked and this is the same assertion as
    before. Getting the number wrong scales the simulated event rate by the same
    factor, with no other symptom.
    """
    rule = _integration.restrict(
        _integration.build(domain.bounds, domain.nodes_per_axis),
        domain.contains,
        domain.volume_element,
    )
    assert float(rule.weights.sum()) == pytest.approx(domain.volume, rel=1e-2)


def test_sample_uniform_lands_inside_the_domain(domain, rng):
    for _ in range(50):
        assert domain.contains(domain.sample_uniform(rng))


def test_wrap_lands_inside_the_domain(domain, rng):
    for _ in range(20):
        assert domain.contains(domain.wrap(off_domain(domain, rng)))


def test_volume_element_is_strictly_positive(domain, rng):
    """The thinning bound dominates only because every quadrature weight does."""
    for _ in range(20):
        assert domain.volume_element(domain.sample_uniform(rng)) > 0.0


def test_wrap_is_idempotent(domain, rng):
    for _ in range(20):
        once = domain.wrap(off_domain(domain, rng))
        twice = domain.wrap(once)
        np.testing.assert_allclose(twice, once)


def test_wrap_lands_inside_bounds(domain, rng):
    bounds = domain.bounds
    for _ in range(20):
        wrapped = np.atleast_1d(domain.wrap(off_domain(domain, rng)))
        assert np.all(wrapped >= bounds[:, 0] - 1e-9)
        assert np.all(wrapped <= bounds[:, 1] + 1e-9)


def test_distance_is_zero_on_identity(domain, rng):
    for _ in range(10):
        x = domain.sample_uniform(rng)
        assert domain.distance(x, x) == pytest.approx(0.0)


def test_distance_is_symmetric(domain, rng):
    for _ in range(20):
        x, y = domain.sample_uniform(rng), domain.sample_uniform(rng)
        assert domain.distance(x, y) == pytest.approx(domain.distance(y, x))


def test_distance_is_non_negative_and_bounded(domain, rng):
    """No two points can be further apart than the domain says they can.

    Against `max_distance`, not against the bounding box: the box half-diagonal
    is the right answer only for a flat domain that fills its box, and it is not
    even the right order of magnitude for a hyperbolic one, whose Poincare disc
    chart is barely a unit across while the surface is several units wide.
    """
    max_dist = domain.max_distance
    for _ in range(30):
        x, y = domain.sample_uniform(rng), domain.sample_uniform(rng)
        d = domain.distance(x, y)
        assert 0.0 <= d <= max_dist + 1e-9


def test_distance_satisfies_triangle_inequality(domain, rng):
    for _ in range(50):
        x, y, z = (domain.sample_uniform(rng) for _ in range(3))
        assert domain.distance(x, z) <= domain.distance(x, y) + domain.distance(y, z) + 1e-9


def test_sample_uniform_stays_in_bounds(domain, rng):
    bounds = domain.bounds
    for _ in range(50):
        s = np.atleast_1d(domain.sample_uniform(rng))
        assert s.shape == (bounds.shape[0],)
        assert np.all(s >= bounds[:, 0])
        assert np.all(s <= bounds[:, 1])


def test_sample_uniform_is_reproducible(domain):
    a = domain.sample_uniform(np.random.default_rng(0))
    b = domain.sample_uniform(np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Domain-specific behaviour
# ---------------------------------------------------------------------------


class TestCircle:
    def test_distance_takes_the_short_arc(self):
        c = Circle()
        d = c.distance(np.array([0.0]), np.array([2 * np.pi - 0.1]))
        assert d == pytest.approx(0.1)

    def test_wrap_stays_in_range(self):
        c = Circle()
        wrapped = c.wrap(np.array([3 * np.pi]))
        assert -np.pi <= float(wrapped[0]) <= np.pi

    def test_volume_is_the_circumference(self):
        assert Circle(radius=1.0).volume == pytest.approx(2 * np.pi)
        assert Circle(radius=2.0).volume == pytest.approx(4 * np.pi)

    def test_antipodal_points_are_maximally_distant(self):
        c = Circle()
        assert c.distance(np.array([0.0]), np.array([np.pi])) == pytest.approx(np.pi)


class TestTorus2D:
    def test_distance_wraps_both_axes(self):
        t = Torus2D(L1=2.0, L2=2.0)
        d = t.distance(np.array([-1.0, -1.0]), np.array([0.9, 0.9]))
        assert d == pytest.approx(np.sqrt(0.1**2 + 0.1**2), rel=1e-6)

    def test_wrap_both_axes(self):
        t = Torus2D(L1=4.0, L2=6.0)
        wrapped = t.wrap(np.array([4.3, -3.5]))
        assert -2.0 <= wrapped[0] <= 2.0
        assert -3.0 <= wrapped[1] <= 3.0

    def test_volume_is_the_area(self):
        assert Torus2D(L1=3.0, L2=5.0).volume == pytest.approx(15.0)

    def test_bounds_are_two_dimensional(self):
        assert Torus2D().bounds.shape == (2, 2)


class TestFundamentalDomainAgreesWithTorus2D:
    """The rectangle case must reproduce the hand-written `Torus2D`.

    This is what makes the general machinery trustworthy: `Torus2D` is the one
    quotient in the package whose geometry was written out by hand and is
    covered by its own tests, so agreement with it exercises the polygon
    construction, the deck group, the reduction and the orbit search against a
    known answer rather than against itself.
    """

    L1, L2 = 3.0, 5.0

    @pytest.fixture
    def pair(self):
        return FundamentalDomain.rectangle(self.L1, self.L2), Torus2D(L1=self.L1, L2=self.L2)

    def test_volume_and_bounds_agree(self, pair):
        polygon, torus = pair
        assert polygon.volume == pytest.approx(torus.volume)
        np.testing.assert_allclose(polygon.bounds, torus.bounds)

    def test_distance_agrees_everywhere(self, pair, rng):
        """Including on lifts far outside the domain, which must reduce first."""
        polygon, torus = pair
        for _ in range(500):
            x, y = rng.uniform(-9, 9, size=2), rng.uniform(-9, 9, size=2)
            assert polygon.distance(x, y) == pytest.approx(torus.distance(x, y), abs=1e-12)

    def test_wrap_agrees_modulo_the_lattice(self, pair, rng):
        """Not pointwise: the two use opposite half-open conventions on a side.

        `Torus2D` folds into ``[-L/2, L/2)`` per axis; the polygon closes the
        edge whose outward normal is lexicographically negative. Both are valid
        choices of representative, so what must agree is the lattice class.
        """
        polygon, torus = pair
        periods = np.array([self.L1, self.L2])
        for _ in range(200):
            x = rng.uniform(-9, 9, size=2)
            offset = (polygon.wrap(x) - torus.wrap(x)) / periods
            np.testing.assert_allclose(offset, np.round(offset), atol=1e-12)


class TestFundamentalDomain:
    def test_hexagon_volume_is_the_regular_hexagon_area(self):
        for side in (0.5, 1.0, 2.5):
            assert FundamentalDomain.hexagon(side).volume == pytest.approx(
                1.5 * np.sqrt(3.0) * side**2
            )

    def test_hexagon_is_a_proper_subset_of_its_bounding_box(self):
        """The property the old box-volume check made impossible."""
        hexagon = FundamentalDomain.hexagon(1.0)
        widths = hexagon.bounds[:, 1] - hexagon.bounds[:, 0]
        assert hexagon.volume < float(np.prod(widths))
        assert not hexagon.contains([hexagon.bounds[0, 1], hexagon.bounds[1, 1]])  # a box corner

    def test_distance_is_zero_to_every_image(self, rng):
        """A point and its translates are the same point of the quotient."""
        hexagon = FundamentalDomain.hexagon(1.0)
        for _ in range(20):
            x = hexagon.sample_uniform(rng)
            for image in hexagon.orbit(x, n_images=2):
                assert hexagon.distance(x, image) == pytest.approx(0.0, abs=1e-9)

    def test_quotient_diameter_is_the_covering_radius(self, rng):
        """For circumradius R the triangular lattice covers at radius exactly R."""
        hexagon = FundamentalDomain.hexagon(1.0)
        far = max(
            hexagon.distance(hexagon.sample_uniform(rng), hexagon.sample_uniform(rng))
            for _ in range(2000)
        )
        assert far <= 1.0 + 1e-9
        assert far > 0.9  # and the bound is attained, not merely respected

    @pytest.mark.parametrize("side", [1.0, 2.5])
    def test_a_boundary_point_has_exactly_one_representative(self, side):
        """Half-open on paired sides, or the quotient double-counts an edge.

        Edge midpoints and corners both, and corners are the harder case: a
        corner lies on two edges at once, so the convention has to leave exactly
        one image of it with *both* of its edges closed.
        """
        hexagon = FundamentalDomain.hexagon(side)
        corners = hexagon.vertices
        midpoints = 0.5 * (corners + np.roll(corners, -1, axis=0))
        for point in np.vstack([corners, midpoints]):
            inside = [img for img in hexagon.orbit(point, 2) if hexagon.contains(img)]
            assert len(inside) == 1, f"boundary point {point} has {len(inside)} representatives"

    def test_wrap_of_a_boundary_point_is_the_representative(self):
        """And `wrap` must agree with `contains` about which image that is."""
        hexagon = FundamentalDomain.hexagon(1.0)
        corners = hexagon.vertices
        for point in np.vstack([corners, 0.5 * (corners + np.roll(corners, -1, axis=0))]):
            wrapped = hexagon.wrap(point)
            assert hexagon.contains(wrapped)
            assert hexagon.distance(wrapped, point) == pytest.approx(0.0, abs=1e-9)

    def test_orientation_reversing_pairing_is_accepted(self):
        """The same glide 0.3.0 rejected is the Klein bottle's own side pairing.

        0.3.0 raised here on the determinant alone, and that was a *policy*, not
        a defect: a non-orientable quotient is a perfectly good closed surface,
        and this exact matrix — a fixed-point-free glide reflection — is what
        presents the second of the two flat ones. What replaced the policy is
        the check that actually matters, freeness, which this passes and a pure
        reflection does not.
        """
        square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        glide = [[-1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]  # det -1
        translation = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        domain = FundamentalDomain(square, [glide, translation])
        assert domain.topology.name == "Klein bottle"
        assert domain.topology.orientable is False

    def test_pure_reflection_is_rejected_where_the_glide_is_not(self):
        """The two differ by a translation along the axis, and by having a fixed line."""
        square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # no glide
        translation = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValueError, match=r"act freely.*reflection"):
            FundamentalDomain(square, [reflection, translation])

    def test_rotation_pairing_is_rejected(self):
        """It is an isometry with determinant +1, and it quotients to an orbifold.

        The case the determinant test could never have caught: a quarter turn
        about the centre passes every check 0.3.0 made, has a fixed point, and
        glues the square to a sphere with four cone points.
        """
        square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        quarter_turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValueError, match=r"act freely.*rotation"):
            FundamentalDomain(square, [quarter_turn])

    def test_non_isometric_pairing_is_rejected(self):
        square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        shear = [[1.0, 0.5, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # det 1, not orthogonal
        with pytest.raises(ValueError, match="isometry"):
            FundamentalDomain(square, [shear])

    def test_non_convex_polygon_is_rejected(self):
        chevron = [[0.0, 0.0], [2.0, 1.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        with pytest.raises(ValueError, match="convex"):
            FundamentalDomain(chevron, [[[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    def test_pairings_that_do_not_tile_are_rejected_at_construction(self):
        """A translation too short to be a period does not pair any side.

        Through 0.3.0 this constructed happily and failed later, from inside
        `wrap`, mid-simulation and a long way from the mistake — and only if a
        point ever needed reducing along the axis nothing moved. The side
        correspondence is now required at construction, so it fails at the call
        that is wrong.
        """
        square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        too_short = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValueError, match="not carried onto another side"):
            FundamentalDomain(square, [too_short])

    def test_clockwise_vertices_are_accepted(self):
        """Winding order is the caller's business, not the domain's."""
        corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        pairings = [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        ]
        assert FundamentalDomain(corners, pairings).volume == pytest.approx(
            FundamentalDomain(corners[::-1], pairings).volume
        )

    def test_interior_point_is_inside(self):
        for domain in (FundamentalDomain.hexagon(1.0), FundamentalDomain.rectangle(3.0, 5.0)):
            assert domain.contains(domain.interior_point)

    def test_is_not_declared_periodic(self):
        """So MCMC proposals off the polygon are rejected, not folded.

        Folding is a reversible move only when the deck group acts by
        translations, which leave a Gaussian proposal invariant. That holds for
        both domains built here but not for a general pairing, and the sampler
        cannot tell the difference from the outside.
        """
        assert FundamentalDomain.hexagon(1.0).periodic is False


def test_circle_distance_rejects_a_two_vector():
    """`.flat[0]` silently measured only the first component."""
    with pytest.raises(ValueError, match="1 coordinate"):
        Circle().distance(np.array([0.0, 5.0]), np.array([1.0, -5.0]))


def test_torus_distance_accepts_a_column_vector():
    t = Torus2D()
    column = t.distance(np.array([[0.5], [1.0]]), np.array([[2.0], [-1.0]]))
    flat = t.distance(np.array([0.5, 1.0]), np.array([2.0, -1.0]))
    assert column == pytest.approx(flat)


def test_torus_distance_rejects_a_scalar():
    with pytest.raises(ValueError, match="2 coordinate"):
        Torus2D().distance(0.5, 1.0)
