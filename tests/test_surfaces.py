"""Every closed surface, and the checks that stop a presentation being the wrong one.

A polygon with side pairings glues to *something*. Whether that something is the
surface the caller meant is decided by three conditions, none of which anything
downstream can detect: the pairings must act freely, they must carry each side
onto exactly one other, and the corner cycles must close up with an angle sum of
:math:`2\\pi`. Violate any of them and the Ogata loop, the quadrature and the
location sampler all run perfectly well on an orbifold, a surface with boundary,
or a cone.

So the tests here come in two halves. The first asserts the topology of each
shipped presentation against the known answer -- orientability, Euler
characteristic, genus and area, cross-checked by Gauss-Bonnet. The second feeds
in presentations that are wrong in each of the available ways and requires each
to be refused at construction.
"""

import numpy as np
import pytest

from hawkes_package import FundamentalDomain, Sphere, Torus2D
from hawkes_package.spatio_temporal._gluing import group_window, window_by_radius
from hawkes_package.spatio_temporal._model import EuclideanPlane, HyperbolicPlane
from hawkes_package.spatio_temporal.domains import _pairings_from_word

# (constructor, orientable, Euler characteristic, genus, name)
PRESENTATIONS = [
    (lambda: FundamentalDomain.rectangle(3.0, 5.0), True, 0, 1, "torus"),
    (lambda: FundamentalDomain.hexagon(1.0), True, 0, 1, "torus"),
    (lambda: FundamentalDomain.klein_bottle(3.0, 5.0), False, 0, 2, "Klein bottle"),
    (lambda: FundamentalDomain.projective_plane(), False, 1, 1, "projective plane"),
    (lambda: FundamentalDomain.genus(1), True, 0, 1, "torus"),
    (lambda: FundamentalDomain.genus(2), True, -2, 2, "genus-2 surface"),
    (lambda: FundamentalDomain.genus(3), True, -4, 3, "genus-3 surface"),
    (lambda: FundamentalDomain.crosscaps(1), False, 1, 1, "projective plane"),
    (lambda: FundamentalDomain.crosscaps(2), False, 0, 2, "Klein bottle"),
    (lambda: FundamentalDomain.crosscaps(3), False, -1, 3, "non-orientable surface of genus 3"),
    (lambda: FundamentalDomain.crosscaps(4), False, -2, 4, "non-orientable surface of genus 4"),
]

IDS = [
    "rectangle",
    "hexagon",
    "klein-bottle",
    "projective-plane",
    "genus-1",
    "genus-2",
    "genus-3",
    "crosscaps-1",
    "crosscaps-2",
    "crosscaps-3",
    "crosscaps-4",
]


#: Built once and shared. A hyperbolic presentation costs seconds to construct
#: -- it enumerates a deck-group window and calibrates a quadrature rule -- and
#: rebuilding one per test made the setup cost more than the tests. Nothing here
#: mutates a domain, so sharing is safe.
_BUILT: dict[int, object] = {}


@pytest.fixture(params=range(len(PRESENTATIONS)), ids=IDS)
def presentation(request):
    index = request.param
    build, orientable, characteristic, genus, name = PRESENTATIONS[index]
    if index not in _BUILT:
        _BUILT[index] = build()
    return _BUILT[index], orientable, characteristic, genus, name


# ---------------------------------------------------------------------------
# What surface did we actually build?
# ---------------------------------------------------------------------------


def test_topology_is_the_one_the_constructor_promises(presentation):
    domain, orientable, characteristic, genus, name = presentation
    assert domain.topology.orientable is orientable
    assert domain.topology.euler_characteristic == characteristic
    assert domain.topology.genus == genus
    assert domain.topology.name == name


def test_gauss_bonnet_ties_area_to_topology(presentation):
    """``integral K dA == 2 pi chi``, over the surface the polygon presents.

    Three numbers arrived at three independent ways -- the area from the angle
    excess, the curvature from the model space, the characteristic from
    counting corner cycles -- and they have to agree. The constructor already
    enforces this; asserting it here is what makes the enforcement itself
    visible, and what would catch a change that quietly disabled it.
    """
    domain, _, characteristic, _, _ = presentation
    assert domain.model.curvature * domain.volume == pytest.approx(
        2 * np.pi * characteristic, abs=1e-9
    )


def test_every_corner_cycle_closes_up_exactly_once(presentation):
    """Poincare's condition, restated on the object rather than inside it."""
    domain = presentation[0]
    for cycle in domain.cycles:
        assert cycle.angle_sum == pytest.approx(2 * np.pi, abs=1e-9)
        np.testing.assert_allclose(cycle.transformation, np.eye(3), atol=1e-7)


def test_corner_cycles_partition_the_corners(presentation):
    """Every corner belongs to exactly one vertex of the quotient."""
    domain = presentation[0]
    visited = [corner for cycle in domain.cycles for corner in cycle.corners]
    assert sorted(visited) == list(range(len(domain.vertices)))


def test_every_side_is_glued_to_exactly_one_other(presentation):
    """The pairing is a fixed-point-free involution on the sides."""
    domain = presentation[0]
    partner = domain._pairing.partner
    assert sorted(partner.tolist()) == list(range(len(partner)))
    np.testing.assert_array_equal(partner[partner], np.arange(len(partner)))
    assert not np.any(partner == np.arange(len(partner)))


def test_each_corner_cycle_contributes_exactly_one_corner(presentation):
    """One representative per orbit, or the quotient counts a point twice.

    Stated over the corner cycles and the glued side pairs rather than over a
    truncated orbit, because those *are* the orbits — the cycle is the set of
    corners that become one point of the surface, and paired sides carry
    midpoint to midpoint.

    A per-side "closed or open" flag cannot deliver this, and the projective
    plane is where that shows: its four sides pair up two and two, and every
    assignment of flags leaves one corner cycle with two representatives and the
    other with none, because a corner's membership is decided by *both* of its
    sides at once.
    """
    domain = presentation[0]
    for cycle in domain.cycles:
        admitted = [c for c in sorted(set(cycle.corners)) if domain.contains(domain.vertices[c])]
        assert len(admitted) == 1, f"corner cycle {cycle.corners} has {len(admitted)} of them"


def test_each_glued_pair_of_sides_contributes_exactly_one_midpoint(presentation):
    """The same property for the interior of a side, where only two images meet."""
    domain = presentation[0]
    corners = domain.vertices
    count = len(corners)
    midpoints = [domain.model.midpoint(corners[i], corners[(i + 1) % count]) for i in range(count)]
    for side in range(count):
        partner = int(domain._pairing.partner[side])
        if partner < side:
            continue
        admitted = [m for m in (midpoints[side], midpoints[partner]) if domain.contains(m)]
        assert len(admitted) == 1, f"sides {side} and {partner} have {len(admitted)} midpoints"


def test_wrap_of_a_boundary_point_is_that_representative(presentation):
    domain = presentation[0]
    corners = domain.vertices
    count = len(corners)
    probes = list(corners) + [
        domain.model.midpoint(corners[i], corners[(i + 1) % count]) for i in range(count)
    ]
    for point in probes:
        wrapped = domain.wrap(point)
        assert domain.contains(wrapped)
        assert domain.distance(wrapped, point) == pytest.approx(0.0, abs=1e-8)


def test_a_point_and_its_images_are_the_same_point_of_the_quotient(presentation, rng):
    """Tolerance scaled to the surface, not absolute.

    A hyperbolic domain's chart is conditioned by its own curvature: near the
    genus-3 polygon's corners a chart displacement is magnified fifteenfold on
    the way to a distance, so an image reduced back through a dozen matrix
    products lands a few parts in :math:`10^8` of the diameter away rather than
    a few parts in :math:`10^{15}`.
    """
    domain = presentation[0]
    tolerance = 1e-7 * domain.max_distance
    for _ in range(6):
        point = domain.sample_uniform(rng)
        for image in domain.orbit(point, 2):
            assert domain.distance(point, image) == pytest.approx(0.0, abs=tolerance)


def test_repr_names_the_surface(presentation):
    domain, _, _, _, name = presentation
    assert name in repr(domain)


# ---------------------------------------------------------------------------
# The flat pair, against the hand-written torus
# ---------------------------------------------------------------------------


class TestKleinBottle:
    """The Klein bottle is the torus with one sign changed, and it shows.

    Both are the same rectangle with the same first pairing; the second is a
    translation for one and a glide reflection for the other. Everything that
    follows -- non-orientability, the identification of ``(x, y)`` with
    ``(-x, y + L2)``, the shorter distances -- follows from that one sign.
    """

    L1, L2 = 3.0, 5.0

    @pytest.fixture
    def bottle(self):
        return FundamentalDomain.klein_bottle(self.L1, self.L2)

    def test_area_matches_the_torus_on_the_same_rectangle(self, bottle):
        assert bottle.volume == pytest.approx(self.L1 * self.L2)

    def test_the_glide_identification_holds(self, bottle, rng):
        """``(x, y) ~ (-x, y + L2)`` is what the second pairing says."""
        for _ in range(30):
            point = bottle.sample_uniform(rng)
            glided = np.array([-point[0], point[1] + self.L2])
            assert bottle.distance(point, glided) == pytest.approx(0.0, abs=1e-9)

    def test_it_refines_the_double_length_torus(self, bottle, rng):
        """Its deck group contains that torus's lattice, and strictly more besides.

        The comparison has to be against ``L2`` *doubled*. Squaring the glide
        gives the translation ``(0, 2 L2)`` — not ``(0, L2)`` — so the Klein
        bottle's group contains the lattice of ``Torus2D(L1, 2 L2)`` at index
        two, and does *not* contain the lattice of ``Torus2D(L1, L2)`` at all.
        Points can therefore be further apart on the Klein bottle than on the
        equal-sized torus, which is not a defect but the difference between the
        two surfaces.
        """
        double = Torus2D(L1=self.L1, L2=2 * self.L2)
        shorter = 0
        for _ in range(300):
            x, y = bottle.sample_uniform(rng), bottle.sample_uniform(rng)
            assert bottle.distance(x, y) <= double.distance(x, y) + 1e-9
            shorter += bottle.distance(x, y) < double.distance(x, y) - 1e-9
        assert shorter > 0, "the glide never shortened anything; it is not being applied"

    def test_orientation_reversal_is_the_whole_difference(self, bottle):
        motions = [bottle.model.classify(pairing) for pairing in bottle.pairings]
        assert sorted(motion.orientation for motion in motions) == [-1.0, 1.0]
        assert all(motion.free for motion in motions)


class TestProjectivePlane:
    """The hemisphere with antipodal boundary points identified."""

    @pytest.fixture
    def plane(self):
        return FundamentalDomain.projective_plane(radius=1.5)

    def test_area_is_half_the_sphere(self, plane):
        assert plane.volume == pytest.approx(2 * np.pi * 1.5**2)

    def test_antipodal_points_are_the_same_point(self, plane, rng):
        for _ in range(30):
            colatitude, longitude = plane.sample_uniform(rng)
            antipode = np.array([np.pi - colatitude, longitude - np.pi])
            assert plane.distance([colatitude, longitude], antipode) == pytest.approx(0.0, abs=1e-8)

    def test_the_diameter_is_a_quarter_of_the_great_circle(self, plane, rng):
        """Identifying antipodes halves the sphere's diameter."""
        far = max(
            plane.distance(plane.sample_uniform(rng), plane.sample_uniform(rng)) for _ in range(400)
        )
        assert far <= np.pi * 1.5 / 2 + 1e-9
        assert far > 0.9 * np.pi * 1.5 / 2

    def test_the_deck_group_is_exactly_two_elements(self, plane):
        window = group_window(plane.model, plane.pairings, plane.interior_point, 4)
        assert len(window) == 2

    def test_it_needs_an_explicit_interior_point(self):
        """Its four corners lie on one great circle and average to the origin."""
        from hawkes_package.spatio_temporal._model import SphericalPlane

        equator = np.array([[np.pi / 2, k * np.pi / 2] for k in (0, 1, 2, -1)])
        with pytest.raises(ValueError, match="centre of the sphere"):
            FundamentalDomain(equator, [-np.eye(3)], model=SphericalPlane())


class TestSphere:
    """The one closed surface with no deck group at all."""

    def test_area_and_diameter(self):
        for radius in (0.5, 1.0, 3.0):
            sphere = Sphere(radius)
            assert sphere.volume == pytest.approx(4 * np.pi * radius**2)
            assert sphere.max_distance == pytest.approx(np.pi * radius)

    def test_it_declares_no_deck_group(self):
        """Simply connected: `make_periodic` must fall back, not sum images."""
        assert Sphere().orbit(np.array([1.0, 0.0])) is None

    def test_uniform_sampling_is_uniform_in_area_not_in_colatitude(self, rng):
        """Drawing theta uniformly would pile the sample at the poles."""
        sphere = Sphere()
        heights = np.array([np.cos(sphere.sample_uniform(rng)[0]) for _ in range(4000)])
        # cos(theta) is uniform on [-1, 1] exactly when the sample is uniform on
        # the sphere; its mean is 0 and its variance 1/3.
        assert abs(float(heights.mean())) < 4 / np.sqrt(3 * len(heights))
        assert float(heights.var()) == pytest.approx(1 / 3, rel=0.1)

    def test_wrap_canonicalises_a_wild_chart_point(self):
        sphere = Sphere()
        wrapped = sphere.wrap([5.0 + 2 * np.pi, 40.0])
        assert 0.0 <= wrapped[0] <= np.pi
        assert -np.pi <= wrapped[1] <= np.pi
        assert sphere.distance(wrapped, [5.0 + 2 * np.pi, 40.0]) == pytest.approx(0.0, abs=1e-9)


class TestHyperbolicSurfaces:
    """Genus two and up: the first quotients whose deck group is infinite."""

    def test_the_regular_octagon_has_the_angle_the_cycle_needs(self):
        surface = FundamentalDomain.genus(2)
        for angle in surface._angles:
            assert angle == pytest.approx(2 * np.pi / 8, rel=1e-9)

    def test_area_is_four_pi_times_genus_minus_one(self):
        for handles in (2, 3):
            assert FundamentalDomain.genus(handles).volume == pytest.approx(
                4 * np.pi * (handles - 1), rel=1e-9
            )

    def test_crosscap_area_is_two_pi_times_k_minus_two(self):
        for crosscaps in (3, 4):
            assert FundamentalDomain.crosscaps(crosscaps).volume == pytest.approx(
                2 * np.pi * (crosscaps - 2), rel=1e-9
            )

    def test_the_polygon_is_compactly_inside_the_disc(self):
        surface = FundamentalDomain.genus(2)
        assert float(np.max(np.linalg.norm(surface.vertices, axis=1))) < 0.99

    def test_the_bounding_box_reaches_outside_the_model_space(self):
        """Which is why `contains` has to answer for a point that is not a point.

        The octagon's vertices sit at ``|z| = 0.84`` on the axes, so the corners
        of its bounding box are at ``1.19`` -- outside the unit disc, and so not
        hyperbolic points at all. A tensor quadrature grid over that box
        contains such nodes, and asking whether they are in the domain must
        answer no rather than raise.
        """
        surface = FundamentalDomain.genus(2)
        corner = surface.bounds[:, 1]
        assert float(np.linalg.norm(corner)) > 1.0
        assert surface.contains(corner) is False

    def test_the_sphere_has_no_fundamental_domain(self):
        with pytest.raises(ValueError, match="simply connected"):
            FundamentalDomain.genus(0)

    def test_needs_more_quadrature_nodes_than_a_flat_polygon(self):
        """And says so itself, rather than leaving the caller to find out.

        The default of 32 per axis misses the octagon's area by 5%, which would
        scale the simulated event rate by the same factor with no symptom but a
        warning.
        """
        assert FundamentalDomain.hexagon(1.0).nodes_per_axis == 32
        assert FundamentalDomain.genus(2).nodes_per_axis >= 128


# ---------------------------------------------------------------------------
# The truncated deck group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: FundamentalDomain.hexagon(1.0),
        lambda: FundamentalDomain.klein_bottle(3.0, 5.0),
        lambda: FundamentalDomain.projective_plane(),
        lambda: FundamentalDomain.genus(2),
        lambda: FundamentalDomain.crosscaps(3),
    ],
    ids=["hexagon", "klein-bottle", "projective-plane", "genus-2", "crosscaps-3"],
)
def test_distance_is_the_true_minimum_over_the_whole_group(build, rng):
    """The certified search must agree with an exhaustive one over a wider window.

    This is the test the radius truncation exists for. The certificate lets the
    scan stop as soon as no unexamined element *can* beat what it has; the
    reference here stops at nothing, over a window half again as wide as the
    one the certificate settled for. Not the full four circumradii the triangle
    inequality permits: in a hyperbolic geometry that ball holds more than fifty
    thousand elements, and enumerating it is out of reach — which is itself the
    reason the certificate stops early rather than searching to the bound.

    Failing this would be quiet. A window that misses the minimiser returns a
    distance that is too *large*, which reads downstream as a kernel that has
    already decayed, so the process loses excitation rather than raising.
    """
    domain = build()
    exhaustive, _ = window_by_radius(
        domain.model,
        domain.pairings,
        domain.interior_point,
        3.0 * domain._scale,
        domain._scale,
    )
    assert len(exhaustive) >= len(domain._radius_window)
    for _ in range(25):
        x, y = domain.sample_uniform(rng), domain.sample_uniform(rng)
        lifted = domain.model.lift(y)
        brute = float(
            np.min(
                domain.model.ambient_distances(
                    domain.model.lift(x), np.einsum("kij,j->ki", exhaustive, lifted)
                )
            )
        )
        assert domain.distance(x, y) == pytest.approx(brute, abs=1e-9)


@pytest.mark.parametrize(
    "build",
    [
        lambda: FundamentalDomain.hexagon(1.0),
        lambda: FundamentalDomain.klein_bottle(3.0, 5.0),
        lambda: FundamentalDomain.projective_plane(),
    ],
    ids=["hexagon", "klein-bottle", "projective-plane"],
)
def test_distance_agrees_with_a_word_length_search(build, rng):
    """A second reference, built the way the truncation used to be.

    Only for the flat and spherical cases: a hyperbolic word of length five
    reaches a displacement of fifteen, where the hyperboloid coordinates are
    already at ``1e7`` and the model runs out of digits.
    """
    domain = build()
    wide = group_window(domain.model, domain.pairings, domain.interior_point, 5)
    for _ in range(25):
        x, y = domain.sample_uniform(rng), domain.sample_uniform(rng)
        brute = min(domain.model.distance(x, domain.model.apply(element, y)) for element in wide)
        assert domain.distance(x, y) == pytest.approx(brute, abs=1e-9)


def test_the_window_grows_when_a_pair_needs_it():
    """A certificate that fails must widen the search rather than round down."""
    domain = FundamentalDomain.hexagon(1.0)
    domain._grow_window(0.0)  # deliberately useless: identity only
    assert len(domain._radius_window) == 1
    corner, opposite = domain.vertices[0], domain.vertices[3]
    measured = domain.distance(corner, opposite)
    assert len(domain._radius_window) > 1
    assert measured == pytest.approx(FundamentalDomain.hexagon(1.0).distance(corner, opposite))


def test_orbit_still_truncates_by_word_length():
    """`make_periodic` wants images, not the nearest one, so it keeps its own knob."""
    hexagon = FundamentalDomain.hexagon(1.0)
    assert len(hexagon.orbit([0.0, 0.0], 1)) < len(hexagon.orbit([0.0, 0.0], 3))


def test_the_deduplication_survives_a_long_hyperbolic_word():
    """Rounded matrices stop deduplicating once the entries are large.

    The failure this guards is not hypothetical: keyed on the matrix, the
    breadth-first search over the genus-2 group converged at word length eight
    and then re-exploded at eleven, because ``cosh R ~ 1e4`` had eaten the last
    digits that made two spellings of one element compare equal. Keyed on where
    the element sends the centre, the count stays finite.
    """
    surface = FundamentalDomain.genus(2)
    depths = (3, 4, 5)
    counts = [
        len(group_window(surface.model, surface.pairings, surface.interior_point, depth))
        for depth in depths
    ]
    # Four generators and their inverses, so the ball of word length n holds at
    # most what the *free* group on four generators holds; a surface group has
    # a relation, so it holds strictly less. Anything above the free bound is
    # the search failing to recognise elements it has already seen.
    free = [1 + 8 * (7**depth - 1) // 6 for depth in depths]
    assert counts[0] < counts[1] < counts[2]
    assert all(count <= bound for count, bound in zip(counts, free, strict=True))


# ---------------------------------------------------------------------------
# Presentations that are not surfaces
# ---------------------------------------------------------------------------


def octagon(disc_radius):
    """A regular octagon in the Poincare disc at a chosen chart radius."""
    angles = 2 * np.pi * np.arange(8) / 8
    return disc_radius * np.column_stack([np.cos(angles), np.sin(angles)])


GENUS_TWO_WORD = [(0, 1), (1, 1), (0, -1), (1, -1), (2, 1), (3, 1), (2, -1), (3, -1)]


def test_the_right_radius_is_accepted():
    """Control for the two rejections below: the same word does work somewhere."""
    model = HyperbolicPlane()
    correct = float(np.tanh(np.arccosh(1 / np.tan(np.pi / 8) ** 2) / 2))
    corners = octagon(correct)
    domain = FundamentalDomain(
        corners, _pairings_from_word(model, corners, GENUS_TWO_WORD), model=model
    )
    assert domain.topology.genus == 2


@pytest.mark.parametrize("disc_radius", [0.70, 0.92])
def test_an_octagon_of_the_wrong_size_fails_poincare(disc_radius):
    """A regular hyperbolic polygon's angle shrinks as it grows, so only one size works.

    Too small and the corners are too wide, so the copies overlap; too large and
    they are too narrow, so they leave a gap. Either way the quotient is not a
    surface — and before the corner walk existed, either way the domain
    constructed happily and produced a process on nothing in particular.
    """
    model = HyperbolicPlane()
    corners = octagon(disc_radius)
    pairings = _pairings_from_word(model, corners, GENUS_TWO_WORD)
    with pytest.raises(ValueError, match=r"angles sum to|cone point"):
        FundamentalDomain(corners, pairings, model=model)


def test_a_square_glued_to_itself_by_a_quarter_turn_is_an_orbifold():
    """Free is the condition; orientation-preserving never was.

    A quarter turn is an isometry with determinant ``+1`` and an orthogonal
    linear part, so it passed every test the package made through 0.3.0. It has
    a fixed point, so the quotient has a cone point of order four.
    """
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    turn = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="act freely"):
        FundamentalDomain(square, [turn])


def test_a_shear_is_refused_as_a_non_isometry():
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    shear = [[1.0, 0.5, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="isometry"):
        FundamentalDomain(square, [shear])


def test_only_half_the_sides_paired_is_a_surface_with_boundary():
    """A cylinder, not a closed surface, and it has to be refused as such."""
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    across = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ValueError, match="not carried onto another side"):
        FundamentalDomain(square, [across])


def test_gauss_bonnet_is_checked_and_not_merely_computed():
    """Guard the guard.

    Poincare's angle condition catches every wrong presentation reachable from
    the shipped constructors, so Gauss-Bonnet never fires on its own and would
    be easy to break without noticing. Feeding it a deliberately wrong
    characteristic proves it is still wired up.
    """
    torus = FundamentalDomain.rectangle(2.0, 2.0)
    torus.topology = torus.topology._replace(euler_characteristic=-2)
    with pytest.raises(ValueError, match="Gauss-Bonnet fails"):
        torus._check_gauss_bonnet()


def test_an_interior_point_on_a_side_is_refused():
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    pairings = [
        [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    ]
    with pytest.raises(ValueError, match="does not orient it"):
        FundamentalDomain(square, pairings, interior=[0.0, -1.0])


def test_a_non_convex_polygon_is_refused_in_the_model_space():
    """The convexity test moved from chart turn angles to model half-spaces."""
    chevron = [[0.0, 0.0], [2.0, 1.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    with pytest.raises(ValueError, match="convex"):
        FundamentalDomain(chevron, [[[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])


def test_the_hexagons_third_pairing_is_found_though_it_is_not_a_generator():
    """Two translations generate three side pairings, and the search must find them.

    The reason the correspondence is inferred over a *group window* rather than
    over the generators: the hexagonal torus supplies two translations and has
    three pairs of opposite sides, the third pairing being their difference.
    """
    hexagon = FundamentalDomain.hexagon(1.0)
    assert len(hexagon.pairings) == 2
    assert sorted(hexagon._pairing.partner.tolist()) == [0, 1, 2, 3, 4, 5]


def test_a_flat_word_and_a_hand_written_pairing_agree():
    """`genus(1)` and `rectangle` build the same torus by different routes."""
    from_word = FundamentalDomain.genus(1)
    by_hand = FundamentalDomain.rectangle(np.sqrt(2.0), np.sqrt(2.0))
    assert from_word.volume == pytest.approx(by_hand.volume, rel=1e-9)
    assert from_word.topology == by_hand.topology
    assert isinstance(from_word.model, EuclideanPlane)
