"""The immersions must identify exactly the points the gluings identify.

An immersion of a quotient surface is a map on the *chart* that happens to
factor through the quotient. So the assertion that it is the right map is not
about the boundary at all: it is that the map is invariant under every element
of the deck group, at every chart point. That is what these tests check, and it
is strictly stronger than checking the glued edges agree.

The Klein bottle is the case worth the trouble. Its two pairings differ from the
torus's by exactly one sign, and lining the immersion's base circle up with the
wrong axis produces a picture that still renders, still looks like a Klein
bottle and still animates -- with the intensity mirrored across one seam and
torn across the other. `test_swapping_the_klein_axes_breaks_the_gluing` is the
negative control that makes the axis choice falsifiable rather than merely
asserted.
"""

import numpy as np
import pytest

from hawkes_package.spatio_temporal import Circle, FundamentalDomain, Sphere, Torus2D
from hawkes_package.viz import embed

WIDTH, HEIGHT = 3.0, 5.0


def _apply(domain, matrix, points):
    """Act by an ambient pairing matrix on a stack of chart points."""
    return np.array([domain.model.apply(matrix, p) for p in points])


def _words(pairings):
    """A handful of deck-group elements: the generators, their inverses, some products."""
    a, b = (np.asarray(g, dtype=float) for g in pairings)
    inv_a, inv_b = np.linalg.inv(a), np.linalg.inv(b)
    return [a, b, inv_a, inv_b, a @ b, b @ a, b @ b, a @ a @ b, inv_b @ a @ b]


def _chart_sample(domain, n=400, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = domain.bounds[:, 0], domain.bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, 2))


# ---------------------------------------------------------------------------
# The Klein bottle
# ---------------------------------------------------------------------------


@pytest.fixture
def bottle():
    return FundamentalDomain.klein_bottle(WIDTH, HEIGHT)


def test_klein_immersion_respects_every_pairing(bottle):
    """The map must be constant on deck-group orbits, not merely on the boundary."""
    surface = embed(bottle)
    points = _chart_sample(bottle)
    here = surface.ambient(points)
    for word in _words(bottle.pairings):
        there = surface.ambient(_apply(bottle, word, points))
        assert np.abs(here - there).max() < 1e-12


def test_klein_immersion_identifies_exactly_the_glued_edges(bottle):
    """The two identifications, spelled out: the glide flips, the translation does not."""
    surface = embed(bottle)

    # The glide reflection (x, -h/2) ~ (-x, +h/2): crossing the top edge comes
    # back through the bottom one mirrored.
    across = np.linspace(-WIDTH / 2, WIDTH / 2, 200)
    bottom = np.column_stack([across, np.full_like(across, -HEIGHT / 2)])
    top = np.column_stack([-across, np.full_like(across, HEIGHT / 2)])
    assert np.abs(surface.ambient(bottom) - surface.ambient(top)).max() < 1e-12

    # The plain translation (-w/2, y) ~ (+w/2, y): no flip on this pair.
    along = np.linspace(-HEIGHT / 2, HEIGHT / 2, 200)
    left = np.column_stack([np.full_like(along, -WIDTH / 2), along])
    right = np.column_stack([np.full_like(along, WIDTH / 2), along])
    assert np.abs(surface.ambient(left) - surface.ambient(right)).max() < 1e-12


def test_swapping_the_klein_axes_breaks_the_gluing(bottle):
    """The negative control: with the base circle on the flipped axis, nothing glues.

    Without this the axis assignment is unfalsifiable. A transposed map is still
    smooth, still periodic and still renders -- it just glues the rectangle into
    a torus instead, and the intensity is torn across the seam the glide was
    supposed to close.
    """
    surface = embed(bottle)
    points = _chart_sample(bottle)
    swapped = surface.ambient(points[:, ::-1])

    glide = next(g for g in bottle.pairings if np.linalg.det(np.asarray(g)[:2, :2]) < 0)
    moved = surface.ambient(_apply(bottle, glide, points)[:, ::-1])
    assert np.abs(swapped - moved).max() > 0.1


def test_klein_immersion_is_the_figure_eight(bottle):
    """The fibre reverses after one turn of the base circle: F(u + 2 pi, v) = F(u, -v).

    Checked through the chart, where one turn of the base circle is a shift of
    `height` along the glide axis and the reversal is a negation across the
    flip axis.
    """
    surface = embed(bottle)
    points = _chart_sample(bottle)
    turned = points + np.array([0.0, HEIGHT])
    mirrored = points * np.array([-1.0, 1.0])
    assert np.abs(surface.ambient(turned) - surface.ambient(mirrored)).max() < 1e-12


# ---------------------------------------------------------------------------
# The flat torus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain",
    [Torus2D(WIDTH, HEIGHT), FundamentalDomain.rectangle(WIDTH, HEIGHT)],
    ids=["Torus2D", "FundamentalDomain.rectangle"],
)
def test_torus_immersion_respects_both_lattice_translations(domain):
    """Both generators are pure translations, so the donut must be doubly periodic."""
    surface = embed(domain)
    points = _chart_sample(domain)
    here = surface.ambient(points)
    for shift in ([WIDTH, 0.0], [0.0, HEIGHT], [WIDTH, HEIGHT], [-WIDTH, 2 * HEIGHT]):
        assert np.abs(here - surface.ambient(points + shift)).max() < 1e-12


def test_the_two_torus_presentations_draw_the_same_picture():
    """The hand-written class and the general machinery must agree on the geometry."""
    points = _chart_sample(Torus2D(WIDTH, HEIGHT))
    hand = embed(Torus2D(WIDTH, HEIGHT)).ambient(points)
    general = embed(FundamentalDomain.rectangle(WIDTH, HEIGHT)).ambient(points)
    assert np.abs(hand - general).max() < 1e-12


def test_the_donut_tube_cannot_swallow_the_hole():
    """A tall thin torus is drawn out of proportion rather than self-intersecting.

    At a tube-to-centre ratio of 1 the tube closes the hole; past it the surface
    passes through itself and renders as an opaque blob with the intensity
    hidden inside.
    """
    surface = embed(Torus2D(1.0, 40.0))  # aspect ratio 40, far past the clamp
    radial = np.linalg.norm(surface.ambient(_chart_sample(Torus2D(1.0, 40.0)))[:, :2], axis=1)
    assert radial.min() > 0.0

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        embed(Torus2D(1.0, 1.0), tube_ratio=1.5)


# ---------------------------------------------------------------------------
# The two spherical surfaces
# ---------------------------------------------------------------------------


def test_sphere_embedding_is_the_model_lift():
    """Exact equality, not a tolerance: the sphere formula must not be written twice.

    `SphericalPlane.lift` is what every geometric predicate in the package runs
    through. A second copy here could drift from it, and the drift would show as
    events plotted slightly off the surface they were drawn on.
    """
    domain = Sphere(2.5)
    surface = embed(domain)
    points = _chart_sample(domain)
    assert np.array_equal(surface.ambient(points), domain.model.lift_many(points))
    assert surface.isometric


def test_sphere_grid_range_closes_the_seam():
    """Both endpoints of the longitude range name the same points of the sphere."""
    surface = embed(Sphere())
    assert surface.urange == (0.0, np.pi)
    assert surface.vrange == (-np.pi, np.pi)

    colatitude = np.linspace(0.0, np.pi, 50)
    west = surface.ambient(np.column_stack([colatitude, np.full_like(colatitude, -np.pi)]))
    east = surface.ambient(np.column_stack([colatitude, np.full_like(colatitude, np.pi)]))
    assert np.abs(west - east).max() < 1e-15


def test_projective_plane_is_drawn_on_the_whole_sphere():
    """Its polygon is a hemisphere; the picture is the double cover, so it is not.

    The covering map is a local isometry, which is what makes the double cover an
    exact picture rather than a distorted one -- and is why no Boy surface is
    attempted.
    """
    domain = FundamentalDomain.projective_plane()
    surface = embed(domain)
    assert surface.urange == (0.0, np.pi)
    assert surface.vrange == (-np.pi, np.pi)
    assert surface.isometric

    # The polygon itself reaches only the northern hemisphere, so the drawn
    # range is strictly larger than `domain.bounds` -- the thing being asserted.
    assert domain.bounds[0, 1] < np.pi

    points = _chart_sample(Sphere())
    assert np.array_equal(surface.ambient(points), domain.model.lift_many(points))


def test_antipodal_chart_points_have_opposite_images():
    """The identification the picture displays: the deck group is the antipodal map."""
    surface = embed(FundamentalDomain.projective_plane())
    points = _chart_sample(Sphere())
    antipodes = np.column_stack([np.pi - points[:, 0], points[:, 1] - np.pi])
    assert np.abs(surface.ambient(points) + surface.ambient(antipodes)).max() < 1e-12


# ---------------------------------------------------------------------------
# What is refused, and why
# ---------------------------------------------------------------------------


def test_a_one_dimensional_domain_is_refused():
    with pytest.raises(ValueError, match="no immersion is implemented for Circle"):
        embed(Circle())


def test_a_non_rectangular_flat_polygon_is_refused():
    """The hexagonal torus is a flat torus, but not one the donut map applies to."""
    with pytest.raises(ValueError, match="centred axis-aligned rectangle"):
        embed(FundamentalDomain.hexagon())


def test_the_diamond_klein_bottle_is_refused():
    """`crosscaps(2)` is a Klein bottle on a diamond, which is a different chart.

    Refused rather than mis-drawn: its four pairings are all orientation
    reversing and its polygon is not axis aligned, so neither the donut nor the
    figure-8 affine map applies.
    """
    with pytest.raises(ValueError, match="centred axis-aligned rectangle"):
        embed(FundamentalDomain.crosscaps(2))


@pytest.mark.slow
def test_a_hyperbolic_surface_is_refused_by_name():
    """Hilbert's theorem: no isometric picture exists, so none is invented."""
    with pytest.raises(ValueError, match="Hilbert"):
        embed(FundamentalDomain.genus(2))
