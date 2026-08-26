"""Tests for ``make_periodic``, which had no coverage at all before 0.2.0."""

import numpy as np
import pytest

from hawkes_package import Circle, FundamentalDomain, Torus2D, make_periodic

from .test_domains import Interval


def gaussian(d):
    return float(np.exp(-((d) ** 2)))


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------


def test_circle_equals_explicit_image_sum():
    domain = Circle()
    n_images = 3
    k = make_periodic(gaussian, domain, n_images=n_images)
    x, y = 0.3, -1.2
    expected = sum(gaussian(abs(x - y - n * domain.volume)) for n in range(-n_images, n_images + 1))
    assert k(x, y) == pytest.approx(expected)


def test_circle_accepts_array_coordinates():
    """Regression: process.py passes shape-(1,) event coordinates, and under
    NumPy 2 ``float()`` on those raised TypeError."""
    k = make_periodic(gaussian, Circle())
    scalar = k(0.3, -1.2)
    arrays = k(np.array([0.3]), np.array([-1.2]))
    assert arrays == pytest.approx(scalar)


def test_circle_kernel_is_periodic():
    domain = Circle()
    k = make_periodic(gaussian, domain, n_images=6)
    base = k(0.4, -0.7)
    shifted = k(0.4 + domain.volume, -0.7)
    assert shifted == pytest.approx(base, rel=1e-9)


def test_circle_kernel_is_symmetric():
    k = make_periodic(gaussian, Circle())
    assert k(0.4, -0.7) == pytest.approx(k(-0.7, 0.4))


def test_circle_image_count_converges():
    """For a fast-decaying kernel, more images must not change the value."""
    few = make_periodic(gaussian, Circle(), n_images=3)
    many = make_periodic(gaussian, Circle(), n_images=6)
    for x, y in [(0.0, 0.0), (0.3, -1.2), (np.pi, -np.pi)]:
        assert few(x, y) == pytest.approx(many(x, y), abs=1e-9)


def test_circle_radius_changes_the_period():
    small = make_periodic(gaussian, Circle(radius=1.0), n_images=2)
    large = make_periodic(gaussian, Circle(radius=5.0), n_images=2)
    # a far-apart pair is bridged by an image on the small circle but not the large
    assert small(0.0, 5.0) > large(0.0, 5.0)


# ---------------------------------------------------------------------------
# Torus2D
# ---------------------------------------------------------------------------


def test_torus_equals_explicit_image_sum():
    domain = Torus2D(L1=2.0, L2=3.0)
    n_images = 2
    k = make_periodic(gaussian, domain, n_images=n_images)
    x, y = np.array([0.2, -0.4]), np.array([-0.9, 1.1])
    expected = sum(
        gaussian(float(np.linalg.norm(x - y - np.array([n1 * 2.0, n2 * 3.0]))))
        for n1 in range(-n_images, n_images + 1)
        for n2 in range(-n_images, n_images + 1)
    )
    assert k(x, y) == pytest.approx(expected)


def test_torus_sums_over_the_full_image_lattice():
    """A (2n+1)^2 lattice: with a constant kernel the total counts the images."""
    n_images = 2
    k = make_periodic(lambda d: 1.0, Torus2D(), n_images=n_images)
    assert k(np.array([0.0, 0.0]), np.array([0.1, 0.1])) == pytest.approx((2 * n_images + 1) ** 2)


@pytest.mark.parametrize("axis", [0, 1])
def test_torus_kernel_is_periodic_in_each_axis(axis):
    domain = Torus2D(L1=2.0, L2=3.0)
    k = make_periodic(gaussian, domain, n_images=5)
    x = np.array([0.2, -0.4])
    y = np.array([-0.9, 1.1])
    shift = np.zeros(2)
    shift[axis] = domain.L1 if axis == 0 else domain.L2
    assert k(x + shift, y) == pytest.approx(k(x, y), rel=1e-9)


def test_torus_kernel_is_symmetric():
    k = make_periodic(gaussian, Torus2D())
    x, y = np.array([0.2, -0.4]), np.array([-0.9, 1.1])
    assert k(x, y) == pytest.approx(k(y, x))


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


def test_generic_domain_uses_its_own_metric():
    """An unrecognised domain falls back to a single term via domain.distance."""
    domain = Interval(0.0, 1.0)
    k = make_periodic(gaussian, domain)
    x, y = np.array([0.2]), np.array([0.9])
    assert k(x, y) == pytest.approx(gaussian(domain.distance(x, y)))


def test_generic_fallback_ignores_n_images():
    """The fallback is a single evaluation, so image count cannot matter."""
    domain = Interval(0.0, 1.0)
    a = make_periodic(gaussian, domain, n_images=1)
    b = make_periodic(gaussian, domain, n_images=9)
    x, y = np.array([0.2]), np.array([0.9])
    assert a(x, y) == pytest.approx(b(x, y))


def test_periodic_kernel_is_marked_pairwise():
    """The marker is how the process knows to pass both points, not a distance."""
    k = make_periodic(gaussian, Circle())
    assert getattr(k, "pairwise", False) is True


def test_circle_kernel_stays_periodic_far_outside_the_window():
    """Regression: beyond n_images periods the nearest image left the window.

    The sum was taken about the raw difference, so once |x - y| exceeded
    n_images * period the kernel decayed to zero instead of staying periodic --
    reachable in practice because the MCMC chain routinely wandered many
    periods out before its proposals were constrained.
    """
    domain = Circle()
    k = make_periodic(gaussian, domain, n_images=3)
    reference = k(0.4, -0.7)
    for periods in (1, 5, 12, 40):
        assert k(0.4 + periods * domain.volume, -0.7) == pytest.approx(reference, rel=1e-9)


def test_torus_kernel_stays_periodic_far_outside_the_window():
    domain = Torus2D(L1=2.0, L2=3.0)
    k = make_periodic(gaussian, domain, n_images=2)
    x, y = np.array([0.2, -0.4]), np.array([-0.9, 1.1])
    reference = k(x, y)
    far = x + np.array([9 * domain.L1, 7 * domain.L2])
    assert k(far, y) == pytest.approx(reference, rel=1e-9)


# ---------------------------------------------------------------------------
# Domains that supply their own deck group, via `orbit`
# ---------------------------------------------------------------------------


def test_fundamental_domain_kernel_is_periodic():
    """A domain implementing `orbit` gets a real image sum, not the fallback.

    Which is the whole point of the hook: without it a `FundamentalDomain` would
    take the generic branch, and `make_periodic` would silently return an
    unperiodised single evaluation.
    """
    hexagon = FundamentalDomain.hexagon(1.0)
    k = make_periodic(gaussian, hexagon)
    x, y = np.array([0.2, -0.35]), np.array([0.4, 0.1])
    reference = k(x, y)
    for image in hexagon.orbit(y, n_images=2):
        assert k(x, image) == pytest.approx(reference, rel=1e-9)


def test_fundamental_domain_kernel_stays_periodic_far_outside_the_window():
    hexagon = FundamentalDomain.hexagon(1.0)
    k = make_periodic(gaussian, hexagon, n_images=2)
    x, y = np.array([0.2, -0.35]), np.array([0.4, 0.1])
    reference = k(x, y)
    lattice_step = np.array([9 * np.sqrt(3.0), 0.0])
    assert k(x + lattice_step, y) == pytest.approx(reference, rel=1e-9)


def test_fundamental_domain_kernel_exceeds_the_unperiodised_one():
    """The image sum adds the neighbours' contributions; it never subtracts."""
    hexagon = FundamentalDomain.hexagon(1.0)
    k = make_periodic(gaussian, hexagon)
    x, y = np.array([0.2, -0.35]), np.array([0.4, 0.1])
    assert k(x, y) > gaussian(hexagon.distance(x, y))


def test_rectangle_kernel_matches_the_torus_one():
    """The two spellings of the same quotient must periodise identically."""
    length_1, length_2 = 2.0, 3.0
    polygon = make_periodic(gaussian, FundamentalDomain.rectangle(length_1, length_2), n_images=3)
    torus = make_periodic(gaussian, Torus2D(L1=length_1, L2=length_2), n_images=3)
    x, y = np.array([0.2, -0.4]), np.array([-0.9, 1.1])
    # Not equal: the two sum over different windows of the same lattice -- the
    # torus branch over a square of offsets, the orbit branch over words of
    # bounded length, which is a diamond. Both cover the terms that carry the
    # mass, so they agree to well past what the kernel's tail contributes.
    assert polygon(x, y) == pytest.approx(torus(x, y), rel=1e-9)


def test_periodic_kernel_over_orbit_is_marked_pairwise():
    assert getattr(make_periodic(gaussian, FundamentalDomain.hexagon(1.0)), "pairwise", False)
