"""Tests for spatial domains, including the contract every SpatialDomain owes."""

import numpy as np
import pytest

from hawkes_package import Circle, SpatialDomain, Torus2D


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


def test_bounds_shape(domain):
    bounds = domain.bounds
    assert bounds.ndim == 2
    assert bounds.shape[1] == 2
    assert np.all(bounds[:, 0] < bounds[:, 1])


def test_volume_matches_bounding_box(domain):
    widths = domain.bounds[:, 1] - domain.bounds[:, 0]
    assert domain.volume == pytest.approx(float(np.prod(widths)))


def test_wrap_is_idempotent(domain, rng):
    for _ in range(20):
        x = rng.uniform(-10, 10, size=domain.bounds.shape[0])
        once = domain.wrap(x)
        twice = domain.wrap(once)
        np.testing.assert_allclose(twice, once)


def test_wrap_lands_inside_bounds(domain, rng):
    bounds = domain.bounds
    for _ in range(20):
        x = rng.uniform(-10, 10, size=bounds.shape[0])
        wrapped = np.atleast_1d(domain.wrap(x))
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
    """No two points can be further apart than half the domain in each axis."""
    widths = domain.bounds[:, 1] - domain.bounds[:, 0]
    max_dist = float(np.sqrt(np.sum((widths / 2) ** 2)))
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
