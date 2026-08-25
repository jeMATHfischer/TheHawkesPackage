import numpy as np
import pytest
from hawkes_package.spatio_temporal.domains import Circle, Torus2D


class TestCircle:
    def test_distance_short_arc(self):
        c = Circle()
        # Points near the boundary: distance should use the short arc
        d = c.distance(np.array([0.0]), np.array([2 * np.pi - 0.1]))
        assert abs(d - 0.1) < 1e-9

    def test_distance_symmetry(self):
        c = Circle()
        d1 = c.distance(np.array([0.5]), np.array([2.0]))
        d2 = c.distance(np.array([2.0]), np.array([0.5]))
        assert abs(d1 - d2) < 1e-12

    def test_distance_zero_self(self):
        c = Circle()
        assert c.distance(np.array([1.0]), np.array([1.0])) == pytest.approx(0.0)

    def test_wrap_stays_in_range(self):
        c = Circle()
        wrapped = c.wrap(np.array([3 * np.pi]))
        assert -np.pi <= float(wrapped[0]) <= np.pi

    def test_sample_in_domain(self):
        c = Circle()
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = c.sample_uniform(rng)
            assert -np.pi <= float(s[0]) <= np.pi

    def test_volume(self):
        c = Circle(radius=1.0)
        assert c.volume == pytest.approx(2 * np.pi)


class TestTorus2D:
    def test_distance_wraps_both_axes(self):
        t = Torus2D(L1=2.0, L2=2.0)
        # Points at opposite corners should be at distance sqrt(2)*1 = sqrt(2)
        d = t.distance(np.array([-1.0, -1.0]), np.array([0.9, 0.9]))
        # min distance in each axis: min(1.9, 0.1) = 0.1
        assert d == pytest.approx(np.sqrt(0.1**2 + 0.1**2), rel=1e-6)

    def test_wrap_both_axes(self):
        t = Torus2D(L1=4.0, L2=6.0)
        wrapped = t.wrap(np.array([4.3, -3.5]))
        assert -2.0 <= wrapped[0] <= 2.0
        assert -3.0 <= wrapped[1] <= 3.0

    def test_distance_symmetry(self):
        t = Torus2D()
        x = np.array([0.5, 1.0])
        y = np.array([2.0, -1.0])
        assert abs(t.distance(x, y) - t.distance(y, x)) < 1e-12

    def test_volume(self):
        t = Torus2D(L1=3.0, L2=5.0)
        assert t.volume == pytest.approx(15.0)

    def test_sample_in_domain(self):
        t = Torus2D(L1=2.0, L2=4.0)
        rng = np.random.default_rng(1)
        for _ in range(50):
            s = t.sample_uniform(rng)
            assert -1.0 <= s[0] <= 1.0
            assert -2.0 <= s[1] <= 2.0
