"""Tests for the domain-aware SpatioTemporalHawkesProcess."""

import numpy as np
import pytest

from hawkes_package import Circle, SpatioTemporalHawkesProcess, Torus2D

ONE_D = [pytest.param(Circle(), id="circle"), pytest.param(Circle(radius=2.0), id="circle-r2")]


# ---------------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dom", ONE_D)
def test_circle_end_to_end(make_st_process, dom):
    p = make_st_process(dom)
    p.simulate(5)
    assert p.Events.shape == (2, 5)  # time + one spatial coordinate
    assert np.all(np.diff(p.Events[0, :]) > 0)
    lo, hi = dom.bounds[0]
    assert np.all(p.Events[1, :] >= lo - 1e-9)
    assert np.all(p.Events[1, :] <= hi + 1e-9)


@pytest.mark.slow
def test_torus_end_to_end(make_st_process):
    """The 2-D path exercises the Monte Carlo integrator, untested before 0.2.0."""
    dom = Torus2D()
    p = make_st_process(dom)
    p.simulate(4)
    assert p.Events.shape == (3, 4)  # time + two spatial coordinates
    assert np.all(np.diff(p.Events[0, :]) > 0)
    bounds = dom.bounds
    for d in range(2):
        assert np.all(p.Events[1 + d, :] >= bounds[d, 0] - 1e-9)
        assert np.all(p.Events[1 + d, :] <= bounds[d, 1] + 1e-9)


def test_non_monotone_kernel_path(flat_base, bump_spatial, triangular_kernel):
    """monotone_temporal_kernel=False is the default and used to crash outright.

    scipy's fmin returns a shape-(1,) array, and float() of that raises under
    NumPy 2, so the constructor failed before it ever simulated anything.
    """
    p = SpatioTemporalHawkesProcess(
        flat_base, bump_spatial, triangular_kernel, domain=Circle(), rng=0
    )
    assert isinstance(p.temporal_extremum, float)
    assert p.temporal_extremum == pytest.approx(0.5, abs=1e-3)
    p.simulate(3)
    assert p.Events.shape == (2, 3)


def test_repeated_simulate_continues_the_realisation(make_st_process):
    p = make_st_process(Circle())
    p.simulate(3)
    first = p.Events.copy()
    p.simulate(2)
    assert p.Events.shape == (2, 5)
    assert p.Sim_num == 5
    np.testing.assert_array_equal(p.Events[:, :3], first)


def test_reproducible_with_seed(make_st_process):
    a, b = make_st_process(Circle(), seed=11), make_st_process(Circle(), seed=11)
    a.simulate(4)
    b.simulate(4)
    np.testing.assert_array_equal(a.Events, b.Events)


def test_different_seeds_differ(make_st_process):
    a, b = make_st_process(Circle(), seed=11), make_st_process(Circle(), seed=12)
    a.simulate(4)
    b.simulate(4)
    assert not np.array_equal(a.Events, b.Events)


# ---------------------------------------------------------------------------
# Integrated intensity
# ---------------------------------------------------------------------------


def test_integrated_intensity_1d_matches_analytic(bump_spatial, exp_kernel):
    """With a constant background and no history, the integral is volume * mu."""
    dom = Circle()
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5,
        bump_spatial,
        exp_kernel,
        domain=dom,
        monotone_temporal_kernel=True,
        rng=0,
    )
    assert p._integrated_intensity(1.0) == pytest.approx(dom.volume * 0.5, rel=1e-6)


def test_integrated_intensity_2d_monte_carlo_branch(bump_spatial, exp_kernel):
    """The >=2-D branch must apply the domain volume to the sample mean."""
    dom = Torus2D(L1=2.0, L2=3.0)
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5,
        bump_spatial,
        exp_kernel,
        domain=dom,
        monotone_temporal_kernel=True,
        rng=0,
    )
    # A constant integrand makes the Monte Carlo estimate exact, isolating the
    # volume factor from the sampling error.
    assert p._integrated_intensity(1.0) == pytest.approx(dom.volume * 0.5, rel=1e-9)


@pytest.mark.statistical
def test_integrated_intensity_2d_with_varying_background(bump_spatial, exp_kernel):
    """A non-constant background: Monte Carlo now carries real sampling error."""
    dom = Torus2D(L1=2 * np.pi, L2=2 * np.pi)

    def base(x):
        return 1.0 + 0.5 * np.sin(float(np.ravel(x)[0]))

    p = SpatioTemporalHawkesProcess(
        base, bump_spatial, exp_kernel, domain=dom, monotone_temporal_kernel=True, rng=7
    )
    # The sine integrates to zero over a full period, leaving volume * 1.0.
    estimates = [p._integrated_intensity(1.0) for _ in range(20)]
    assert float(np.mean(estimates)) == pytest.approx(dom.volume, rel=0.1)


# ---------------------------------------------------------------------------
# Intensity accessors
# ---------------------------------------------------------------------------


def test_intensity_matches_hand_computation(flat_base, bump_spatial, exp_kernel):
    dom = Circle()
    p = SpatioTemporalHawkesProcess(
        flat_base, bump_spatial, exp_kernel, domain=dom, monotone_temporal_kernel=True, rng=0
    )
    # Inject a known history rather than simulating one.
    p.Events = np.array([[0.0, 1.0, 2.0], [0.0, 0.5, -1.0]])

    t, x = 3.0, np.array([0.2])
    expected = 0.5
    for time, coord in [(1.0, 0.5), (2.0, -1.0)]:
        expected += exp_kernel(t - time) * bump_spatial(dom.distance(x, np.array([coord])))
    assert p.intensity(t, x) == pytest.approx(expected)


def test_intensity_is_floored_at_zero(bump_spatial, exp_kernel):
    """A negative background must not produce a negative intensity."""
    p = SpatioTemporalHawkesProcess(
        lambda x: -5.0,
        bump_spatial,
        exp_kernel,
        domain=Circle(),
        monotone_temporal_kernel=True,
        rng=0,
    )
    assert p.intensity(1.0, np.array([0.0])) == 0.0


def test_intensity_over_interval_orientation(make_st_process):
    """Rows index space, columns index time: the contourf layout.

    Transposing this is the single easiest mistake to make here, so the shape
    is pinned explicitly rather than inferred.
    """
    p = make_st_process(Circle())
    p.simulate(3)
    times, points, intensity = p.intensity_over_interval(np.linspace(0, 2, 7))
    assert points.shape == (200, 1)  # default 1-D grid
    assert intensity.shape == (points.shape[0], times.shape[0])
    assert np.all(intensity >= 0)


def test_intensity_over_interval_merges_event_times(make_st_process):
    p = make_st_process(Circle())
    p.simulate(3)
    times, _, _ = p.intensity_over_interval(np.linspace(0, 1, 5))
    assert np.isin(p.Events[0, :], times).all()
    assert np.all(np.diff(times) > 0), "times must be sorted and de-duplicated"


def test_intensity_over_interval_accepts_explicit_points(make_st_process):
    p = make_st_process(Circle())
    p.simulate(2)
    pts = np.array([[-1.0], [0.0], [1.0]])
    times, points, intensity = p.intensity_over_interval(np.linspace(0, 1, 4), points=pts)
    np.testing.assert_array_equal(points, pts)
    assert intensity.shape == (3, times.shape[0])


def test_intensity_over_interval_agrees_with_intensity(make_st_process):
    """The grid accessor and the pointwise accessor must not diverge."""
    p = make_st_process(Circle())
    p.simulate(2)
    pts = np.array([[-0.5], [0.7]])
    times, points, grid = p.intensity_over_interval(np.linspace(0, 1, 4), points=pts)
    for i in range(points.shape[0]):
        for j, t in enumerate(times):
            assert grid[i, j] == pytest.approx(p.intensity(t, points[i]))


def test_intensity_over_interval_requires_points_in_2d(make_st_process):
    p = make_st_process(Torus2D())
    with pytest.raises(ValueError, match="points is required"):
        p.intensity_over_interval(np.linspace(0, 1, 3))


def test_intensity_over_interval_rejects_wrong_point_width(make_st_process):
    p = make_st_process(Circle())
    with pytest.raises(ValueError, match=r"shape \(n_x, 1\)"):
        p.intensity_over_interval(np.linspace(0, 1, 3), points=np.zeros((4, 2)))
