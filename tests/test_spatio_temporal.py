"""Tests for the domain-aware SpatioTemporalHawkesProcess."""

import numpy as np
import pytest

from hawkes_package import (
    Circle,
    SpatioTemporalHawkesProcess,
    Torus2D,
    make_periodic,
)

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


def test_integrated_intensity_2d_uses_a_deterministic_rule(bump_spatial, exp_kernel):
    """The >=2-D path must integrate exactly, not estimate."""
    dom = Torus2D(L1=2.0, L2=3.0)
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5,
        bump_spatial,
        exp_kernel,
        domain=dom,
        monotone_temporal_kernel=True,
        rng=0,
    )
    assert p._integrated_intensity(1.0) == pytest.approx(dom.volume * 0.5, rel=1e-9)


def test_integrated_intensity_2d_with_varying_background(bump_spatial, exp_kernel):
    """A non-constant background integrates exactly, with no sampling error."""
    dom = Torus2D(L1=2 * np.pi, L2=2 * np.pi)
    p = SpatioTemporalHawkesProcess(
        lambda x: 1.0 + 0.5 * np.sin(x[0]),
        bump_spatial,
        exp_kernel,
        domain=dom,
        monotone_temporal_kernel=True,
        rng=7,
    )
    # The sine integrates to zero over a full period, leaving volume * 1.0.
    assert p._integrated_intensity(1.0) == pytest.approx(dom.volume, rel=1e-6)


# ---------------------------------------------------------------------------
# Intensity accessors
# ---------------------------------------------------------------------------


def test_intensity_matches_hand_computation(flat_base, bump_spatial, exp_kernel):
    dom = Circle()
    p = SpatioTemporalHawkesProcess(
        flat_base, bump_spatial, exp_kernel, domain=dom, monotone_temporal_kernel=True, rng=0
    )
    # Inject a known history rather than simulating one. There is no longer a
    # hidden bootstrap column to skip: every column here is a real event.
    p.Events = np.array([[1.0, 2.0], [0.5, -1.0]])

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


@pytest.mark.parametrize("dom", [Circle(), Torus2D()])
def test_user_callables_receive_a_shape_ndim_point(bump_spatial, exp_kernel, dom):
    """Every path must hand `base` a shape-(ndim,) array, never a bare float.

    The quadrature path used to pass a Python float while Monte Carlo and the
    MCMC sampler passed arrays, so no non-constant background could be written
    that worked on both.
    """
    seen = []

    def base(x):
        seen.append(np.asarray(x))
        return 0.5

    ndim = dom.bounds.shape[0]
    p = SpatioTemporalHawkesProcess(
        base, bump_spatial, exp_kernel, domain=dom, monotone_temporal_kernel=True, rng=0
    )
    p.simulate(2)
    p.intensity(1.0, np.zeros(ndim))
    assert seen, "base was never called"
    assert all(isinstance(x, np.ndarray) and x.shape == (ndim,) for x in seen)


@pytest.mark.parametrize("dom", [Circle(), Torus2D()])
def test_non_constant_background_is_usable(bump_spatial, exp_kernel, dom):
    """`base=lambda x: ... x[0] ...` must work, and port unchanged across domains."""
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5 + 0.2 * np.cos(x[0]),
        bump_spatial,
        exp_kernel,
        domain=dom,
        monotone_temporal_kernel=True,
        rng=0,
    )
    p.simulate(2)
    assert p.Events.shape == (dom.bounds.shape[0] + 1, 2)


def test_intensity_is_invariant_to_coordinate_shape(make_st_process):
    p = make_st_process(Circle())
    p.Events = np.array([[0.3, 0.6], [0.1, -0.4]])
    values = [p.intensity(1.0, s) for s in (0.15, [0.15], np.array([0.15]), np.array([[0.15]]))]
    assert values == pytest.approx([values[0]] * 4)


def test_integrated_intensity_is_deterministic(bump_spatial, exp_kernel):
    """The bound must be a bound, not an estimate.

    A Monte Carlo integral redrawn on every call made `_upper_bound` an
    unbiased estimate rather than an upper bound, and the acceptance test drew
    an independent one to compare against: P(lambda_hat > M_hat) was 0.437
    where Ogata's algorithm requires 0.
    """
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5,
        bump_spatial,
        exp_kernel,
        domain=Torus2D(),
        monotone_temporal_kernel=True,
        rng=0,
    )
    p.Events = np.array([[0.5, 0.8], [0.2, -1.0], [0.3, 1.1]])

    state = p.rng.bit_generator.state
    values = {p._integrated_intensity(1.0) for _ in range(20)}
    assert len(values) == 1, "the spatial integral must not depend on the RNG"
    assert p.rng.bit_generator.state == state, "integration must not consume the stream"

    bounds = [p._upper_bound(1.0) for _ in range(20)]
    lambdas = [p._integrated_intensity(1.0) for _ in range(20)]
    assert all(lam <= m for lam, m in zip(lambdas, bounds, strict=True))


def test_quadrature_agrees_with_a_refined_rule(bump_spatial, exp_kernel):
    """Doubling the node count must not move the answer."""
    kwargs = {
        "base": lambda x: 0.5 + 0.2 * np.cos(x[0]),
        "spatial": bump_spatial,
        "temporal": exp_kernel,
        "domain": Circle(),
        "monotone_temporal_kernel": True,
        "rng": 0,
    }
    coarse = SpatioTemporalHawkesProcess(**kwargs, n_quad=64)
    fine = SpatioTemporalHawkesProcess(**kwargs, n_quad=512)
    for p in (coarse, fine):
        p.Events = np.array([[0.4], [0.1]])
    assert coarse._integrated_intensity(1.0) == pytest.approx(
        fine._integrated_intensity(1.0), rel=1e-3
    )


def test_non_box_domain_is_rejected(bump_spatial, exp_kernel):
    """Integration runs over `bounds`, so volume must equal the box volume."""

    class Disc(Circle):
        @property
        def volume(self):
            return np.pi  # a disc, not its bounding box

    with pytest.raises(ValueError, match="bounding box"):
        SpatioTemporalHawkesProcess(lambda x: 0.5, bump_spatial, exp_kernel, domain=Disc(), rng=0)


def test_narrow_spatial_kernel_warns_about_resolution(exp_kernel):
    """A kernel narrower than the node spacing silently distorts the rate.

    This replaces inspecting the error estimate `quad` returned and the code
    discarded: with a width-0.005 kernel `quad` returned exactly the
    background-only integral, so the excitation was invisible to the temporal
    thinning while the spatial sampler still saw it.
    """
    narrow = lambda d: 20.0 * np.exp(-(d**2) / 5e-5)
    with pytest.warns(UserWarning, match="too narrow"):
        SpatioTemporalHawkesProcess(
            lambda x: 0.5, narrow, exp_kernel, domain=Circle(), rng=0, n_quad=32
        )


def test_no_resolution_warning_at_a_higher_node_count(exp_kernel, recwarn):
    narrow = lambda d: 20.0 * np.exp(-(d**2) / 5e-5)
    SpatioTemporalHawkesProcess(
        lambda x: 0.5, narrow, exp_kernel, domain=Circle(), rng=0, n_quad=8192
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_periodic_kernel_composes_with_the_process(exp_kernel):
    """Regression: `make_periodic` could not be used as `spatial=` at all.

    It returns a two-point kernel, but `spatial` was called with a single
    geodesic distance, so the second event raised TypeError. README presents
    `make_periodic` as the way to build a domain-respecting kernel.
    """
    domain = Circle()
    kernel = make_periodic(lambda d: np.exp(-(d**2)), domain)
    p = SpatioTemporalHawkesProcess(
        lambda x: 0.5,
        kernel,
        exp_kernel,
        domain=domain,
        monotone_temporal_kernel=True,
        rng=0,
    )
    p.simulate(3)
    assert p.Events.shape == (2, 3)

    # and the intensity uses the image sum, not a bare geodesic distance
    p.Events = np.array([[1.0], [0.4]])
    x = np.array([0.15])
    expected = 0.5 + exp_kernel(2.0 - 1.0) * kernel(x, np.array([0.4]))
    assert p.intensity(2.0, x) == pytest.approx(expected)
