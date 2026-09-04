"""The frames must be the intensity, not a fast approximation of it.

`test_the_factorised_path_is_the_intensity_hook_exactly` is the linchpin. The
fast path hoists the time-independent factors out of the frame loop, which is a
30-to-40-fold speedup and is also the shape of defect `CONTRIBUTING.md` names:
a second, faster expression for the intensity is a second *model*, and the
difference shows up as a plausible-but-wrong picture rather than as an error.

The agreement asserted is **exact**, not approximate. That is not fussiness. The
factorisation reduces the same products, in the same order, over an array of the
same length, so it agrees to the last bit; a tolerance would let a genuinely
different reduction pass. In particular a BLAS matrix-vector product over the
whole grid at once is faster still and is *not* bit-identical -- measured 454
one-ulp disagreements in 5 625 values against 0 for the loop -- so this test is
what stops the loop in `_field._factorised` from being tidied away.

The one thing not asserted here is that the matvec differs, even though it does.
Whether a particular BLAS sums in numpy's order is a property of the build, and
`CLAUDE.md` is explicit that the spatio-temporal path's last bits vary across
SciPy and BLAS builds. Exact agreement with the hook is portable; exact
*disagreement* with a third implementation is not.
"""

import numpy as np
import pytest

from hawkes_package.spatio_temporal import FundamentalDomain, Torus2D
from hawkes_package.spatio_temporal.kernels import make_periodic
from hawkes_package.viz import embed, event_opacities, intensity_frames

from .conftest import HEIGHT, WIDTH, seeded_process

RESOLUTION = (9, 9)


def _antipodes(points):
    """The antipodal images of chart points on the sphere.

    ``(theta, phi) -> (pi - theta, phi -/+ pi)``. Note this is *not* a reversal
    of the grid, which is what makes the symmetry worth checking through the
    chart rather than through array indices.
    """
    longitude = points[:, 1]
    return np.column_stack(
        [np.pi - points[:, 0], np.where(longitude > 0, longitude - np.pi, longitude + np.pi)]
    )


def _hook_values(process, frames):
    """The same grid, evaluated one point at a time through `process.intensity`."""
    return np.array(
        [
            [process.intensity(float(t), point) for point in frames.chart.reshape(-1, 2)]
            for t in frames.times
        ]
    ).reshape(frames.values.shape)


# ---------------------------------------------------------------------------
# The linchpin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["torus", "sphere"])
def test_the_factorised_path_is_the_intensity_hook_exactly(request, fixture, frame_times):
    process = request.getfixturevalue(fixture)
    frames = intensity_frames(process, frame_times, resolution=RESOLUTION)
    assert frames.factorised
    assert np.array_equal(frames.values, _hook_values(process, frames))


@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["bottle", "projective"])
def test_the_factorised_path_is_exact_on_the_quotient_surfaces(request, fixture, frame_times):
    process = request.getfixturevalue(fixture)
    frames = intensity_frames(process, frame_times, resolution=RESOLUTION)
    assert np.array_equal(frames.values, _hook_values(process, frames))


def test_fast_and_slow_paths_agree_exactly(torus, frame_times):
    fast = intensity_frames(torus, frame_times, resolution=RESOLUTION, fast=True)
    slow = intensity_frames(torus, frame_times, resolution=RESOLUTION, fast=False)
    assert not slow.factorised
    assert np.array_equal(fast.values, slow.values)


def test_a_pairwise_kernel_still_factorises(frame_times):
    """An image sum does not factor through a distance, but it is still time-independent.

    `make_periodic` returns a `PairwiseKernel`, and it is the documented way to
    get a periodic kernel on a torus -- the domain most likely to be drawn. The
    cache is built through `process._spatial_at`, which *is* the dispatch, so
    both kernel protocols are hoisted by the same code and neither needs a
    branch here.
    """
    domain = Torus2D(WIDTH, HEIGHT)
    kernel = make_periodic(lambda d: 0.8 * np.exp(-2.0 * d), domain, 2)
    assert kernel.pairwise is True
    process = seeded_process(domain, spatial=kernel)

    frames = intensity_frames(process, frame_times, resolution=RESOLUTION)
    assert frames.factorised
    assert np.array_equal(frames.values, _hook_values(process, frames))


# ---------------------------------------------------------------------------
# The conventions the intensity carries
# ---------------------------------------------------------------------------


def test_the_sum_is_over_strictly_earlier_events(torus):
    """At an event time the intensity is the left limit, so that event is excluded."""
    at_event = torus.events[0, 2]
    frames = intensity_frames(torus, [at_event], resolution=RESOLUTION)
    assert np.array_equal(frames.values, _hook_values(torus, frames))

    # And it is a real boundary: a hair later the event has joined the sum.
    later = intensity_frames(torus, [at_event + 1e-9], resolution=RESOLUTION)
    assert later.values.max() > frames.values.max()


def test_the_floor_is_applied_after_summing(frame_times):
    """An inhibitory kernel must floor the total, not each term.

    `_full_intensity` takes `max(0, base + sum)`. Flooring the terms instead
    would leave a positive intensity where the model says there is none, and the
    picture would show excitation that the simulator would never have thinned
    against.
    """
    process = seeded_process(Torus2D(WIDTH, HEIGHT), base=lambda x: 0.05, spatial=lambda d: -1.0)
    frames = intensity_frames(process, frame_times, resolution=RESOLUTION)
    assert (frames.values == 0.0).any()
    assert np.array_equal(frames.values, _hook_values(process, frames))


def test_a_process_with_no_events_is_its_background(frame_times):
    process = seeded_process(Torus2D(WIDTH, HEIGHT), n_events=0)
    frames = intensity_frames(process, frame_times, resolution=RESOLUTION)
    assert np.allclose(frames.values, 0.3)


# ---------------------------------------------------------------------------
# The gluings show up in the data, not only in the geometry
# ---------------------------------------------------------------------------


def test_the_field_carries_the_klein_flip_across_the_seam(bottle, frame_times):
    """Crossing the glued edge comes back mirrored, in the values themselves.

    The last column of the grid is the wrapped image of the first, reversed --
    which is the glide reflection. If the fold were skipped or the axes
    transposed, the two would merely be equal and the surface would be a torus.
    """
    frames = intensity_frames(bottle, frame_times, resolution=(11, 11))
    assert np.allclose(frames.values[:, :, -1], frames.values[:, ::-1, 0], atol=1e-12, rtol=0)


@pytest.mark.slow
def test_the_field_is_antipodally_symmetric_on_the_projective_plane(projective):
    """The deck group is the antipodal map, so the colouring must be antipodal.

    Checked through the chart rather than through grid indices: the antipode of
    ``(theta, phi)`` is ``(pi - theta, phi -/+ pi)``, which is not a reversal of
    the grid.
    """
    domain = projective.domain
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.uniform(0.0, np.pi, 200), rng.uniform(-np.pi, np.pi, 200)])
    antipodes = _antipodes(points)
    here = np.array([projective.intensity(3.0, domain.wrap(p)) for p in points])
    there = np.array([projective.intensity(3.0, domain.wrap(p)) for p in antipodes])
    assert np.abs(here - there).max() < 1e-12


@pytest.mark.slow
def test_skipping_the_fold_would_tear_the_projective_plane():
    """The fold is load-bearing, not defensive.

    `FundamentalDomain.distance` reduces both of its arguments into the polygon
    on its own, so the excitation term is already group-invariant. `base` is
    not: it is handed the raw chart point. With a chart-dependent background the
    unfolded field is discontinuous across the equator by 0.8 on an intensity of
    order 1 -- which would read as a feature of the process.
    """
    process = seeded_process(
        FundamentalDomain.projective_plane(),
        n_events=4,
        base=lambda x: 0.5 + 0.4 * np.cos(x[0]),
    )
    domain = process.domain
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.uniform(0.0, np.pi, 60), rng.uniform(-np.pi, np.pi, 60)])
    antipodes = _antipodes(points)

    folded = np.abs(
        np.array([process.intensity(3.0, domain.wrap(p)) for p in points])
        - np.array([process.intensity(3.0, domain.wrap(p)) for p in antipodes])
    ).max()
    unfolded = np.abs(
        np.array([process.intensity(3.0, p) for p in points])
        - np.array([process.intensity(3.0, p) for p in antipodes])
    ).max()

    assert folded < 1e-12
    assert unfolded > 0.1


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_the_colour_range_is_global_across_every_frame(torus, frame_times):
    frames = intensity_frames(torus, frame_times, resolution=RESOLUTION)
    assert frames.vmin == 0.0
    assert frames.vmax == frames.values.max()

    hottest = int(np.argmax([frame.max() for frame in frames.values]))
    assert frames.normalised(hottest).max() == pytest.approx(1.0)
    # Every other frame is dimmer *on the same scale*, which is the whole point.
    coolest = int(np.argmin([frame.max() for frame in frames.values]))
    assert frames.normalised(coolest).max() < 1.0


@pytest.mark.parametrize("index", range(5))
def test_normalised_frames_stay_in_the_unit_interval(torus, frame_times, index):
    frames = intensity_frames(torus, frame_times, resolution=RESOLUTION, vmax=0.5)
    unit = frames.normalised(index)
    assert unit.min() >= 0.0
    assert unit.max() <= 1.0


def test_a_flat_field_normalises_to_the_middle_of_the_scale(frame_times):
    """Zero would read as the colour map's darkest value, implying a measured minimum."""
    process = seeded_process(Torus2D(WIDTH, HEIGHT), n_events=0)
    frames = intensity_frames(process, frame_times, resolution=RESOLUTION, vmax=0.0, vmin=0.0)
    assert np.all(frames.normalised(0) == 0.5)


def test_an_empty_colour_range_raises(torus, frame_times):
    with pytest.raises(ValueError, match="colour range is empty"):
        intensity_frames(torus, frame_times, resolution=RESOLUTION, vmin=1.0, vmax=0.5)


# ---------------------------------------------------------------------------
# The grid, and the runtime guard
# ---------------------------------------------------------------------------


def test_the_grid_includes_both_endpoints_of_both_ranges(sphere):
    """What closes the seam: the last column names the same points as the first."""
    surface = embed(sphere.domain)
    frames = intensity_frames(sphere, [1.0], resolution=(7, 7), embedding=surface)
    grid = frames.chart
    assert grid.shape == (7, 7, 2)
    west = surface.ambient(grid[:, 0, :])
    east = surface.ambient(grid[:, -1, :])
    assert np.abs(west - east).max() < 1e-14


def test_resolution_must_be_at_least_two(torus):
    with pytest.raises(ValueError, match="at least 2 along each axis"):
        intensity_frames(torus, [1.0], resolution=(1, 8))


def test_the_self_check_catches_a_kernel_that_is_not_a_function(frame_times):
    """The hoist is valid only for pure callables, and nothing can check that statically.

    A background that changes between calls makes the cached value and the hook
    disagree. That must raise, not render: a picture drawn from a stale cache is
    wrong everywhere and looks fine.
    """
    counter = {"n": 0}

    def drifting_base(x):
        counter["n"] += 1
        return 0.3 + 1e-3 * counter["n"]

    process = seeded_process(Torus2D(WIDTH, HEIGHT), base=drifting_base)
    with pytest.raises(RuntimeError, match=r"disagrees with process\.intensity"):
        intensity_frames(process, frame_times, resolution=RESOLUTION)


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------


def test_event_markers_are_inclusive_in_time(torus):
    """`t_i <= t`, unlike the strict inequality the intensity uses.

    The left-limit convention is about a *rate*. A marker answers whether the
    event has happened, and one that appeared a frame late would read as lag in
    the simulation.
    """
    at_event = float(torus.events[0, 2])
    points, _ = event_opacities(torus, at_event)
    assert len(points) == 3

    before, _ = event_opacities(torus, at_event - 1e-9)
    assert len(before) == 2


def test_event_markers_fade_at_the_kernels_own_rate(torus):
    t = float(torus.events[0, -1]) + 0.5
    points, opacity = event_opacities(torus, t)
    assert len(points) == torus.n_simulated

    peak = torus.temporal(0.0)
    expected = np.clip([torus.temporal(t - s) / peak for s in torus.events[0, :]], 0.15, 1.0)
    assert np.allclose(opacity, expected)
    # Monotone kernel, events in time order: the oldest is the faintest.
    assert opacity[0] <= opacity[-1]


def test_a_spent_event_stays_visible(torus):
    """A floor rather than a fade to nothing: the event happened, and it is history."""
    _, opacity = event_opacities(torus, 1e6)
    assert np.all(opacity == pytest.approx(0.15))


def test_no_events_have_happened_before_the_first_one(torus):
    points, opacity = event_opacities(torus, float(torus.events[0, 0]) - 1.0)
    assert len(points) == 0
    assert len(opacity) == 0
