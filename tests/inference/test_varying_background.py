"""A background that varies over the domain.

`ConstantBase` says events are equally likely everywhere, which is false for
every applied dataset anyone has, and the failure it causes is the one this
package exists to prevent: **a constant-background fit to data whose background
is clustered attributes the clustering to self-excitation** and reports a
confident, wrong excitation. Nothing raises, and the fit looks converged.

Two things are asserted here that are cheaper to state than to trust:

* a log-linear background with **no covariates at all** reproduces
  `ConstantBase` bit for bit, which is the whole plumbing in one equality; and
* it is positive everywhere by construction, so it cannot trip the cached
  backend's non-negativity precondition -- the identity that backend relies on
  holds only where the pre-floor integrand is non-negative at every node, and it
  raises rather than degrading.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstantBase,
    ExponentialKernel,
    GaussianSpatial,
    History,
    LogLinearBase,
    SpatioTemporalLogLikelihood,
    spatio_temporal_model,
)


def east(points):
    """The first coordinate, vectorized over an ``(m, ndim)`` array."""
    return np.asarray(points, dtype=float)[:, 0]


def model_with(base):
    """`st_model_1d` with the background swapped, everything else fixed."""
    return spatio_temporal_model(
        hp.Circle(),
        base=base,
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=32,
    )


def test_the_coefficients_live_on_the_whole_real_line():
    """The first parameters here that are not positive rates.

    A coefficient's sign is the direction of the effect and `log_mu0` is a log,
    so `ParameterSpec` must transform both with the identity rather than with
    `exp` -- which it decides from the bounds alone.
    """
    base = LogLinearBase((east,), names=("east",))
    assert base.spec.names == ("log_mu0", "b_east")
    assert [p.kind for p in base.spec.parameters] == ["real", "real"]


def test_positional_names_when_none_are_given():
    base = LogLinearBase((east, east))
    assert base.spec.names == ("log_mu0", "b_0", "b_1")


def test_a_name_per_covariate_or_none_at_all():
    with pytest.raises(ValueError, match="2 names for 1 covariates"):
        LogLinearBase((east,), names=("east", "north"))


def test_no_covariates_reproduces_the_constant_background_exactly(spatial_history):
    """One equality for the whole plumbing.

    At ``log_mu0 = 0`` the value is ``exp(0) = 1.0`` exactly, so this is an
    honest bit-for-bit comparison against `ConstantBase(mu=1.0)` rather than a
    tolerance hiding a reparameterisation. Every node, every event, and the
    whole log-likelihood.
    """
    nodes = np.linspace(-np.pi, np.pi, 41).reshape(-1, 1)
    flat = LogLinearBase()
    np.testing.assert_array_equal(
        flat.at(np.array([0.0]), nodes), ConstantBase().at(np.array([1.0]), nodes)
    )

    varying = SpatioTemporalLogLikelihood(model_with(flat))
    constant = SpatioTemporalLogLikelihood(model_with(ConstantBase()))
    theta = np.array([0.6, 1.5, 0.6])
    assert varying.total(np.concatenate([[0.0], theta]), spatial_history) == constant.total(
        np.concatenate([[1.0], theta]), spatial_history
    )


def test_a_zero_coefficient_is_the_same_background_as_no_covariate(spatial_history):
    """The covariate is evaluated and then multiplied by zero, not skipped."""
    with_covariate = SpatioTemporalLogLikelihood(model_with(LogLinearBase((east,))))
    without = SpatioTemporalLogLikelihood(model_with(LogLinearBase()))
    assert with_covariate.total(
        np.array([0.0, 0.0, 0.6, 1.5, 0.6]), spatial_history
    ) == pytest.approx(without.total(np.array([0.0, 0.6, 1.5, 0.6]), spatial_history), rel=1e-14)


def test_the_background_integral_is_the_one_the_quadrature_reports():
    """`mu` is per unit measure, so the rate is its integral over the domain.

    For ``exp(b0 + b x)`` on ``[-pi, pi]`` that integral is available in closed
    form, which is what makes this a check on the quadrature rather than a
    restatement of it.
    """
    process = hp.SpatioTemporalHawkesProcess(
        base=LogLinearBase((east,)).build(np.array([-0.5, 0.4])),
        spatial=lambda d: np.zeros_like(np.asarray(d, dtype=float)),
        temporal=lambda s: np.zeros_like(np.asarray(s, dtype=float)),
        domain=hp.Circle(),
        monotone_temporal_kernel=True,
        n_quad=64,
        rng=0,
    )
    b0, b = -0.5, 0.4
    closed_form = math.exp(b0) * (math.exp(b * np.pi) - math.exp(-b * np.pi)) / b
    assert process._integrated_intensity(0.0) == pytest.approx(closed_form, rel=1e-9)


def test_a_background_too_peaked_for_the_rule_says_so():
    """The kernel is not the only thing the quadrature has to resolve.

    A background lump narrower than a panel loses its mass *between* nodes, in
    exactly the places the events are, and the only symptom is a simulated event
    rate wrong by that fraction -- the same silent failure
    `check_resolution` already catches for the spatial kernel, arriving from the
    other term of the intensity.
    """
    flat = lambda values: np.zeros_like(np.asarray(values, dtype=float))
    narrow = LogLinearBase((lambda points: -((np.asarray(points)[:, 0] / 0.01) ** 2) / 2,))

    with pytest.warns(UserWarning, match="the background is too narrow"):
        hp.SpatioTemporalHawkesProcess(
            base=narrow.build(np.array([0.0, 1.0])),
            spatial=flat,
            temporal=flat,
            domain=hp.Circle(),
            monotone_temporal_kernel=True,
            n_quad=16,
            rng=0,
        )


def test_a_constant_background_is_not_checked_for_resolution():
    """It is resolved exactly, and this constructor runs once per particle per move.

    Checking it would double the quadrature work of every fit that predates a
    varying background, to prove a rule integrates a constant.
    """
    flat = lambda values: np.zeros_like(np.asarray(values, dtype=float))
    hp.SpatioTemporalHawkesProcess(
        base=lambda x: 0.5,
        spatial=flat,
        temporal=flat,
        domain=hp.Circle(),
        monotone_temporal_kernel=True,
        n_quad=16,
        rng=0,
    )


def test_the_cached_backend_stays_usable(spatial_history):
    """The log link is what keeps it usable, and that is the reason for the link.

    An affine background with a coefficient of the wrong sign goes negative at
    some node and the cached backend refuses -- mid-fit, on a particle that
    looked fine a move ago. This one cannot go negative at all, so a strong
    covariate effect never costs the fast path.
    """
    likelihood = SpatioTemporalLogLikelihood(model_with(LogLinearBase((east,))), backend="auto")
    value = likelihood.total(np.array([0.0, 3.0, 0.6, 1.5, 0.6]), spatial_history)
    assert math.isfinite(value)
    assert likelihood.backend_used == "cached"


def test_an_enormous_exponent_is_finite_rather_than_a_warning():
    """`filterwarnings = ["error"]`, and `exp(800)` warns before it returns `inf`.

    A rejuvenation move can propose a coordinate that far out. The particle is
    rejected either way -- a background of 1e307 per unit measure gives a
    compensator no finite log-likelihood survives -- so the choice is between
    rejecting it and raising in the middle of a healthy fit.
    """
    values = LogLinearBase().at(np.array([5000.0]), np.zeros((3, 1)))
    assert np.all(np.isfinite(values))
    assert np.all(values > 1e300)


@pytest.mark.parametrize(
    ("covariate", "match"),
    [
        (lambda points: np.zeros(2), "returned 2 values for 5 positions"),
        (lambda points: np.full(5, np.nan), "non-finite"),
        (lambda points: np.full(5, np.inf), "non-finite"),
    ],
)
def test_a_covariate_that_cannot_be_used_is_refused_where_it_appears(covariate, match):
    """A `nan` covariate reads to the sampler as an invalid particle.

    It would be resampled away without anything saying the covariate was the
    reason, so the refusal names the covariate instead.
    """
    base = LogLinearBase((covariate,), names=("broken",))
    with pytest.raises(ValueError, match=match):
        base.at(np.array([0.0, 1.0]), np.zeros((5, 1)))


def test_build_and_at_are_the_same_function():
    """The simulator uses `build`, the cached likelihood uses `at`.

    Two spellings of one background is how an accessor drifts from its hook, so
    they are checked against each other on a grid rather than at one point.
    """
    base = LogLinearBase((east,), names=("east",))
    theta = np.array([-0.3, 0.7])
    callable_form = base.build(theta)
    grid = np.linspace(-np.pi, np.pi, 33).reshape(-1, 1)
    np.testing.assert_allclose(
        [callable_form(x) for x in grid], base.at(theta, grid), rtol=0.0, atol=0.0
    )


def _background_history(seed, truth, n_events=60, horizon=20.0):
    """Locations drawn from ``exp(truth * x)`` on the circle, times uniform.

    Placed by hand rather than simulated. The question is whether the
    coefficient is estimable from the likelihood at all, and simulating 60
    events on a `Circle` takes 65 seconds and would put the simulator in the
    same assertion.
    """
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < n_events:
        x = rng.uniform(-np.pi, np.pi)
        if rng.uniform() < math.exp(truth * (x - np.pi)):
            draws.append(x)
    times = np.sort(rng.uniform(0.0, horizon, size=n_events))
    return History(times, np.asarray(draws).reshape(1, -1), 0.0, end=horizon)


def _profile(history, grid, *, concentrate):
    """Return where the likelihood peaks over `grid`, in ``b_east``.

    With ``concentrate`` the intercept is set for each ``b`` to the value that
    makes the model's expected count equal the observed one --
    ``exp(log_mu0) * T * int exp(b x) dx = n``, in closed form on ``[-pi, pi]``.
    That is the profile likelihood in the intercept, for a background whose
    excitation is small enough that it carries the count.
    """
    likelihood = SpatioTemporalLogLikelihood(model_with(LogLinearBase((east,))))
    n, horizon = history.n_events, history.end
    values = []
    for b in grid:
        integral = 2.0 * np.pi if abs(b) < 1e-12 else 2.0 * math.sinh(b * np.pi) / b
        intercept = math.log(n / (horizon * integral)) if concentrate else -1.0
        values.append(likelihood.total(np.array([intercept, b, 0.05, 2.0, 0.6]), history))
    return float(grid[int(np.argmax(values))])


def gaussian_log_density(reference, bandwidth):
    """A kernel-density covariate: ``log`` of a Gaussian KDE over fixed points.

    Written here rather than shipped as a `KernelDensityBase`, and the reason is
    arithmetic. A background ``exp(b0 + b * log f(x))`` is ``exp(b0) * f(x)**b``,
    so at ``b = 1`` it *is* the density up to a constant, and the constant is
    what a fitted ``b0`` absorbs -- including the normaliser that would make the
    density integrate to one over the domain. A separate family would be that
    with one parameter fewer and a normalising integral of its own to keep in
    step with the process's quadrature rule.
    """
    points = np.asarray(reference, dtype=float)

    def covariate(query):
        gaps = np.asarray(query, dtype=float)[:, None, :] - points[None, :, :]
        squared = np.sum(gaps**2, axis=2)
        return np.log(np.mean(np.exp(-squared / (2.0 * bandwidth**2)), axis=1))

    return covariate


def test_a_density_covariate_is_the_kernel_density_background():
    """`exp(b0 + 1 * log f) == exp(b0) * f`, which is why there is no KDE family.

    Asserted as a *proportionality* on a grid, since the two differ by exactly
    the constant the intercept is there to carry.
    """
    reference = np.array([[-2.0], [0.1], [0.4], [1.7]])
    covariate = gaussian_log_density(reference, 0.5)
    grid = np.linspace(-np.pi, np.pi, 65).reshape(-1, 1)

    base = LogLinearBase((covariate,), names=("density",))
    values = base.at(np.array([-0.7, 1.0]), grid)
    density = np.exp(covariate(grid))
    ratio = values / density
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    assert ratio[0] == pytest.approx(math.exp(-0.7), rel=1e-12)


@pytest.mark.statistical
def test_the_coefficient_is_identified_by_the_likelihood():
    """A profile over `b_east` on data whose background genuinely varies.

    Over seeds 0-20 the peak has mean 0.933 against a truth of 0.9 and a worst
    error of 0.275, so the threshold below is the measured spread and not a
    guess.
    """
    truth = 0.9
    peak = _profile(_background_history(0, truth), np.linspace(-1.0, 3.0, 161), concentrate=True)
    assert peak == pytest.approx(truth, abs=0.35), f"profile peaked at {peak}, truth {truth}"


@pytest.mark.statistical
def test_holding_the_intercept_fixed_biases_the_coefficient_low():
    """The ridge the class docstring warns about, measured rather than described.

    `log_mu0` and a coefficient trade off through the total event count: a
    larger `b` predicts more events, so an intercept pinned too high is paid for
    by a coefficient too small. Over seeds 0-20 the same profile at a fixed
    ``log_mu0 = -1`` peaks in **[0.53, 0.62]** -- mean 0.583 against a truth of
    0.9, and never once near it -- while concentrating the intercept out
    recovers 0.933. That is a systematic 0.32, not sampling noise, and on real
    data it would read as a weaker spatial trend rather than as a bad fit.
    """
    grid = np.linspace(-1.0, 3.0, 161)
    history = _background_history(0, 0.9)
    assert _profile(history, grid, concentrate=False) < 0.7
    assert _profile(history, grid, concentrate=True) > 0.7


CENTRES = np.array([-1.6, 0.9])
LUMP_WIDTH = 0.35


def two_lump_density(x):
    """A background concentrated in two places, on the circle's own distance."""
    values = np.asarray(x, dtype=float)
    gaps = np.abs(values[..., None] - CENTRES)
    gaps = np.minimum(gaps, 2 * np.pi - gaps)
    return np.mean(np.exp(-(gaps**2) / (2 * LUMP_WIDTH**2)), axis=-1)


def clustered_history(seed, n_events=80, horizon=20.0):
    """Locations from the two lumps, times uniform: **no self-excitation at all**."""
    rng = np.random.default_rng(seed)
    top = float(two_lump_density(np.linspace(-np.pi, np.pi, 4001)).max())
    draws = []
    while len(draws) < n_events:
        x = rng.uniform(-np.pi, np.pi)
        if rng.uniform() * top < two_lump_density(x):
            draws.append(x)
    times = np.sort(rng.uniform(0.0, horizon, size=n_events))
    return History(times, np.asarray(draws).reshape(1, -1), 0.0, end=horizon)


def _excitation_profile(history, base, head):
    """Where the likelihood peaks in `alpha`, the excitation amplitude."""
    likelihood = SpatioTemporalLogLikelihood(model_with(base))
    grid = np.linspace(0.0, 3.0, 61)[1:]  # alpha is positive
    values = [likelihood.total(np.array([*head, alpha, 2.0, 0.5]), history) for alpha in grid]
    return float(grid[int(np.argmax(values))])


@pytest.mark.statistical
def test_a_constant_background_blames_the_excitation():
    """The measurement this whole package exists for.

    The data has **no self-excitation whatsoever** -- locations drawn from two
    fixed lumps, times uniform on the window. A constant background cannot
    express the lumps, so the only thing left to explain events landing near
    each other is excitation, and the likelihood duly finds some.

    Over seeds 0-20 the constant background peaks at ``alpha`` **0.614** on
    average (min 0.450, max 0.700) -- a branching ratio near 0.31 invented out
    of nothing -- while the same data under a background carrying the density as
    a covariate peaks at **0.062** (max 0.150). The constant one is higher on
    **21 of 21 seeds**. Nothing raises in either case; both fits look converged.
    """
    history = clustered_history(0)
    grid = np.linspace(-np.pi, np.pi, 4001)
    mass = float(np.mean(two_lump_density(grid)) * 2 * np.pi)
    flat_rate = history.n_events / (history.end * 2 * np.pi)
    intercept = math.log(history.n_events / (history.end * mass))

    def log_lumps(points):
        return np.log(two_lump_density(np.asarray(points, dtype=float)[:, 0]))

    constant = _excitation_profile(history, ConstantBase(), [flat_rate])
    varying = _excitation_profile(history, LogLinearBase((log_lumps,)), [intercept, 1.0])

    assert varying < 0.2, f"a background that fits the lumps still found alpha={varying}"
    assert constant > 3 * varying, (
        f"a constant background should attribute the lumps to excitation, but "
        f"alpha peaked at {constant} against {varying}"
    )


@pytest.mark.statistical
def test_a_flat_background_leaves_the_coefficient_at_zero():
    """The negative control: no trend in the data, no trend in the profile.

    Without it the test above says only that the profile moves, not that it
    moves for the right reason. Over seeds 0-20 the peak lies in [-0.07, 0.12].
    """
    peak = _profile(_background_history(0, 0.0), np.linspace(-1.0, 3.0, 161), concentrate=True)
    assert abs(peak) < 0.2, f"a flat background profiled to {peak}"
