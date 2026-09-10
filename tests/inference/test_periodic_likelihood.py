"""Fitting a background that varies in time.

The arithmetic that matters is a factorisation. The intensity is
``m(x) s(t) + excitation``, so its compensator over a block is

    (spatial integral of m) x (closed-form integral of s) + quadrature(excitation),

which is why `PeriodicBase.at` returns the **shape alone**: the spatial integral
the cached backend already computes stays exactly what it was, and the schedule
contributes a number rather than an axis.

Taking the background's time integral in closed form rather than through the
panels is not a micro-optimisation, and the last tests measure why. Panels are
placed at the events, so a cycle shorter than the gaps between them oscillates
*inside* one, where a fixed-order rule cannot follow it. Against the exact
integral, on this history's panels (median width 0.58):

=========  ==============  ===============
period     order 8         order 16
=========  ==============  ===============
4.00       -0.00%          -0.00%
0.50       +0.02%          +0.00%
0.35       -0.55%          -0.00%
0.20       **+10.72%**     -0.00%
0.10       **-5.69%**      **+5.82%**
=========  ==============  ===============

Two things in that table matter more than the headline. The error goes **both
ways** -- a background integrated too large is a penalty over-applied, so the
excitation comes back too *small*, which is the mirror of this package's usual
worry and no better. And below about a fifth of the panel width **raising the
order stops helping**: at period 0.10 order 16 is as wrong as order 8, because
the rule is aliasing a cycle it cannot see rather than approximating one it can.
The closed form is exact at every row.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstantBase,
    ExponentialKernel,
    GaussianSpatial,
    History,
    PeriodicBase,
    SpatioTemporalLogLikelihood,
    spatio_temporal_model,
)
from hawkes_package.inference import _compensator as compensator

TIMES = np.array([0.31, 0.47, 1.05, 1.09, 2.30, 2.88, 3.02, 4.51, 5.60, 5.93, 7.15, 8.02, 9.44])


@pytest.fixture
def history():
    points = np.random.default_rng(4).uniform(-np.pi, np.pi, size=(1, TIMES.size))
    return History(TIMES, points, 0.0, end=10.0)


def periodic_model(n_harmonics=1, period=4.0):
    return spatio_temporal_model(
        hp.Circle(),
        base=PeriodicBase(ConstantBase(), n_harmonics, period),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=64,
    )


def constant_model():
    return spatio_temporal_model(
        hp.Circle(),
        base=ConstantBase(),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=64,
    )


def test_the_coordinates_extend_the_constant_background():
    model = periodic_model()
    assert model.spec.names == ("mu", "cos_1", "sin_1", "alpha", "beta", "sigma")
    assert [p.kind for p in model.spec.parameters[1:3]] == ["real", "real"]


def test_more_harmonics_add_two_coordinates_each():
    assert periodic_model(n_harmonics=3).spec.names[:7] == (
        "mu",
        "cos_1",
        "sin_1",
        "cos_2",
        "sin_2",
        "cos_3",
        "sin_3",
    )
    assert periodic_model(n_harmonics=0).spec.names == ("mu", "alpha", "beta", "sigma")


def test_zero_amplitude_is_the_constant_background_exactly(history):
    """The assertion that says the new term did not disturb the old path.

    Bit for bit, not to a tolerance: at zero coefficients the schedule is 1.0 at
    every time and its integral is the interval length, so every float in the
    log-likelihood must be the one the constant background produced.
    """
    flat = SpatioTemporalLogLikelihood(constant_model())
    cycle = SpatioTemporalLogLikelihood(periodic_model())
    tail = np.array([0.6, 1.5, 0.6])

    assert cycle.total(np.concatenate([[0.5, 0.0, 0.0], tail]), history) == flat.total(
        np.concatenate([[0.5], tail]), history
    )


def test_both_backends_agree_with_a_cycle(history):
    """The cached path factorises the background; the hooks path does not.

    They are two arrangements of one integral, so agreement is a real check on
    the factorisation rather than a restatement of it.
    """
    model = periodic_model()
    theta = np.array([0.5, 0.5, 0.2, 0.6, 1.5, 0.6])
    cached = SpatioTemporalLogLikelihood(model, backend="cached").total(theta, history)
    hooks = SpatioTemporalLogLikelihood(model, backend="hooks").total(theta, history)
    assert cached == pytest.approx(hooks, rel=1e-8)


def test_the_cycle_changes_the_answer(history):
    """A guard on the guard above: the terms must actually be reaching the sum."""
    model = periodic_model()
    likelihood = SpatioTemporalLogLikelihood(model)
    flat = likelihood.total(np.array([0.5, 0.0, 0.0, 0.6, 1.5, 0.6]), history)
    strong = likelihood.total(np.array([0.5, 0.7, 0.0, 0.6, 1.5, 0.6]), history)
    assert abs(strong - flat) > 0.1


def test_the_schedule_is_reachable_from_the_family():
    family = PeriodicBase(ConstantBase(), 1, 24.0)
    schedule = family.schedule(np.array([0.5, 0.4, 0.2]))
    assert schedule.period == 24.0
    np.testing.assert_allclose(schedule.cosine, [0.4])
    np.testing.assert_allclose(schedule.sine, [0.2])


def test_the_family_builds_the_simulator_s_callable():
    """One object, two consumers: the likelihood reads the halves, the simulator calls it."""
    family = PeriodicBase(ConstantBase(), 1, 24.0)
    theta = np.array([0.5, 0.4, 0.0])
    background = family.build(theta)
    assert background.time_varying is True
    assert float(background(0.0, np.array([0.1]))) == pytest.approx(0.5 * 1.4)
    assert float(background.supremum(np.array([0.1]))) == pytest.approx(0.5 * 1.4, rel=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_harmonics": -1}, "non-negative whole number"),
        ({"n_harmonics": 1.5}, "non-negative whole number"),
        ({"period": 0.0}, "period must be positive"),
    ],
)
def test_an_impossible_family_is_refused(kwargs, match):
    arguments = {"shape": ConstantBase(), "n_harmonics": 1, "period": 24.0}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        PeriodicBase(**arguments)


def test_the_closed_form_beats_the_panels_on_a_fast_cycle(history):
    """Why the background's time integral is not left to the quadrature.

    At a period of 0.20 against a median panel width of 0.58, the panelled
    integral is **10.7% above** the exact one -- and the sign is worth reading. A
    background integrated too large is a penalty over-applied, so the excitation
    comes back too small: the mirror of this package's usual worry, and no more
    acceptable.
    """
    family = PeriodicBase(ConstantBase(), 1, 0.20)
    theta = np.array([0.5, 0.8, 0.0])
    schedule = family.schedule(theta)

    exact = family.time_integral(theta, 0.0, 10.0)
    edges = compensator.breakpoints(0.0, 10.0, TIMES, ())
    nodes, weights = compensator.panels(edges, compensator.DEFAULT_ORDER)
    panelled = float(np.dot(weights, schedule(nodes)))

    # The closed form against a grid fine enough to resolve the cycle. Not
    # against the span: 10 is not a whole number of periods here, so the
    # schedule's mean of one does not make the integral 10.
    grid = np.linspace(0.0, 10.0, 400_001)
    values = schedule(grid)
    dense = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))
    assert exact == pytest.approx(dense, rel=1e-6)
    assert abs(panelled - exact) / exact > 0.05, (
        f"the panels were expected to miss the cycle, but they came within "
        f"{abs(panelled - exact) / exact:.1%}"
    )


def test_a_cycle_slower_than_a_panel_is_integrated_by_either_route():
    """The negative control, and the boundary between the two regimes.

    At a period of 4 -- seven times the median panel width -- the panels agree
    with the closed form to five decimal places. So nothing here claims the
    quadrature is generally inadequate: it is inadequate for a cycle it cannot
    see inside a panel, which is a statement about a ratio and not about a rule.
    """
    family = PeriodicBase(ConstantBase(), 1, 4.0)
    theta = np.array([0.5, 0.8, 0.0])
    schedule = family.schedule(theta)
    edges = compensator.breakpoints(0.0, 10.0, TIMES, ())
    nodes, weights = compensator.panels(edges, compensator.DEFAULT_ORDER)

    exact = family.time_integral(theta, 0.0, 10.0)
    panelled = float(np.dot(weights, schedule(nodes)))
    assert panelled == pytest.approx(exact, rel=1e-5)
