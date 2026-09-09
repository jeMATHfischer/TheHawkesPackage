"""Marks that scale an event's productivity, in the ETAS shape.

Two things here are new to the package and both are worth the tests they get.

An **unbounded productivity** looks like a threat to the thinning bound and is
not: the intensity sums over `t_i < t` strictly and the bound over `t_i <= t_0`,
so both range over marks that have already been drawn. The supremum of `g` over
the mark *distribution* never enters.

A **divergent expected productivity** is the opposite case — it looks harmless
and is not. `E[g(m)] = b/(b-a)` is infinite for `a >= b`, while every realised
mark stays finite and every simulated catalogue looks ordinary. Nothing at run
time notices, so the constructor does.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.marked import expected_productivity


def exponential_marks(rate):
    return lambda rng: float(rng.exponential(1.0 / rate))


def test_the_expected_productivity_matches_its_closed_form():
    """``b / (b - a)`` below the boundary, infinite at or above it."""
    assert expected_productivity(0.0, 1.5) == pytest.approx(1.0)
    assert expected_productivity(1.0, 1.5) == pytest.approx(3.0)
    assert expected_productivity(0.5, 1.5) == pytest.approx(1.5)
    assert math.isinf(expected_productivity(1.5, 1.5))
    assert math.isinf(expected_productivity(2.0, 1.5))

    with pytest.raises(ValueError, match="mark rate b must be positive"):
        expected_productivity(0.5, 0.0)


@pytest.mark.statistical
def test_the_expected_productivity_is_what_marks_actually_produce():
    """The closed form against a Monte Carlo mean of ``g(m)``.

    Two independent statements about the same quantity: if they disagreed, the
    branching ratio would be computed from a number the simulator does not obey.
    """
    a, b = 0.5, 1.5
    rng = np.random.default_rng(0)
    marks = rng.exponential(1.0 / b, size=200_000)
    empirical = float(np.mean(np.exp(a * marks)))
    assert empirical == pytest.approx(expected_productivity(a, b), rel=0.02)


def test_the_branching_ratio_multiplies_the_two_factors():
    process = hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=1.5)
    assert process.branching_ratio == pytest.approx(0.3 / 2.0 * 1.5)


def test_a_divergent_expected_productivity_is_refused():
    """The genuinely new failure in this package's stability story.

    Every realisation is finite; the expectation is not. A run at these
    parameters produces an ordinary-looking catalogue, so the refusal has to
    happen at construction or not at all.
    """
    with pytest.raises(ValueError, match="expected productivity is infinite"):
        hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.01, beta=2.0, scale=1.5, b_value=1.5)
    with pytest.raises(ValueError, match="expected productivity is infinite"):
        hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.01, beta=2.0, scale=2.0, b_value=1.5)


def test_a_merely_supercritical_ratio_is_refused_too():
    """Finite expected productivity, branching ratio still above one."""
    with pytest.raises(ValueError, match=r"branching ratio is 1\.5"):
        hp.ExponentialMarkedHawkes(mu=1.0, alpha=1.0, beta=1.0, scale=0.5, b_value=1.5)


def test_a_process_just_inside_the_boundary_simulates():
    process = hp.ExponentialMarkedHawkes(
        mu=0.5, alpha=0.1979, beta=2.0, scale=0.9, b_value=1.0, rng=0
    )
    assert 0.98 < process.branching_ratio < 1.0
    process.simulate(200)
    assert process.events.shape == (2, 200)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": 0.0}, "beta must be positive"),
        ({"alpha": -0.1}, "alpha must be non-negative"),
        ({"b_value": 0.0}, "mark rate b must be positive"),
        ({"mu": -1.0}, "mu must be finite and non-negative"),
        ({"m0": np.inf}, "m0 must be finite"),
    ],
)
def test_construction_is_validated(kwargs, match):
    defaults = {"mu": 1.0, "alpha": 0.3, "beta": 2.0, "scale": 0.5, "b_value": 1.5}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        hp.ExponentialMarkedHawkes(**defaults)


def test_the_record_carries_times_and_marks():
    process = hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=1.5, rng=1)
    process.simulate(50)

    assert process.events.shape == (2, 50)
    assert process.n_simulated == 50
    assert np.all(np.diff(process.events[0]) > 0)
    np.testing.assert_array_equal(process.marks, process.events[1])
    assert np.all(process.marks >= 0.0), "m0 defaults to zero"


@pytest.mark.statistical
def test_the_marks_follow_the_law_they_were_drawn_from():
    """Otherwise the fitted b-value would describe a distribution nobody sampled."""
    from scipy import stats

    b, m0 = 1.5, 2.0
    process = hp.ExponentialMarkedHawkes(
        mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=b, m0=m0, rng=3
    )
    process.simulate(3000)

    assert np.all(process.marks >= m0)
    result = stats.kstest(process.marks - m0, stats.expon(scale=1.0 / b).cdf)
    assert result.pvalue > 1e-3, f"marks are not Exp({b}) above m0 (p={result.pvalue:.4g})"


def test_a_flat_productivity_gives_the_unmarked_intensity():
    """At ``scale = 0`` every event is equally productive, so the hooks coincide.

    Pointwise, to one ulp -- and *not* bit-identically, because the two factor
    the same sum differently: this class computes ``sum(g(m) * kappa)`` where
    `ExponentialHawkes` computes ``alpha * sum(exp(...))``, and distributivity is
    not exact in floating point.
    """
    mu, alpha, beta = 1.0, 0.3, 2.0
    marked = hp.ExponentialMarkedHawkes(
        mu=mu, alpha=alpha, beta=beta, scale=0.0, b_value=1.5, rng=0
    )
    marked.simulate(60)

    plain = hp.ExponentialHawkes(np.array([mu, alpha, beta]), rng=0)
    plain.events = marked.events[0]

    last = float(marked.events[0, -1])
    for t in np.linspace(0.0, last + 1.0, 300):
        assert marked._conditional_intensity(float(t)) == pytest.approx(
            plain._conditional_intensity(float(t)), abs=1e-15
        )
    for t in np.linspace(last, last + 1.0, 200):
        assert marked._upper_bound(float(t)) == pytest.approx(
            plain._upper_bound(float(t)), abs=1e-15
        )


def test_the_mark_is_drawn_only_after_acceptance():
    """Which is why the first event time is identical to the unmarked process.

    A mark drawn before the acceptance test would consume a variate on every
    rejected candidate too, and the two streams would part company immediately.
    They part at the *first acceptance* instead -- a mark is real extra
    randomness, so a marked run cannot be bit-identical to an unmarked one, and
    this is the most that can be asked.
    """
    mu, alpha, beta = 1.0, 0.3, 2.0
    marked = hp.ExponentialMarkedHawkes(
        mu=mu, alpha=alpha, beta=beta, scale=0.0, b_value=1.5, rng=0
    )
    marked.simulate(5)
    plain = hp.ExponentialHawkes(np.array([mu, alpha, beta]), rng=0)
    plain.simulate(5)

    assert marked.events[0, 0] == plain.events[0]
    assert not np.array_equal(marked.events[0], plain.events)


def test_a_negative_productivity_is_refused_when_it_appears():
    """`sup(g * kappa) = g * sup(kappa)` needs ``g >= 0``.

    Checked where the value is used rather than at construction, because a
    productivity is an arbitrary callable and its sign cannot be known until it
    is handed a mark.
    """
    process = hp.MarkedHawkes(
        mu=1.0,
        temporal=lambda s: 0.3 * np.exp(-2.0 * np.asarray(s, dtype=float)),
        productivity=lambda m: -np.ones_like(np.asarray(m, dtype=float)),
        mark_sampler=exponential_marks(1.0),
        rng=0,
    )
    with pytest.raises(RuntimeError, match="productivity returned"):
        process.simulate(20)


def test_simulate_until_reproduces_simulate():
    counted = hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=1.5, rng=5)
    counted.simulate(30)
    horizon = float(counted.events[0, -1])

    timed = hp.ExponentialMarkedHawkes(mu=1.0, alpha=0.3, beta=2.0, scale=0.5, b_value=1.5, rng=5)
    timed.simulate_until(horizon)

    np.testing.assert_array_equal(timed.events, counted.events)


@pytest.mark.statistical
def test_a_heavier_mark_tail_clusters_harder():
    """The property the whole package exists for, stated as an observable.

    Two processes with the *same* branching ratio but different mark tails. The
    heavy one concentrates the same expected offspring into fewer, larger
    events, so its counts should be more overdispersed.

    **Averaged over seeds, not asserted per seed**, and that is the measurement
    rather than caution: over twelve seeds at bin width 1.0 the mean
    variance-to-mean ratio is 1.47 light against 2.02 heavy, but the heavy case
    exceeds the light one on only **7 of 12 individual seeds**. The burstiness a
    heavy mark tail produces is itself heavy-tailed -- a run either happens to
    contain a very large mark or it does not -- so one realisation is a poor
    guide to the distribution it came from. A per-seed assertion here would fail
    two times in five on correct code.
    """
    horizon = 400.0
    bins = np.linspace(0.0, horizon, 400)  # width 1.0, twice the kernel scale

    def dispersion(scale, b_value, alpha, seed):
        process = hp.ExponentialMarkedHawkes(
            mu=1.0, alpha=alpha, beta=2.0, scale=scale, b_value=b_value, rng=seed
        )
        process.simulate_until(horizon)
        counts = np.histogram(process.events[0], bins=bins)[0]
        return float(np.var(counts) / np.mean(counts))

    # Both have branching ratio 0.3: 0.6/2 * 1 and 0.06/2 * 10.
    seeds = range(12)
    light = np.mean([dispersion(0.0, 1.0, 0.6, s) for s in seeds])
    heavy = np.mean([dispersion(0.9, 1.0, 0.06, s) for s in seeds])

    assert heavy > 1.15 * light, (
        f"a heavy mark tail should cluster harder at equal branching ratio, but "
        f"the mean dispersion was {heavy:.3f} against {light:.3f}"
    )
