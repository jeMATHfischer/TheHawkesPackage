"""Distributional correctness: does the simulator produce the right law?

The thinning invariant proves the algorithm is *self-consistent*; these tests
prove it produces the *intended distribution*. Seeds are fixed and thresholds
deliberately loose (KS p-value floor 1e-3, not 0.05) -- the question is "is this
broken", not "is this publication-grade". PCG64 is bit-reproducible across
platforms, so a seed that passes locally passes in CI.
"""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

import hawkes_package as hp
from hawkes_package.mcmc import mcmc_sampler

SEEDS = [1, 2, 7]


def compensator(times, event_times, mu, alpha, beta):
    """Integrated intensity Lambda(t) for the exponential-kernel Hawkes process.

    Lambda(t) = mu*t + (alpha/beta) * sum_{t_i < t} (1 - exp(-beta (t - t_i)))
    """
    out = []
    for t in np.atleast_1d(times):
        past = event_times[event_times < t]
        out.append(mu * t + (alpha / beta) * np.sum(1 - np.exp(-beta * (t - past))))
    return np.array(out)


# ---------------------------------------------------------------------------
# Exponential Hawkes
# ---------------------------------------------------------------------------


@pytest.mark.statistical
@pytest.mark.parametrize("seed", SEEDS)
def test_time_rescaling_gives_unit_exponential_gaps(seed):
    """The time-rescaling theorem is the sharpest test available here.

    If the events really come from this intensity, then transforming them by
    their own compensator yields a unit-rate Poisson process, whose gaps are
    Exp(1). This catches a wrong kernel, a wrong bound and a wrong baseline at
    once -- none of the structural tests would.
    """
    mu, alpha, beta = 1.0, 0.5, 2.0
    p = hp.ExponentialHawkes(np.array([mu, alpha, beta]), rng=seed)
    p.simulate(2000)

    rescaled = compensator(p.Events, p.Events, mu, alpha, beta)
    gaps = np.diff(rescaled)

    result = stats.kstest(gaps, "expon")
    assert result.pvalue > 1e-3, (
        f"seed={seed}: rescaled gaps are not unit-exponential (p={result.pvalue:.2e}); "
        "the simulated process does not match its own intensity"
    )


@pytest.mark.statistical
@pytest.mark.parametrize("seed", SEEDS)
def test_stationary_rate_matches_theory(seed):
    """The long-run rate of a stable Hawkes process is mu / (1 - alpha/beta)."""
    mu, alpha, beta = 1.0, 0.5, 2.0
    p = hp.ExponentialHawkes(np.array([mu, alpha, beta]), rng=seed)
    p.simulate(4000)

    burn = 500
    events = p.Events
    empirical = (len(events) - burn) / (events[-1] - events[burn])
    theoretical = mu / (1 - alpha / beta)
    assert empirical == pytest.approx(theoretical, rel=0.12)


@pytest.mark.statistical
@pytest.mark.parametrize("seed", SEEDS)
def test_vanishing_excitation_degenerates_to_poisson(seed):
    """With alpha -> 0 the process is Poisson(mu), so gaps are Exp(mu)."""
    mu = 2.0
    p = hp.ExponentialHawkes(np.array([mu, 1e-9, 1.0]), rng=seed)
    p.simulate(2000)

    gaps = np.diff(p.Events)
    result = stats.kstest(gaps, "expon", args=(0, 1 / mu))
    assert result.pvalue > 1e-3, f"seed={seed}: gaps are not Exp({mu}) (p={result.pvalue:.2e})"


@pytest.mark.statistical
def test_stronger_excitation_raises_the_rate():
    """A monotonicity check that no single-parameter fluke can satisfy."""
    mu, beta = 1.0, 2.0
    rates = []
    for alpha in (0.2, 0.8, 1.4):
        p = hp.ExponentialHawkes(np.array([mu, alpha, beta]), rng=5)
        p.simulate(2000)
        rates.append(len(p.Events) / p.Events[-1])
    assert rates[0] < rates[1] < rates[2]


# ---------------------------------------------------------------------------
# Monotone kernel
# ---------------------------------------------------------------------------


@pytest.mark.statistical
def test_monotone_matches_exponential_for_the_same_model():
    """MonotoneKernelHawkes with an identity-shifted nonlinearity is a linear
    Hawkes process, so its rate must match the closed-form value."""
    mu, alpha, beta = 2.0, 0.5, 2.0
    p = hp.MonotoneKernelHawkes(
        lambda dt: alpha * np.exp(-beta * np.asarray(dt, dtype=float)),
        nonlinearity=lambda x: x + mu,
        rng=13,
    )
    p.simulate(3000)
    burn = 500
    empirical = (len(p.Events) - burn) / (p.Events[-1] - p.Events[burn])
    assert empirical == pytest.approx(mu / (1 - alpha / beta), rel=0.12)


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


@pytest.mark.statistical
@pytest.mark.slow
def test_zero_excitation_is_poisson_in_time_and_uniform_in_space():
    """With no temporal excitation the process must collapse to a Poisson
    process of rate ``volume * mu`` whose locations are uniform on the domain.

    This is the only end-to-end check of the spatial sampler: an incorrect
    conditional density would show up as a non-uniform marginal.
    """
    mu = 0.5
    domain = hp.Circle()
    p = hp.SpatioTemporalHawkesProcess(
        base=lambda x: mu,
        spatial=lambda d: 0.0,
        temporal=lambda dt: 0.0,
        domain=domain,
        monotone_temporal_kernel=True,
        rng=3,
    )
    p.simulate(120)

    times, coords = p.Events[0, :], p.Events[1, :]

    gaps = np.diff(times)
    expected_rate = domain.volume * mu
    ks = stats.kstest(gaps, "expon", args=(0, 1 / expected_rate))
    assert ks.pvalue > 1e-3, f"inter-arrival times are not Exp({expected_rate}) (p={ks.pvalue:.2e})"

    lo, hi = domain.bounds[0]
    counts, _ = np.histogram(coords, bins=8, range=(lo, hi))
    chi2 = stats.chisquare(counts)
    assert chi2.pvalue > 1e-3, f"spatial marginal is not uniform (p={chi2.pvalue:.2e})"


@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_spatial_sampler_reproduces_the_conditional_density():
    """Locations must be drawn from the conditional spatial density itself.

    A history is injected rather than simulated, so the target density is known
    exactly and can be integrated bin by bin with quadrature. This is the
    sharpest available check on the spatial half of the algorithm: a
    mis-normalised or mis-centred density shows up immediately, where a
    clustering statistic would not -- with a narrow kernel on a wide domain the
    peak holds only a modest share of the mass, so nearest-neighbour distances
    barely move even when sampling is correct.
    """
    domain = hp.Circle()
    p = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 0.3,
        spatial=lambda d: 3.0 * np.exp(-8.0 * d),
        temporal=lambda dt: 2.0 * np.exp(-2.0 * float(dt)),
        domain=domain,
        monotone_temporal_kernel=True,
        rng=0,
    )
    # One real event at the origin.
    p.Events = np.array([[1.0], [0.0]])

    def density(x):
        return p._full_intensity(x, 1.5)

    samples = np.array(
        [
            float(np.ravel(domain.wrap(mcmc_sampler(density, domain.bounds, seed=s)))[0])
            for s in range(400)
        ]
    )

    lo, hi = domain.bounds[0]
    edges = np.linspace(lo, hi, 9)
    observed, _ = np.histogram(samples, bins=edges)
    expected = np.array(
        [quad(lambda x: density(np.array([x])), edges[i], edges[i + 1])[0] for i in range(8)]
    )
    expected = expected / expected.sum() * observed.sum()

    result = stats.chisquare(observed, expected)
    assert result.pvalue > 1e-3, (
        f"sampled locations do not follow the conditional density (p={result.pvalue:.2e});\n"
        f"observed={observed}\nexpected={expected.round(1)}"
    )
