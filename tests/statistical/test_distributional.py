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
from hawkes_package.spatio_temporal import _integration

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
def test_sampler_confined_to_a_fundamental_domain_is_uniform():
    """Rejecting proposals off the polygon must leave the marginal uniform *on* it.

    This is the check that licenses ``FundamentalDomain.periodic = False``. The
    domain is a proper subset of the box the chain proposes in, so a third of
    every proposal distribution falls outside; if rejecting those biased the
    result the bias would sit near the boundary, exactly where the six corners
    are. Compared against quadrature over the polygon rather than against a
    closed form, since the moments of a uniform hexagon are not memorable.

    Note what would *not* work: sampling the bounding box and folding the draw
    back in. `_full_intensity` off the polygon is the periodic extension, so the
    box covers parts of the polygon twice and others once.
    """
    hexagon = hp.FundamentalDomain.hexagon(1.0)
    rule = _integration.restrict(
        _integration.build(hexagon.bounds, 256), hexagon.contains, hexagon.volume_element
    )
    area = float(rule.weights.sum())

    rng = np.random.default_rng(2024)
    samples = np.array(
        [
            mcmc_sampler(
                lambda x: 1.0 if hexagon.contains(x) else 0.0,
                hexagon.bounds,
                n_iter=400,
                burn_in=120,
                seed=rng,
            )
            for _ in range(1500)
        ]
    )
    assert all(hexagon.contains(point) for point in samples)

    for label, fn in (
        ("x", lambda p: p[0]),
        ("y", lambda p: p[1]),
        ("x^2", lambda p: p[0] ** 2),
        ("|p|", lambda p: float(np.hypot(p[0], p[1]))),
        ("1[|p| < 0.5]", lambda p: float(np.hypot(p[0], p[1]) < 0.5)),
    ):
        values = np.array([fn(point) for point in samples])
        expected = rule.integrate(fn) / area
        tolerance = 5 * float(values.std()) / np.sqrt(len(values))
        assert abs(float(values.mean()) - expected) < tolerance, (
            f"E[{label}] is {values.mean():.5f}, expected {expected:.5f} "
            f"(5 standard errors is {tolerance:.5f})"
        )

    # Six equal-area sectors: the moments above are blind to a rotation.
    angles = np.arctan2(samples[:, 1], samples[:, 0])
    counts, _ = np.histogram(angles, bins=6, range=(-np.pi, np.pi))
    chi2 = stats.chisquare(counts)
    assert chi2.pvalue > 1e-3, f"the hexagon's sectors are unevenly filled (p={chi2.pvalue:.2e})"


@pytest.mark.statistical
def test_masked_quadrature_integrates_a_constant_to_the_polygon_area():
    """The background rate on a masked domain must be exactly ``mu * area``.

    The historical failure mode of every integration bug in this package is a
    rate that is quietly wrong by a constant factor, and for a domain that does
    not fill its bounding box the factor to get wrong is the fraction of the box
    the domain occupies -- 3/4 for a regular hexagon.
    """
    hexagon = hp.FundamentalDomain.hexagon(1.0)
    for mu in (0.5, 2.0, 7.3):
        process = hp.SpatioTemporalHawkesProcess(
            base=lambda x, rate=mu: rate,
            spatial=lambda d: 0.0,
            temporal=lambda dt: 0.0,
            domain=hexagon,
            monotone_temporal_kernel=True,
            rng=7,
        )
        assert process._integrated_intensity(0.0) == pytest.approx(mu * hexagon.volume, rel=1e-12)


@pytest.mark.statistical
@pytest.mark.slow
def test_simulation_on_a_fundamental_domain_stays_inside_it():
    """End to end: every simulated location lands in the polygon, off its edges.

    Events heaped on the boundary is the signature of a `wrap` that clips rather
    than folds, and it is what the pre-0.2.0 unbounded walk produced.
    """
    hexagon = hp.FundamentalDomain.hexagon(1.0)
    process = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 0.5,
        spatial=lambda d: max(0.0, 1.0 - d),
        temporal=lambda dt: 0.9 * np.exp(-2.0 * dt),
        domain=hexagon,
        monotone_temporal_kernel=True,
        n_quad=16,
        rng=11,
    )
    process.simulate(30)

    locations = process.Events[1:].T
    assert all(hexagon.contains(point) for point in locations)
    hugging = [point for point in locations if not hexagon._inside(point, -1e-3)]
    assert not hugging, f"{len(hugging)} of {len(locations)} events sit on the boundary"


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


@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_sampler_matches_a_narrow_target_on_a_wide_domain():
    """The proposal scale must follow the domain, not a hard-wired 1.0.

    On a circle of circumference 62.8 with a peak 0.5 wide, a unit proposal
    could not equilibrate in 2000 steps, so the draw was dominated by the
    uniform initial point: 30.3% of samples fell in a peak holding 52.2% of
    the mass.
    """
    domain = hp.Circle(radius=10.0)
    lo, hi = domain.bounds[0]

    def density(x):
        return 0.05 + 3.0 * np.exp(-(float(np.ravel(x)[0]) ** 2) / (2 * 0.25**2))

    samples = np.array(
        [
            float(mcmc_sampler(density, domain.bounds, n_iter=3000, burn_in=800, seed=s)[0])
            for s in range(300)
        ]
    )
    assert np.all((samples >= lo) & (samples <= hi))

    total = quad(lambda v: density(np.array([v])), lo, hi, limit=400)[0]
    in_peak = quad(lambda v: density(np.array([v])), -0.5, 0.5, limit=400)[0]
    assert float(np.mean(np.abs(samples) < 0.5)) == pytest.approx(in_peak / total, abs=0.08)


# ---------------------------------------------------------------------------
# The curved measure
# ---------------------------------------------------------------------------


class Weighted(hp.SpatialDomain):
    """A one-dimensional domain whose chart measure is deliberately not flat.

    Nothing about it is geometric — it is ``[0, 1]`` with the measure density
    ``1 + 3x``, chosen only so that the density is four times heavier at one end
    than the other. The point is to have a domain where the chart measure and
    the true measure differ *without* also changing the distance, the boundary,
    or anything else at the same time, so a failure can only mean one thing.
    """

    def distance(self, x, y):
        return abs(float(np.ravel(x)[0]) - float(np.ravel(y)[0]))

    def wrap(self, x):
        return np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, 1.0)

    def sample_uniform(self, rng):
        # Inverse transform for the density (1 + 3x) / 2.5 on [0, 1].
        u = rng.uniform()
        return np.array([(-1 + np.sqrt(1 + 15 * u)) / 3])

    def volume_element(self, x):
        return 1.0 + 3.0 * float(np.ravel(x)[0])

    @property
    def volume(self):
        return 2.5  # integral of 1 + 3x over [0, 1]

    @property
    def bounds(self):
        return np.array([[0.0, 1.0]])


@pytest.mark.statistical
def test_the_location_sampler_targets_the_surface_measure_not_the_chart():
    """The event location is distributed as ``lambda dA``, not as ``lambda dx``.

    The sampler walks in *chart* coordinates with a symmetric Gaussian proposal
    and accepts on the raw ratio, so the density it must be handed is
    ``lambda * volume_element``. Handing it ``lambda`` alone samples from the
    chart measure instead, which on this domain shifts the mean by 8% and on a
    sphere would pile every event at the poles.

    Dormant until 0.4.0 because every domain that had shipped by then carried
    ``volume_element == 1``, and silently wrong on the first one that did not —
    which is why this domain exists only in this file.
    """
    domain = Weighted()
    process = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 1.0,
        spatial=lambda d: 0.0,
        temporal=lambda dt: 0.0,
        domain=domain,
        monotone_temporal_kernel=True,
        rng=11,
    )

    rng = np.random.default_rng(4242)
    samples = np.array(
        [
            mcmc_sampler(
                lambda x: process._confined_density(x, 0.0),
                domain.bounds,
                n_iter=500,
                burn_in=200,
                seed=rng,
            )[0]
            for _ in range(1500)
        ]
    )

    # With a constant background the target is the measure itself, normalised:
    # (1 + 3x) / 2.5, whose mean is 0.6 and second moment 0.425. Ignoring the
    # measure would sample the chart uniformly instead, giving 0.5 and 1/3 --
    # both far outside the bands below, which is what makes this a test and not
    # a tolerance exercise.
    error = 4 * float(samples.std()) / np.sqrt(len(samples))
    assert float(samples.mean()) == pytest.approx(0.6, abs=error)
    assert float((samples**2).mean()) == pytest.approx(0.425, rel=0.08)
    assert abs(float(samples.mean()) - 0.5) > 0.05

    def cdf(x):
        """The integral of the density, for a Kolmogorov-Smirnov comparison."""
        return (x + 1.5 * x**2) / 2.5

    assert stats.kstest(samples, cdf).pvalue > 1e-3


@pytest.mark.statistical
def test_event_locations_on_a_sphere_are_uniform_in_area():
    """A constant background must scatter events evenly over the sphere.

    The chart is where this goes wrong. ``(theta, phi)`` gives every colatitude
    the same width, so a sampler handed the intensity without the area element
    concentrates events at the poles -- by a factor that goes to infinity there.
    ``cos(theta)`` is uniform on ``[-1, 1]`` exactly when the sample is uniform
    on the sphere, so its distribution is the whole test.

    Drawn from the conditional density directly rather than simulated: the
    location sampler is what is under test, and a run long enough to be
    statistically interesting costs quadratic time in the event count for no
    extra coverage.
    """
    sphere = hp.Sphere()
    process = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 1.0,
        spatial=lambda d: 0.0,
        temporal=lambda dt: 0.0,
        domain=sphere,
        monotone_temporal_kernel=True,
        rng=5,
    )

    rng = np.random.default_rng(909)
    samples = np.array(
        [
            mcmc_sampler(
                lambda x: process._confined_density(x, 0.0),
                sphere.bounds,
                n_iter=400,
                burn_in=150,
                seed=rng,
            )
            for _ in range(1000)
        ]
    )

    heights = np.cos(samples[:, 0])
    assert abs(float(heights.mean())) < 4 / np.sqrt(3 * len(heights))
    assert float(heights.var()) == pytest.approx(1 / 3, rel=0.15)
    assert stats.kstest(heights, "uniform", args=(-1, 2)).pvalue > 1e-3


@pytest.mark.statistical
def test_the_background_rate_on_a_sphere_is_mu_times_its_area():
    """The integrated intensity must be ``mu * 4 pi R^2``, exactly.

    The same check the hexagon gets, on the domain where the factor to get wrong
    is not the fraction of a box but the whole curved measure.
    """
    for radius in (1.0, 2.0):
        sphere = hp.Sphere(radius)
        process = hp.SpatioTemporalHawkesProcess(
            base=lambda x: 0.75,
            spatial=lambda d: 0.0,
            temporal=lambda dt: 0.0,
            domain=sphere,
            monotone_temporal_kernel=True,
            rng=3,
        )
        assert process._integrated_intensity(0.0) == pytest.approx(0.75 * sphere.volume, rel=1e-9)
