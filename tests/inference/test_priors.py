"""Marginal densities, the independent product, and constrained sampling.

The log-pdfs are checked against ``scipy.stats``, which is a development
dependency and may be used freely in tests -- the package itself keeps SciPy to
one call site, which is why they are hand-written in the first place.
"""

import numpy as np
import pytest
from scipy import stats

from hawkes_package.inference import (
    ConstrainedPrior,
    Gamma,
    IndependentPrior,
    LogNormal,
    Normal,
    Uniform,
    exponential_model,
    stationarity,
)

MARGINALS = {
    "lognormal": (LogNormal(0.4, 0.8), stats.lognorm(s=0.8, scale=np.exp(0.4))),
    "gamma": (Gamma(2.5, 1.3), stats.gamma(a=2.5, scale=1 / 1.3)),
    "normal": (Normal(-0.5, 2.0), stats.norm(loc=-0.5, scale=2.0)),
    "uniform": (Uniform(-1.0, 3.0), stats.uniform(loc=-1.0, scale=4.0)),
}


@pytest.fixture(params=sorted(MARGINALS), ids=sorted(MARGINALS))
def marginal(request):
    return MARGINALS[request.param]


def test_log_pdf_matches_scipy(marginal):
    ours, theirs = marginal
    grid = np.linspace(-2.0, 6.0, 401)
    mine = ours.log_pdf(grid)
    reference = theirs.logpdf(grid)
    finite = np.isfinite(reference)
    np.testing.assert_allclose(mine[finite], reference[finite], rtol=1e-10, atol=1e-10)
    # Off the support both must be -inf, not merely small.
    assert np.all(np.isneginf(mine[~finite]))


def test_samples_land_in_the_support(marginal, rng):
    ours, _ = marginal
    draws = ours.sample(2000, rng)
    assert draws.shape == (2000,)
    assert np.all(np.isfinite(ours.log_pdf(draws)))


def test_sample_moments_are_right(marginal, rng):
    """Four standard errors, so a correct sampler passes for every seed."""
    ours, theirs = marginal
    draws = ours.sample(20_000, rng)
    mean, variance = theirs.stats(moments="mv")
    tolerance = 4 * np.sqrt(float(variance) / draws.size)
    assert abs(draws.mean() - float(mean)) < tolerance


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (LogNormal, {"mean_log": 0.0, "sd_log": 0.0}),
        (Gamma, {"shape": 0.0, "rate": 1.0}),
        (Normal, {"mean": 0.0, "sd": -1.0}),
        (Uniform, {"low": 1.0, "high": 1.0}),
    ],
)
def test_degenerate_parameters_are_refused(factory, kwargs):
    with pytest.raises(ValueError, match=r"must be positive|low < high"):
        factory(**kwargs)


# ---------------------------------------------------------------------------
# Joint priors
# ---------------------------------------------------------------------------


def test_independent_prior_sums_its_marginals(rng):
    marginals = (LogNormal(0.0, 1.0), Normal(1.0, 0.5), Uniform(0.0, 2.0))
    prior = IndependentPrior(marginals)
    theta = prior.sample(50, rng)
    assert theta.shape == (50, 3)
    expected = sum(m.log_pdf(theta[:, j]) for j, m in enumerate(marginals))
    np.testing.assert_allclose(prior.log_pdf(theta), expected, rtol=1e-12)


def test_independent_prior_refuses_the_wrong_width(rng):
    prior = IndependentPrior((LogNormal(0.0, 1.0), Normal(0.0, 1.0)))
    with pytest.raises(ValueError, match="marginal"):
        prior.log_pdf(np.ones((4, 3)))


def test_an_empty_prior_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        IndependentPrior(())


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_constrained_prior_samples_only_inside_the_support(rng):
    model = exponential_model()
    base = IndependentPrior((LogNormal(0.0, 1.0), LogNormal(0.0, 1.0), LogNormal(0.0, 1.0)))
    prior = ConstrainedPrior(base, model.support)
    theta = prior.sample(500, rng)
    assert theta.shape == (500, 3)
    assert np.all(model.support(theta))
    assert np.all(theta[:, 1] < theta[:, 2]), "alpha >= beta is outside the support"


def test_constrained_prior_is_minus_inf_outside_and_the_base_inside(rng):
    model = exponential_model()
    base = IndependentPrior((LogNormal(0.0, 1.0), LogNormal(0.0, 1.0), LogNormal(0.0, 1.0)))
    prior = ConstrainedPrior(base, model.support)
    theta = np.array([[1.0, 0.5, 2.0], [1.0, 3.0, 2.0]])
    values = prior.log_pdf(theta)
    assert np.isfinite(values[0])
    assert np.isneginf(values[1])
    np.testing.assert_allclose(values[0], base.log_pdf(theta[:1])[0], rtol=1e-12)


def test_a_support_the_prior_never_reaches_raises_rather_than_looping(rng):
    """The failure mode of rejection sampling, reported with the measured rate."""
    base = IndependentPrior((Normal(0.0, 1.0),))
    prior = ConstrainedPrior(base, lambda theta: theta[:, 0] > 50.0, max_draws=5000)
    with pytest.raises(RuntimeError, match="barely overlap"):
        prior.sample(10, rng)


def test_stationarity_reads_the_branching_ratio():
    model = exponential_model()
    keep = stationarity(model.branching_ratio, limit=0.5)
    theta = np.array([[1.0, 0.4, 2.0], [1.0, 1.5, 2.0], [1.0, 0.9, 2.0]])
    # ratios 0.2, 0.75, 0.45
    np.testing.assert_array_equal(keep(theta), [True, False, True])


def test_stationarity_refuses_a_non_positive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        stationarity(exponential_model().branching_ratio, limit=0.0)
