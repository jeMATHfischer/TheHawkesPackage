"""Time-rescaling residuals and the hand-rolled Kolmogorov-Smirnov test.

The KS machinery is written out in the package because SciPy is held to one call
site there. Here SciPy is a development dependency, so the natural check is
against :func:`scipy.stats.kstest` -- which is the whole reason the package
version can be trusted.
"""

import numpy as np
import pytest
from scipy import stats

import hawkes_package as hp
from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    KSResult,
    ParticleCloud,
    ks_exponential,
    posterior_report,
    residuals,
)
from hawkes_package.inference.diagnostics import kolmogorov_sf

# ---------------------------------------------------------------------------
# The KS test against SciPy
# ---------------------------------------------------------------------------

SAMPLES = {
    "exponential-200": np.random.default_rng(0).exponential(size=200),
    "exponential-1000": np.random.default_rng(1).exponential(size=1000),
    "too-fast": np.random.default_rng(2).exponential(scale=0.6, size=400),
    "too-slow": np.random.default_rng(3).exponential(scale=1.7, size=400),
    "uniform": np.random.default_rng(4).uniform(0.0, 2.0, size=300),
}


@pytest.fixture(params=sorted(SAMPLES), ids=sorted(SAMPLES))
def sample(request):
    return SAMPLES[request.param]


def test_the_statistic_matches_scipy_exactly(sample):
    """The statistic is a definition, so it must agree to rounding."""
    ours = ks_exponential(sample)
    theirs = stats.kstest(sample, "expon")
    assert ours.statistic == pytest.approx(theirs.statistic, rel=1e-12)


def test_the_pvalue_agrees_with_scipy_where_it_matters(sample):
    """Close enough that the two never disagree about the 1e-3 threshold.

    The package's p-value is the asymptotic Kolmogorov series with the standard
    small-sample correction; SciPy's is exact for these sizes. They differ by a
    few per cent, which is far below what any decision here turns on.
    """
    ours = ks_exponential(sample).pvalue
    theirs = float(stats.kstest(sample, "expon").pvalue)
    assert (ours > 1e-3) == (theirs > 1e-3)
    if theirs > 1e-6:
        assert ours == pytest.approx(theirs, rel=0.1, abs=0.02)


def test_a_rate_other_than_one_is_honoured():
    values = np.random.default_rng(5).exponential(scale=1 / 3.0, size=500)
    assert ks_exponential(values, rate=3.0).pvalue > 1e-3
    assert ks_exponential(values, rate=1.0).pvalue < 1e-3


def test_the_kolmogorov_survival_function_is_a_survival_function():
    assert kolmogorov_sf(0.0) == 1.0
    assert kolmogorov_sf(-1.0) == 1.0
    assert kolmogorov_sf(20.0) == 0.0
    grid = np.linspace(0.05, 4.0, 200)
    values = [kolmogorov_sf(float(x)) for x in grid]
    assert np.all(np.diff(values) <= 1e-12), "must be non-increasing"
    assert np.all((np.array(values) >= 0.0) & (np.array(values) <= 1.0))


def test_the_survival_function_matches_scipy():
    grid = np.linspace(0.2, 3.0, 40)
    ours = np.array([kolmogorov_sf(float(x)) for x in grid])
    theirs = stats.kstwobign.sf(grid)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("values", "match"),
    [([], "empty"), ([1.0, -0.5], "cannot be negative")],
)
def test_bad_samples_are_refused(values, match):
    with pytest.raises(ValueError, match=match):
        ks_exponential(np.asarray(values, dtype=float))


def test_a_zero_rate_is_refused():
    with pytest.raises(ValueError, match="rate must be positive"):
        ks_exponential(np.array([1.0, 2.0]), rate=0.0)


def test_the_result_reprs_readably():
    assert "statistic" in repr(KSResult(0.1, 0.5))


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def test_residuals_are_the_compensator_increments(history, exp_model, theta):
    likelihood = ExponentialLogLikelihood(exp_model)
    gaps = residuals(likelihood, theta, history)
    compensated = likelihood.compensator(theta, history, history.times)
    np.testing.assert_allclose(gaps, np.diff(np.concatenate([[0.0], compensated])), rtol=1e-12)
    assert gaps.size == history.n_events
    assert np.all(gaps > 0)


def test_the_first_gap_is_measured_from_the_window_start(history, exp_model, theta):
    """It is the residual a wrong background rate distorts most."""
    likelihood = ExponentialLogLikelihood(exp_model)
    gaps = residuals(likelihood, theta, history)
    first = likelihood.compensator(theta, history, history.times[:1])[0]
    assert gaps[0] == pytest.approx(first, rel=1e-12)


def test_residuals_of_an_empty_history_are_empty(exp_model, theta):
    empty = History(np.array([]), None, 0.0, end=3.0)
    assert residuals(ExponentialLogLikelihood(exp_model), theta, empty).size == 0


def test_a_decreasing_compensator_is_refused(history, exp_model, theta):
    """Not a rounding artefact: the integral of a non-negative function."""

    class Broken(ExponentialLogLikelihood):
        def compensator(self, theta, history, times):
            return -np.asarray(super().compensator(theta, history, times))

    with pytest.raises(ValueError, match="compensator decreased"):
        residuals(Broken(exp_model), theta, history)


@pytest.mark.statistical
def test_residuals_at_the_truth_are_unit_exponential():
    """The theorem itself: rescaled gaps from the true parameter are Exp(1)."""
    truth = np.array([2.0, 0.5, 1.0])
    process = hp.ExponentialHawkes(truth, rng=13)
    process.simulate(1500)
    history = History.from_simulation(process)
    gaps = residuals(ExponentialLogLikelihood(hp.inference.exponential_model()), truth, history)
    assert ks_exponential(gaps).pvalue > 1e-3


@pytest.mark.statistical
def test_a_wrong_compensator_is_rejected():
    """Guard the guard: scale the compensator by 0.8 and the test must reject.

    Without this the whole battery could pass vacuously -- a `residuals` that
    returned something else, or a `ks_exponential` that never rejected, would go
    unnoticed while every other assertion still held.
    """
    truth = np.array([2.0, 0.5, 1.0])
    process = hp.ExponentialHawkes(truth, rng=13)
    process.simulate(1500)
    history = History.from_simulation(process)

    class Undercounting(ExponentialLogLikelihood):
        """A compensator 20% too small -- the failure this module exists for."""

        def compensator(self, theta, history, times):
            return 0.8 * np.asarray(super().compensator(theta, history, times))

    model = hp.inference.exponential_model()
    honest = residuals(ExponentialLogLikelihood(model), truth, history)
    broken = residuals(Undercounting(model), truth, history)
    assert ks_exponential(honest).pvalue > 1e-3
    assert ks_exponential(broken).pvalue < 1e-3


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_covers_the_marginals_and_the_residuals(history, exp_model, theta, rng):
    cloud = ParticleCloud(
        np.tile(theta, (16, 1)) * rng.lognormal(0.0, 0.05, size=(16, 3)),
        np.full(16, -np.log(16)),
        exp_model.spec,
    )
    text = posterior_report(cloud, ExponentialLogLikelihood(exp_model), history, truth=theta)
    assert "time-rescaling" in text
    assert "coverage" in text
    assert "mu" in text
