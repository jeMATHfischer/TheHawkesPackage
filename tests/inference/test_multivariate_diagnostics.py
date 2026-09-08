"""Residuals and forecasting on a multivariate fit.

Almost nothing here is new code, and that is the finding worth pinning.
`residuals` rescales time through whatever compensator it is handed, and
`posterior_predictive` slices `record[:, observed:]` on the non-scalar branch --
which is already right for a ``(2, n)`` record. So these tests exist to hold
that true rather than to exercise something added.

The residual test does more than that, though: time-rescaling on the *pooled*
process is an end-to-end check of the compensator that does not share the
log-likelihood's arithmetic. A compensator computed too small inflates the
rescaled gaps, and the KS test sees it.
"""

import numpy as np
import pytest

from hawkes_package.inference import (
    History,
    MultivariateExponentialLogLikelihood,
    MultivariateLogLikelihood,
    ParticleCloud,
    ks_exponential,
    multivariate_model,
    posterior_predictive,
    predictive_counts,
    residuals,
)

TRUTH = np.array([0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 2.0])


@pytest.fixture
def model():
    return multivariate_model(2)


@pytest.fixture
def history(model):
    process = model(TRUTH, rng=11)
    process.simulate(1500)
    return History.from_multivariate_events(
        process.events, n_types=2, end=float(process.events[0, -1])
    )


@pytest.mark.statistical
def test_time_rescaling_at_the_truth_is_uniform(model, history):
    """The pooled process, rescaled by its own compensator, must be unit-rate.

    Independent evidence that the compensator is right, and the strongest
    available: it does not reuse the log-sum's arithmetic. Measured at seed 11
    with 1500 events -- mean gap 0.978 against 1, KS p = 0.905. A compensator
    20% too small would push the mean to 1.25 and the KS test would reject.
    """
    values = residuals(MultivariateExponentialLogLikelihood(model), TRUTH, history)

    assert values.size == history.n_events
    assert float(values.mean()) == pytest.approx(1.0, abs=0.15)
    assert ks_exponential(values).pvalue > 1e-3


@pytest.mark.statistical
def test_the_two_backends_give_the_same_residuals(model, history):
    """The closed form and the quadrature path rescale to the same gaps."""
    closed = residuals(MultivariateExponentialLogLikelihood(model), TRUTH, history)
    general = residuals(MultivariateLogLikelihood(model), TRUTH, history)
    np.testing.assert_allclose(closed, general, rtol=1e-4)


def test_forecast_paths_carry_types_and_stay_in_the_window(model, history):
    """`posterior_predictive` needs no change for a ``(2, n)`` record."""
    cloud = ParticleCloud(np.tile(TRUTH, (16, 1)), np.full(16, -np.log(16)), model.spec)
    horizon = 20.0
    paths = [
        np.asarray(path)
        for path in posterior_predictive(model, cloud, history, horizon=horizon, n_paths=8, rng=0)
    ]

    assert len(paths) == 8
    for path in paths:
        assert path.shape[0] == 2, "a forecast path keeps the record's layout"
        if path.size == 0:
            continue
        assert np.all(path[0] > history.end), "paths must start after the observed window"
        assert np.all(path[0] <= history.end + horizon)
        assert set(path[1].astype(int).tolist()) <= {0, 1}

    counts = predictive_counts(paths)
    assert counts.shape == (8,)
    assert np.all(counts >= 0)


def test_forecasting_does_not_mutate_the_observed_history(model, history):
    """Each path conditions a fresh copy, so the caller's array is its own."""
    cloud = ParticleCloud(np.tile(TRUTH, (8, 1)), np.full(8, -np.log(8)), model.spec)
    before_times = history.times.copy()
    before_types = history.types.copy()

    posterior_predictive(model, cloud, history, horizon=10.0, n_paths=4, rng=1)

    np.testing.assert_array_equal(history.times, before_times)
    np.testing.assert_array_equal(history.types, before_types)
    assert history.n_events == before_times.size
