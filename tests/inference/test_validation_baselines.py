"""Is the fitted model worth having, or only not obviously wrong.

A goodness-of-fit test answers the second question. Nothing in the package
answered the first until now: a Hawkes fit that passes every residual check and
predicts no better than a constant rate has found nothing.

Swept over seeds 0-20 at a fixed horizon. On Hawkes data the fit beats the best
constant rate on 21 of 21 seeds, gaining 0.027 nats per event on average. On
data with the excitation switched off it beats it on 0 of 21 — which is the half
that makes the first half mean something.
"""

import math

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    SpatioTemporalLogLikelihood,
    exponential_model,
    spatio_temporal_model,
)
from hawkes_package.inference.validation import (
    compare_with_baseline,
    homogeneous_log_likelihood,
)

EXCITED = np.array([2.0, 0.5, 1.0])
#: The same background with the excitation as good as switched off. Not exactly
#: zero: every coordinate lives on an *open* positive interval.
FLAT = np.array([2.0, 1e-6, 1.0])
HORIZON = 200.0


@pytest.fixture
def likelihood():
    return ExponentialLogLikelihood(exponential_model())


def observed(theta, seed):
    process = hp.ExponentialHawkes(theta, rng=seed)
    process.simulate_until(HORIZON)
    return History.from_events(process.events, end=HORIZON)


def test_the_baseline_is_the_best_constant_rate_not_a_convenient_one(likelihood):
    """`n log(n/T) - n`, the maximum over the rate, in closed form.

    A baseline handed a bad parameter is worse than no baseline, because beating
    it reads as evidence. This one cannot be beaten by any constant rate.
    """
    history = observed(EXCITED, 0)
    best = homogeneous_log_likelihood(likelihood, history)
    n, span = history.n_events, history.end - history.start

    for rate in (0.5, 1.0, n / span * 0.8, n / span * 1.2, 5.0):
        worse = n * math.log(rate) - rate * span
        assert worse <= best + 1e-9, f"rate {rate} beat the supposed maximum"

    assert best == pytest.approx(n * math.log(n / span) - n)


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 6, 17])
def test_a_hawkes_fit_beats_a_constant_rate_on_hawkes_data(likelihood, seed):
    """Seeds 0-20: beaten on 21 of 21, per-event gain from 0.0004 to a mean of 0.027."""
    comparison = compare_with_baseline(likelihood, EXCITED, observed(EXCITED, seed))
    assert comparison.beats_baseline
    assert comparison.improvement > 0.0
    assert "beats" in comparison.summary()


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 6, 17])
def test_it_does_not_beat_a_constant_rate_on_constant_rate_data(likelihood, seed):
    """Guard the guard: the bar must not be clearable by arithmetic.

    With the excitation switched off the model *is* a Poisson process, at the
    fitted `mu` rather than at the maximum-likelihood rate, so it must lose --
    narrowly and reliably. Seeds 0-20: positive on 0 of 21, mean -0.0008 per
    event. If this ever passes, the baseline has become something a model beats
    for free.
    """
    comparison = compare_with_baseline(likelihood, FLAT, observed(FLAT, seed))
    assert not comparison.beats_baseline
    assert "LOSES TO" in comparison.summary()


def test_the_gain_is_reported_per_event(likelihood):
    """Two windows of different length are only comparable per event."""
    history = observed(EXCITED, 0)
    comparison = compare_with_baseline(likelihood, EXCITED, history)
    assert comparison.n_events == history.n_events
    assert comparison.per_event == pytest.approx(comparison.improvement / history.n_events)


def test_an_empty_history_gives_minus_infinity(likelihood):
    """A constant rate of zero assigns that to anything, including nothing."""
    empty = History(np.empty(0), None, 0.0, end=5.0)
    assert homogeneous_log_likelihood(likelihood, empty) == -math.inf
    assert compare_with_baseline(likelihood, EXCITED, empty).per_event == 0.0


@pytest.mark.slow
def test_the_domain_measure_enters_a_spatio_temporal_baseline():
    """A spatio-temporal intensity is a density in space too.

    Without the domain measure the baseline would be off by ``n log|D|`` and
    would credit or penalise the model for the size of the surface it lives on
    -- a factor of 6.28 on a unit circle.
    """
    model = spatio_temporal_model(hp.Circle())
    theta = np.array([0.5, 0.9, 2.0, 1.0])
    process = model(theta, rng=1)
    process.simulate(8)
    history = History.from_simulation(process)

    likelihood = SpatioTemporalLogLikelihood(model, backend="cached")
    baseline = homogeneous_log_likelihood(likelihood, history)

    n = history.n_events
    measure = (history.end - history.start) * model.components.domain.volume
    assert baseline == pytest.approx(n * math.log(n / measure) - n)

    # And the temporal formula would have been visibly different.
    temporal_only = n * math.log(n / (history.end - history.start)) - n
    assert abs(baseline - temporal_only) > 1.0
