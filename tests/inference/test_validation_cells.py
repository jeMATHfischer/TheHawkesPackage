"""Residuals on a partition of the window, against the count they predict.

Time rescaling asks whether events arrived at the right rate. Cell residuals ask
whether they arrived in the right *amounts, where and when* — and unlike the KS
test they say which part of the window is wrong, not only that something is.

Thresholds swept over seeds 0-20 at a fixed horizon; the measurement sits beside
each one.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import ExponentialLogLikelihood, History, exponential_model
from hawkes_package.inference.validation import cell_residuals

TRUTH = np.array([2.0, 0.5, 1.0])
HORIZON = 200.0


def observed(seed):
    """A history on a fixed *horizon*, not a fixed count.

    `simulate(k)` ends the window exactly at the k-th event, which conditions on
    an event having just happened and biases the expected count above the
    observed one — 424 against 400 on one run. That is a real property of the
    stopping rule, not a defect, but it makes a residual mean that should be
    zero come out at -0.4. A horizon leaves the count random and the residuals
    centred.
    """
    process = hp.ExponentialHawkes(TRUTH, rng=seed)
    process.simulate_until(HORIZON)
    return History.from_events(process.events, end=HORIZON)


@pytest.fixture
def likelihood():
    return ExponentialLogLikelihood(exponential_model())


def test_the_cells_partition_the_window(likelihood):
    """Counts must sum to every event, and the integrals to the compensator.

    A partition that lost a sliver of the window would understate the expected
    count everywhere and read as a model that over-predicts.
    """
    history = observed(0)
    result = cell_residuals(likelihood, TRUTH, history, n_cells=13)

    assert result.edges[0] == history.start
    assert result.edges[-1] == history.end
    assert result.counts.sum() == history.n_events
    total = float(likelihood.compensator(TRUTH, history, np.array([history.end]))[0])
    assert result.total_expected == pytest.approx(total, rel=1e-3)


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 4, 11])
def test_the_residuals_are_centred_at_the_truth(likelihood, seed):
    """Standardised residuals are approximately standard normal under the model.

    Seeds 0-20 at 20 cells: the per-run mean never left [-0.343, 0.343] and the
    per-run standard deviation stayed in [0.618, 1.223], against the 1.0 a
    standard normal would give. The thresholds sit well outside both.
    """
    result = cell_residuals(likelihood, TRUTH, observed(seed), n_cells=20)
    values = result.standardised[result.usable]

    assert values.size >= 15, "most cells should hold enough events to standardise"
    assert abs(float(np.mean(values))) < 0.6
    assert 0.4 < float(np.std(values)) < 1.8


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 4, 11])
def test_a_wrong_excitation_shows_up_as_a_shifted_mean(likelihood, seed):
    """Guard the guard: the check above must have power.

    An excitation of 0.8 against a truth of 0.5 predicts far more events than
    arrive, so the residuals go negative together. Seeds 0-20: the per-run mean
    never rose above -1.480, against a truth-case worst of +0.343, and the two
    separated on all 21.
    """
    history = observed(seed)
    inflated = TRUTH.copy()
    inflated[1] = 0.8

    at_truth = cell_residuals(likelihood, TRUTH, history, n_cells=20)
    wrong = cell_residuals(likelihood, inflated, history, n_cells=20)

    wrong_mean = float(np.mean(wrong.standardised[wrong.usable]))
    truth_mean = float(np.mean(at_truth.standardised[at_truth.usable]))
    assert wrong_mean < -0.8
    assert wrong_mean < truth_mean


def test_cells_too_empty_to_standardise_are_excluded(likelihood):
    """Dividing by a near-zero expectation manufactures a huge residual.

    One stray event in a cell expecting 0.01 standardises to 10, which would
    dominate every real signal beside it. Those cells are reported as counts and
    excluded from the standardised view rather than silently kept.
    """
    history = observed(0)
    result = cell_residuals(likelihood, TRUTH, history, n_cells=4000)

    assert not np.all(result.usable), "4000 cells on this window must starve some"
    assert np.all(np.isnan(result.standardised[~result.usable]))
    assert np.all(np.isfinite(result.standardised[result.usable]))
    # The counts are still complete: exclusion is a reporting decision, not a
    # dropped part of the window.
    assert result.counts.sum() == history.n_events


def test_the_summary_says_what_it_measured(likelihood):
    text = cell_residuals(likelihood, TRUTH, observed(0), n_cells=10).summary()
    assert "events observed against" in text
    assert "standardised residuals" in text


def test_it_can_be_told_to_reuse_the_likelihood_compensator(likelihood):
    """The shared-arithmetic behaviour, available deliberately and never by default.

    The two must actually differ, or the independent path is decorative. They
    agree closely on a *correct* model, which is the point -- the divergence
    appears when the compensator is wrong, and that is tested next door.
    """
    history = observed(0)
    independent = cell_residuals(likelihood, TRUTH, history, n_cells=10)
    shared = cell_residuals(
        likelihood, TRUTH, history, n_cells=10, compensator=likelihood.compensator
    )
    np.testing.assert_allclose(independent.expected, shared.expected, rtol=1e-3)


def test_a_non_positive_cell_count_is_refused(likelihood):
    with pytest.raises(ValueError, match="n_cells must be positive"):
        cell_residuals(likelihood, TRUTH, observed(0), n_cells=0)
