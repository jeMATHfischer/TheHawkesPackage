"""The cancellation this subpackage exists to catch.

A fit made with a compensator 20% too small inflates the intensity, and then
rescaling the events through that *same* broken integral gives unit-rate gaps.
The goodness-of-fit test passes, and the more badly the compensator is wrong the
more exactly the fit compensates for it.

Measured below on 400 events from the exponential model: a 20%-too-small
integral pushes `alpha` from 0.500 to 0.690, the residuals taken through the
broken compensator give KS p = 0.46, and the same residuals through an honest
one give p = 3.3e-05. `compensator_agreement` reports 0.2 -- exactly the defect.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    TemporalLogLikelihood,
    exponential_model,
    ks_exponential,
    residuals,
    spatio_temporal_model,
)
from hawkes_package.inference.validation import compensator_agreement, independent_compensator

TRUTH = np.array([2.0, 0.5, 1.0])


@pytest.fixture
def history():
    process = hp.ExponentialHawkes(TRUTH, rng=0)
    process.simulate(400)
    return History.from_simulation(process)


class ScaledCompensator(ExponentialLogLikelihood):
    """A likelihood whose integral is deliberately short by a fixed factor.

    Exactly the defect `CLAUDE.md` names: not a random perturbation but a
    systematic under-integration, which is what a quadrature rule too coarse for
    its integrand actually does.
    """

    scale = 0.8

    def compensator(self, theta, history, times):
        return self.scale * super().compensator(theta, history, times)


def test_it_agrees_with_the_closed_form(history):
    """Simpson on a dense uniform grid against the exact exponential integral.

    Measured at 2.9e-05 relative on 400 events. That is the grid resolving a
    *discontinuous* integrand -- the intensity jumps at every event and a uniform
    rule cannot place a node there -- and it is three hundred times finer than
    the smallest bias worth reporting.
    """
    likelihood = ExponentialLogLikelihood(exponential_model())
    assert compensator_agreement(likelihood, TRUTH, history) < 1e-3


def test_it_agrees_with_the_quadrature_path_too(history):
    """The other temporal implementation, which uses panelled Gauss-Legendre."""
    likelihood = TemporalLogLikelihood(exponential_model())
    assert compensator_agreement(likelihood, TRUTH, history) < 1e-3


def test_the_result_is_non_decreasing(history):
    likelihood = ExponentialLogLikelihood(exponential_model())
    values = independent_compensator(likelihood, TRUTH, history, history.times)
    assert values.size == history.n_events
    assert np.all(np.diff(values) >= 0.0)
    assert values[0] >= 0.0


def test_it_never_calls_the_likelihood_compensator(history, monkeypatch):
    """The one thing that would make the whole subpackage theatre."""
    likelihood = ExponentialLogLikelihood(exponential_model())

    def refuse(*args, **kwargs):
        raise AssertionError("independent_compensator reused the likelihood's integral")

    monkeypatch.setattr(likelihood, "compensator", refuse)
    independent_compensator(likelihood, TRUTH, history, history.times[:20])


def test_it_catches_a_compensator_the_residuals_cannot(history):
    """The whole argument, in one test.

    Three assertions in sequence, and each is load-bearing. The residuals taken
    through the broken compensator must *pass* -- otherwise there is nothing to
    catch and this file is unnecessary. The same residuals through an honest
    compensator must *reject* -- otherwise the two are not really different.
    And the agreement number must report the defect at its actual size.
    """
    broken = ScaledCompensator(exponential_model())
    honest = ExponentialLogLikelihood(exponential_model())

    # The parameter a 20%-too-small integral talks you into: the log-sum is
    # unchanged, so the fit raises the excitation until the smaller penalty
    # balances it. Measured at 0.690 against a truth of 0.500.
    inflated = TRUTH.copy()
    inflated[1] = 0.690

    passing = ks_exponential(residuals(broken, inflated, history))
    assert passing.pvalue > 0.05, (
        "the broken compensator should hide the bias -- if it does not, this test "
        f"is not exercising the cancellation (p={passing.pvalue:.4g})"
    )

    rejected = ks_exponential(residuals(honest, inflated, history))
    assert rejected.pvalue < 1e-3, (
        f"an honest compensator must reject the inflated fit (p={rejected.pvalue:.4g})"
    )

    assert compensator_agreement(broken, inflated, history) == pytest.approx(0.2, rel=1e-3)
    assert compensator_agreement(honest, inflated, history) < 1e-3


def test_an_empty_query_and_an_empty_history_are_handled():
    likelihood = ExponentialLogLikelihood(exponential_model())
    empty = History(np.empty(0), None, 0.0, end=5.0)
    assert independent_compensator(likelihood, TRUTH, empty, np.empty(0)).size == 0
    assert compensator_agreement(likelihood, TRUTH, empty) == 0.0


def test_unsorted_query_times_are_refused(history):
    likelihood = ExponentialLogLikelihood(exponential_model())
    with pytest.raises(ValueError, match="sorted"):
        independent_compensator(likelihood, TRUTH, history, np.array([2.0, 1.0]))


@pytest.mark.slow
def test_it_works_on_a_spatio_temporal_model():
    """Through the space-integrated hook, which is the expensive one.

    Slow, and unavoidably: the hook integrates over the whole domain per call,
    and this integrator asks for thousands of calls precisely so that it does
    not have to know where the jumps are. That trade is the design, not an
    oversight -- the accuracy is bought with nodes rather than with an
    assumption shared with the thing being checked.
    """
    from hawkes_package.inference import SpatioTemporalLogLikelihood

    model = spatio_temporal_model(hp.Circle())
    theta = np.array([0.5, 0.9, 2.0, 1.0])
    process = model(theta, rng=1)
    process.simulate(5)
    observed = History.from_simulation(process)

    likelihood = SpatioTemporalLogLikelihood(model, backend="cached")
    assert compensator_agreement(likelihood, theta, observed) < 1e-2
