"""Fitting a marked model, and the term that is not optional.

For marks drawn independently of the history the mark density does not depend on
``mu``, ``alpha``, ``beta`` or ``scale``, so an optimiser over those four may
treat it as a constant. It is included by default anyway, and this file measures
why: **without it, `b_value` does not move the likelihood at all.**

Its only other appearance is in the *support*, as the stationarity boundary
``scale < b_value``. Drop the mark term and the parameter is identified by a
constraint rather than by data, and the posterior reported for it is the prior
truncated at a line.
"""

import math

import numpy as np
import pytest

from hawkes_package.inference import (
    History,
    MarkedLogLikelihood,
    TemporalLogLikelihood,
    exponential_model,
    marked_model,
)

TRUTH = np.array([1.0, 0.3, 2.0, 0.5, 1.5])  # mu, alpha, beta, scale, b_value


@pytest.fixture
def model():
    return marked_model()


@pytest.fixture
def history(model):
    process = model(TRUTH, rng=3)
    process.simulate(300)
    return History.from_marked_events(process.events, end=float(process.events[0, -1]))


def test_the_coordinates_extend_the_exponential_model(model):
    assert model.spec.names == ("mu", "alpha", "beta", "scale", "b_value")
    assert model.spec.names[:3] == exponential_model().spec.names
    assert model.family == "marked"
    assert model.ndim == 0


def test_the_support_refuses_a_divergent_expectation(model):
    """A relation between two coordinates, which `spec.contains` cannot express.

    It enters through the branching callable instead, which returns `inf` there
    and lets the existing `isfinite(ratio) & (ratio < 1)` gate do the work.
    """
    batch = np.array(
        [
            [1.0, 0.3, 2.0, 0.5, 1.5],  # ratio 0.225
            [1.0, 0.3, 2.0, 1.6, 1.5],  # scale above b_value: E[g] = inf
            [1.0, 0.3, 2.0, 1.5, 1.5],  # exactly at the boundary
            [1.0, 1.0, 1.0, 0.5, 1.5],  # finite expectation, ratio 1.5
        ]
    )
    ratios = model.branching_ratio(batch)
    assert ratios[0] == pytest.approx(0.225)
    assert math.isinf(ratios[1])
    assert math.isinf(ratios[2])
    assert ratios[3] == pytest.approx(1.5)
    np.testing.assert_array_equal(model.support(batch), [True, False, False, False])


def test_computing_the_branching_ratio_warns_about_nothing(model):
    """`filterwarnings = ["error"]`, and the divergent branch is a division by zero.

    Written to fill `inf` and compute the ratio only where it is finite, rather
    than with `np.where`, which evaluates both branches and warns.
    """
    batch = np.tile(TRUTH, (64, 1))
    batch[:, 3] = np.linspace(0.1, 3.0, 64)  # crosses b_value = 1.5
    ratios = model.branching_ratio(batch)
    assert np.all(np.isinf(ratios[batch[:, 3] >= batch[:, 4]]))
    assert np.all(np.isfinite(ratios[batch[:, 3] < batch[:, 4]]))


def test_the_mark_term_is_its_closed_form(model, history):
    """`n log b - b sum(m - m0)`, added to the process term and nothing else."""
    with_marks = MarkedLogLikelihood(model).total(TRUTH, history)
    without = MarkedLogLikelihood(model, include_mark_density=False).total(TRUTH, history)

    b = float(TRUTH[4])
    expected = history.n_events * math.log(b) - b * float(np.sum(history.marks))
    assert with_marks - without == pytest.approx(expected, rel=1e-12)


def test_without_the_mark_term_b_value_does_nothing(model, history):
    """The measurement that makes the default the right default.

    Three very different mark rates, one log-likelihood. `b_value` enters the
    process term not at all -- it is a property of the mark law, and the ground
    process does not know it exists.
    """
    likelihood = MarkedLogLikelihood(model, include_mark_density=False)
    values = []
    for rate in (1.2, 1.5, 2.0):
        theta = TRUTH.copy()
        theta[4] = rate
        values.append(likelihood.total(theta, history))

    assert values[0] == values[1] == values[2]


@pytest.mark.statistical
def test_with_the_mark_term_b_value_is_identified(model, history):
    """And identified at the right place: the likelihood peaks at the truth.

    A coarse profile is enough -- this package holds SciPy to one call site, and
    the question is whether the parameter is estimable at all, not to how many
    figures.
    """
    likelihood = MarkedLogLikelihood(model)
    grid = np.linspace(1.0, 2.4, 57)
    values = []
    for rate in grid:
        theta = TRUTH.copy()
        theta[4] = rate
        values.append(likelihood.total(theta, history))

    peak = float(grid[int(np.argmax(values))])
    assert peak == pytest.approx(float(TRUTH[4]), abs=0.15)


def test_the_temporal_likelihood_refuses_a_marked_model(model):
    """The silent hole, and it is a subtle one.

    Unlike the multivariate case the intensity term would come out *right* --
    `_bind_history` puts the marks in the record and the hook reads them. Only
    the mark density would go missing, and with it every constraint on `b_value`
    except the support boundary.
    """
    with pytest.raises(ValueError, match="silently drop the mark density"):
        TemporalLogLikelihood(model)


def test_a_marked_likelihood_refuses_other_models():
    with pytest.raises(ValueError, match="is for marked models"):
        MarkedLogLikelihood(exponential_model())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "carrying event marks"),
        ({"points": np.zeros((1, 3))}, "temporal only"),
        ({"types": np.array([0, 1, 0]), "n_types": 2}, "single-type"),
    ],
)
def test_a_history_it_cannot_use_is_refused(model, kwargs, match):
    times = np.array([0.4, 1.1, 2.9])
    fields = {"points": None, "types": None, "n_types": None, "marks": None}
    fields.update(kwargs)
    if match != "carrying event marks":
        fields["marks"] = np.array([1.0, 2.0, 3.0])

    bad = History(
        times, fields["points"], 0.0, 4.0, fields["types"], fields["n_types"], fields["marks"]
    )
    with pytest.raises(ValueError, match=match):
        MarkedLogLikelihood(model).total(TRUTH, bad)


@pytest.mark.parametrize("blocks", [1, 2, 3, 7])
def test_extending_in_blocks_equals_one_shot(model, history, blocks):
    """Including the mark term, which is summed per block and must not double."""
    likelihood = MarkedLogLikelihood(model)
    state = likelihood.initial_state(history.start)
    for upto in np.linspace(history.start, history.end, blocks + 1)[1:]:
        state, _ = likelihood.extend(state, TRUTH, history, float(upto))

    assert state.log_lik == pytest.approx(likelihood.total(TRUTH, history), rel=1e-7)
    assert state.n_events == history.n_events


def test_a_mark_below_the_threshold_is_impossible(model):
    """Density zero there, so a likelihood of zero rather than an error."""
    likelihood = MarkedLogLikelihood(marked_model(m0=1.0))
    history = History(
        np.array([0.4, 1.1, 2.9]), None, 0.0, 4.0, None, None, np.array([2.0, 0.5, 3.0])
    )
    assert likelihood.total(TRUTH, history) == -math.inf


def test_the_compensator_ignores_the_marks(model, history):
    """It is the rate of the ground process; the mark density is not a rate."""
    likelihood = MarkedLogLikelihood(model)
    values = likelihood.compensator(TRUTH, history, history.times)
    assert np.all(np.diff(values) >= 0.0)

    other = TRUTH.copy()
    other[4] = 2.5  # a different mark law entirely
    np.testing.assert_allclose(
        likelihood.compensator(other, history, history.times), values, rtol=1e-12
    )


def test_the_marked_record_round_trips(model, history):
    rebuilt = history.as_process_events()
    assert rebuilt.shape == (2, history.n_events)
    again = History.from_marked_events(rebuilt, end=history.end)
    np.testing.assert_array_equal(again.times, history.times)
    np.testing.assert_array_equal(again.marks, history.marks)


def test_from_marked_events_needs_two_rows():
    with pytest.raises(ValueError, match="2 rows"):
        History.from_marked_events(np.zeros((3, 4)), end=1.0)
