"""Out-of-sample scoring, and the leakage it exists to make impossible.

Every other check in this subpackage is in-sample: the model is judged against
the events that chose its parameters. This one refits at a series of origins and
scores only what came after each — which is worth nothing at all if the fitter
can see the future, so that is what most of this file tests.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import ExponentialLogLikelihood, History, exponential_model
from hawkes_package.inference.validation import rolling_origin

TRUTH = np.array([2.0, 0.5, 1.0])
HORIZON = 200.0
ORIGINS = np.linspace(40.0, 160.0, 5)


@pytest.fixture
def likelihood():
    return ExponentialLogLikelihood(exponential_model())


@pytest.fixture
def history():
    process = hp.ExponentialHawkes(TRUTH, rng=0)
    process.simulate_until(HORIZON)
    return History.from_events(process.events, end=HORIZON)


def constant_fit(_prefix):
    """A fitter that ignores its input, so tests isolate the orchestration."""
    return TRUTH


def test_the_fitter_never_sees_past_its_origin(likelihood, history):
    """The one guarantee this function exists to provide.

    Not "the fitter is passed a truncated history" as a matter of code reading,
    but the arithmetic statement: no event it was shown lies after the origin it
    was called for.
    """
    shown = []

    def spy(prefix):
        shown.append((prefix.end, prefix.times.copy()))
        return TRUTH

    rolling_origin(likelihood, history, spy, origins=ORIGINS)

    assert len(shown) == len(ORIGINS)
    for (end, times), origin in zip(shown, ORIGINS, strict=True):
        assert end == pytest.approx(origin)
        assert times.size == 0 or float(times.max()) <= origin


def test_changing_the_future_leaves_earlier_scores_untouched(likelihood, history):
    """The cheapest proof there is no leakage: the past cannot depend on the future.

    Truncate the record after the third origin and the scores at the first two
    must be bit-identical, because nothing that changed was visible to them.
    """
    full = rolling_origin(likelihood, history, constant_fit, origins=ORIGINS)

    cut = float(ORIGINS[2])
    shortened = History(history.times[history.times <= cut], None, history.start, cut)
    partial = rolling_origin(likelihood, shortened, constant_fit, origins=ORIGINS[:2])

    for a, b in zip(full.scores[:2], partial.scores[:2], strict=True):
        assert a.n_train == b.n_train
        assert a.n_test == b.n_test
        assert a.log_score == b.log_score


def test_the_blocks_partition_what_follows_the_first_origin(likelihood, history):
    """No event scored twice, none dropped."""
    result = rolling_origin(likelihood, history, constant_fit, origins=ORIGINS)

    scored = sum(s.n_test for s in result.scores)
    after_first = int(np.sum(history.times > ORIGINS[0]))
    assert scored == after_first

    assert result.scores[0].origin == pytest.approx(ORIGINS[0])
    assert result.scores[-1].horizon == pytest.approx(history.end)


def test_the_training_set_grows(likelihood, history):
    """Expanding window, so later origins are fitted on strictly more data."""
    result = rolling_origin(likelihood, history, constant_fit, origins=ORIGINS)
    counts = [s.n_train for s in result.scores]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


@pytest.mark.statistical
def test_the_true_parameters_out_predict_a_constant_rate(likelihood, history):
    """At the truth, the model should be worth having out of sample.

    Not at every origin -- a block can be quiet by chance, and one of the five
    here is -- so the assertion is on the aggregate. Measured at mean skill
    +0.033 nats per event, beating the constant rate at 4 of 5 origins.
    """
    result = rolling_origin(likelihood, history, constant_fit, origins=ORIGINS)

    assert result.mean_skill > 0.0
    assert result.beat_baseline >= 3
    assert "mean skill" in result.summary()


def test_a_worse_parameter_scores_worse(likelihood, history):
    """Guard the guard: the score has to respond to the parameters.

    A skill number that ignored `theta` would pass every test above.
    """
    good = rolling_origin(likelihood, history, constant_fit, origins=ORIGINS)

    wrong = TRUTH.copy()
    wrong[1] = 0.95  # nearly explosive against a truth of 0.5
    bad = rolling_origin(likelihood, history, lambda _p: wrong, origins=ORIGINS)

    assert bad.mean_skill < good.mean_skill


@pytest.mark.parametrize(
    ("origins", "match"),
    [
        ([], "at least one cut"),
        ([50.0, 40.0], "strictly increasing"),
        ([50.0, 50.0], "strictly increasing"),
        ([0.0, 50.0], "strictly inside"),
        ([50.0, HORIZON], "strictly inside"),
    ],
)
def test_the_origins_are_validated(likelihood, history, origins, match):
    with pytest.raises(ValueError, match=match):
        rolling_origin(likelihood, history, constant_fit, origins=origins)
