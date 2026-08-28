"""`HawkesEstimator` against the real scikit-learn, rather than against our reading of it.

`HawkesEstimator` reimplements ``get_params``/``set_params`` instead of inheriting
:class:`sklearn.base.BaseEstimator`, for the reasons its module docstring gives.
Reimplementing a protocol means it can drift from the thing it copies, and the
drift would be silent: a ``get_params`` that quietly stopped listing a
keyword-only parameter would still look fine, and ``clone`` would hand back an
estimator configured differently from the one it was given.

So the linchpin here is `test_get_params_matches_base_estimators_own_implementation`,
which runs scikit-learn's implementation and ours over the same ``__init__`` and
compares. Everything else checks that the duck typing actually carries.

scikit-learn is a test-only dependency, so every test in this file skips without
it -- which is itself part of the claim being made.
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import BaseEstimator, clone  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from hawkes_package.inference import HawkesEstimator  # noqa: E402

PARTICLES = 16


@pytest.fixture
def estimator(exp_model, exp_prior):
    """An unfitted estimator, with two parameters moved off their defaults."""
    return HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, blocks=2, rng=0)


def test_get_params_matches_base_estimators_own_implementation(exp_model, exp_prior):
    """The anti-drift test.

    ``_Ref`` resolves ``get_params`` to scikit-learn's and ``_get_param_names`` to
    `HawkesEstimator.__init__`, so any divergence in which parameters we list, how
    we sort them, or how we recurse fails right here.
    """

    class _Ref(BaseEstimator, HawkesEstimator):
        pass

    theirs = _Ref(exp_model, exp_prior, n_particles=PARTICLES, rng=0).get_params()
    ours = HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, rng=0).get_params()

    assert list(theirs) == list(ours), "the parameter list or its order has drifted"
    for name, value in theirs.items():
        assert ours[name] is value


def test_clone_round_trips(estimator, history):
    """A clone is the same configuration, unfitted -- and it fits to the same numbers.

    Identity is the wrong assertion here: `clone` calls ``clone(param, safe=False)``
    on each value, which deep-copies anything without a ``get_params``, so the model
    and prior on the copy are equal-but-distinct objects. What has to survive is the
    *configuration*, and the sharpest statement of that is that both fit to the same
    cloud bit-for-bit.
    """
    copy = clone(estimator)
    assert type(copy) is HawkesEstimator
    assert list(copy.get_params()) == list(estimator.get_params())
    for name in ("n_particles", "blocks", "rng", "on_invalid", "n_move"):
        assert getattr(copy, name) == getattr(estimator, name)

    np.testing.assert_array_equal(
        copy.fit(history).cloud_.theta, estimator.fit(history).cloud_.theta
    )


def test_clone_does_not_carry_the_fit(estimator, history):
    estimator.fit(history)
    copy = clone(estimator)

    assert estimator.__sklearn_is_fitted__()
    assert not copy.__sklearn_is_fitted__()
    assert not hasattr(copy, "sampler_")


def test_check_is_fitted_agrees_with_the_estimator(estimator, history):
    with pytest.raises(sklearn.exceptions.NotFittedError):
        check_is_fitted(estimator)

    estimator.fit(history)
    check_is_fitted(estimator)


def test_set_params_drives_a_sweep_the_way_a_grid_search_would(estimator, history):
    """`GridSearchCV` cannot be used here, so this is the sanctioned loop instead.

    Not for a technical reason: a point-process history cannot be split into folds
    when every fold's likelihood depends on the events before it. Cloning,
    setting and fitting chronologically is what replaces it.
    """
    cut = float(history.times[7])
    early, late = history.upto(cut), history.times[history.times > cut]

    scores = {}
    for particles in (8, 16):
        candidate = clone(estimator).set_params(n_particles=particles)
        candidate.fit(early)
        scores[particles] = candidate.score(late, end=history.end)

    assert set(scores) == {8, 16}
    assert all(np.isfinite(value) for value in scores.values())


@pytest.mark.skipif(sklearn.__version__ < "1.6", reason="__sklearn_tags__ is the 1.6+ tag protocol")
def test_sklearn_tags_are_available_without_inheriting_them(estimator):
    from sklearn.utils import get_tags

    tags = get_tags(estimator)
    assert tags.target_tags.required is False


def test_the_estimator_is_not_a_base_estimator_subclass(estimator):
    """Pins the decision, not just the current state.

    Inheriting `BaseEstimator` when it happens to be installed would make `repr`,
    parameter ordering and pickling depend on the environment rather than on the
    code -- and it is refused by ``mypy --strict`` besides, since scikit-learn
    ships no ``py.typed`` and ``disallow_subclassing_any`` applies. A caller who
    genuinely needs the ``isinstance`` subclasses it themselves, in two lines.
    """
    assert not isinstance(estimator, BaseEstimator)
    assert hasattr(estimator, "get_params")
    assert hasattr(estimator, "set_params")
