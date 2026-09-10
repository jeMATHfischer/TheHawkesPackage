"""Saving a fit as the recipe that produced it, and the end-to-end seeding it rests on.

A recipe is a configuration, a seed and a reference to the data -- not the
posterior. Rerunning it is what reproduces the fit, which makes the claim worth
testing directly: **the same recipe, rerun, gives the same numbers.**

Two refusals carry as much weight as the round trip. A recipe cannot name a
`ProcessModel` object, because that object is three closures and closures cannot
be written down; and it cannot carry a marginal this module has no fields for,
because a prior silently replaced by a near one is a different model reported
under the same name.

The seeding half is an audit rather than a build. `SeedLike` already covers every
form, `default_rng` is called once per process, and `numpy.random.seed` has not
influenced a simulation since 0.2.0. What was missing is the statement end to
end: one seed through simulate, fit and predict, twice, with the global stream
disturbed in between.
"""

import json
import warnings

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior,
    Gamma,
    HawkesEstimator,
    History,
    IndependentPrior,
    LogNormal,
    Normal,
    Uniform,
    exponential_model,
    from_recipe,
    read_recipe,
    to_recipe,
    write_recipe,
)

TRUTH = np.array([1.0, 0.5, 2.0])


def make_prior(model):
    return ConstrainedPrior(
        IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )


@pytest.fixture(scope="module")
def history():
    process = hp.ExponentialHawkes(TRUTH, rng=0)
    process.simulate(200)
    return History.from_simulation(process)


@pytest.fixture
def estimator():
    model = exponential_model()
    return HawkesEstimator("exponential", make_prior(model), n_particles=64, blocks=3, rng=7)


def test_a_recipe_is_json(estimator):
    """Which is what makes it small enough to paste into an appendix."""
    recipe = to_recipe(estimator, data="events.npz")
    text = json.dumps(recipe)
    assert json.loads(text) == recipe
    assert recipe["data"] == "events.npz"
    assert recipe["estimator"]["model"] == "exponential"
    assert recipe["estimator"]["rng"] == 7


def test_the_round_trip_reproduces_the_fit(estimator, history):
    """The claim the whole module makes, asserted on the numbers.

    Exactly, because this is the temporal path: the same seed consumes the same
    draws in the same order. On the spatio-temporal path it would have to be a
    distributional statement, which is why a recipe of one is not offered.
    """
    original = estimator.fit(history)
    rebuilt = from_recipe(to_recipe(estimator)).fit(history)

    np.testing.assert_array_equal(rebuilt.theta_, original.theta_)
    np.testing.assert_array_equal(rebuilt.cloud_.theta, original.cloud_.theta)
    assert rebuilt.diagnostics_.log_evidence == original.diagnostics_.log_evidence


def test_a_recipe_survives_a_file(tmp_path, estimator, history):
    path = tmp_path / "fit.json"
    write_recipe(estimator, path, data="doi:10.5281/zenodo.0000000")
    rebuilt = read_recipe(path)

    assert rebuilt.get_params() == estimator.get_params() | {"prior": rebuilt.prior}
    np.testing.assert_array_equal(rebuilt.fit(history).theta_, estimator.fit(history).theta_)


def test_every_supported_marginal_round_trips():
    """A prior rebuilt from a recipe must be the prior, not one like it."""
    model = exponential_model()
    prior = IndependentPrior((LogNormal(0.2, 0.9), Normal(-0.5, 1.5), Gamma(2.0, 3.0)))
    estimator = HawkesEstimator("exponential", ConstrainedPrior(prior, model.support), rng=1)

    rebuilt = from_recipe(to_recipe(estimator)).prior.base
    assert [type(m).__name__ for m in rebuilt.marginals] == ["LogNormal", "Normal", "Gamma"]

    grid = np.array([[0.4, -0.2, 1.1], [1.3, 0.8, 2.2]])
    np.testing.assert_allclose(rebuilt.log_pdf(grid), prior.log_pdf(grid), rtol=0, atol=0)


def test_a_uniform_marginal_round_trips():
    model = exponential_model()
    prior = IndependentPrior((Uniform(0.1, 2.0), LogNormal(0.0, 1.0), LogNormal(0.0, 1.0)))
    estimator = HawkesEstimator("exponential", ConstrainedPrior(prior, model.support), rng=1)
    rebuilt = from_recipe(to_recipe(estimator)).prior.base
    assert type(rebuilt.marginals[0]).__name__ == "Uniform"


def test_a_model_object_is_refused_with_its_reason():
    """The structural obstacle, named rather than worked around.

    `ProcessModel` holds a builder, a branching callable and a support. It
    carries a hand-written `__repr__` precisely because those print as
    addresses, and an address is not a thing a reader can rerun.
    """
    model = exponential_model()
    estimator = HawkesEstimator(model, make_prior(model), rng=1)
    with pytest.raises(TypeError, match="closures cannot be written down"):
        to_recipe(estimator)


def test_a_generator_seed_is_refused():
    """A `Generator` carries state, and state is not a recipe."""
    model = exponential_model()
    estimator = HawkesEstimator("exponential", make_prior(model), rng=np.random.default_rng(3))
    with pytest.raises(TypeError, match="integer seed"):
        to_recipe(estimator)


def test_an_explicit_likelihood_is_refused():
    """Because the reader cannot reconstruct the object, only the choice."""
    from hawkes_package.inference import ExponentialLogLikelihood

    model = exponential_model()
    estimator = HawkesEstimator(
        "exponential", make_prior(model), likelihood=ExponentialLogLikelihood(model)
    )
    with pytest.raises(TypeError, match="cannot carry an explicit likelihood"):
        to_recipe(estimator)


def test_a_recipe_from_a_newer_release_warns(estimator):
    """A stamp that is written and never compared implies a check nobody runs."""
    recipe = to_recipe(estimator)
    recipe["hawkes_package"] = "99.0.0"
    with pytest.warns(UserWarning, match="written by hawkes_package 99.0.0"):
        from_recipe(recipe)


def test_a_recipe_from_an_older_release_does_not_warn(estimator):
    """The format is what has to stay readable, and it is versioned separately."""
    recipe = to_recipe(estimator)
    recipe["hawkes_package"] = "0.0.1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_recipe(recipe)


def test_an_unreadable_format_is_refused(estimator):
    recipe = to_recipe(estimator)
    recipe["format"] = 99
    with pytest.raises(ValueError, match="format 99"):
        from_recipe(recipe)


# ---------------------------------------------------------------------------
# The seeding audit
# ---------------------------------------------------------------------------


def test_one_seed_reproduces_a_whole_pipeline(history):
    """Simulate, fit and predict -- twice, from one seed, to the last bit."""

    def run():
        process = hp.ExponentialHawkes(TRUTH, rng=11)
        process.simulate(150)
        events = History.from_simulation(process)
        model = exponential_model()
        fitted = HawkesEstimator(
            "exponential", make_prior(model), n_particles=64, blocks=2, rng=5
        ).fit(events)
        grid = np.linspace(events.start, events.end, 25)
        return process.events, fitted.theta_, fitted.predict(grid)

    first, second = run(), run()
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_global_numpy_stream_still_does_not_matter(history):
    """`tests/test_base.py` proves this for simulation; here it is for a fit.

    The suite runs under `pytest-randomly`, which reseeds the global stream
    between tests, so any dependence on it would surface as an intermittent
    failure rather than an honest one.
    """
    model = exponential_model()

    np.random.seed(4)
    first = HawkesEstimator("exponential", make_prior(model), n_particles=64, blocks=2, rng=9).fit(
        history
    )

    np.random.seed(9999)
    second = HawkesEstimator("exponential", make_prior(model), n_particles=64, blocks=2, rng=9).fit(
        history
    )

    np.testing.assert_array_equal(first.cloud_.theta, second.cloud_.theta)


def test_a_recipe_does_not_carry_the_data(estimator):
    """The events are the user's; a recipe points at them."""
    recipe = to_recipe(estimator, data="s3://bucket/events.npz")
    assert recipe["data"] == "s3://bucket/events.npz"
    assert "times" not in json.dumps(recipe)
    assert len(json.dumps(recipe)) < 2000
