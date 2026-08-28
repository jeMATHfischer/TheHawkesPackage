"""The scikit-learn-shaped wrapper, and the two ways it could quietly stop being one.

`HawkesEstimator` infers nothing of its own, so almost everything here is an
*equality* rather than a threshold: a fit is `fit_smc` bit-for-bit, a `partial_fit`
per block is `fit(blocks=k)` bit-for-bit, and a `score` is the log-evidence
increment the next `partial_fit` records. If any of the three drifts, something is
being computed twice, and the second copy is the one that will be wrong.

Two failures the rest of the file exists to catch, neither of which raises:

* **A `predict` that plugs the posterior mean into the intensity** instead of
  averaging the intensity over the particles. It returns a plausible curve that is
  biased low wherever the posterior has width, by Jensen on a convex `beta`.
* **A `partial_fit` that re-reads events it has already absorbed.** Counting a
  block twice tightens the posterior around whatever it already believed, which
  reads as convergence.
"""

import subprocess
import sys

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstantBase,
    ConstrainedPrior,
    ExponentialKernel,
    ExponentialLogLikelihood,
    GaussianSpatial,
    HawkesEstimator,
    IndependentPrior,
    LogNormal,
    ParticleCloud,
    SpatioTemporalLogLikelihood,
    TemporalLogLikelihood,
    bell_shape_model,
    block_boundaries,
    exponential_model,
    fit_smc,
    monotone_model,
    spatio_temporal_model,
)
from hawkes_package.inference.likelihood import _bind_history

#: Small enough that a whole file of fits stays inside a second or two. The
#: equalities asserted here are exact, so cloud size changes nothing about
#: whether they hold.
PARTICLES = 32


@pytest.fixture
def estimator(exp_model, exp_prior):
    """An unfitted estimator on the exponential model."""
    return HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, blocks=2, rng=0)


@pytest.fixture
def fitted(estimator, history):
    """The same estimator, fitted to the injected history."""
    return estimator.fit(history)


@pytest.fixture
def st_prior(st_model_1d):
    """A prior for the spatio-temporal fixture model, truncated to its support."""
    base = IndependentPrior(
        (LogNormal(-1.0, 0.5), LogNormal(-1.0, 0.5), LogNormal(0.5, 0.5), LogNormal(-0.5, 0.5))
    )
    return ConstrainedPrior(base, st_model_1d.support)


# ---------------------------------------------------------------------------
# The three equalities. These are what make this a wrapper.
# ---------------------------------------------------------------------------


def test_the_estimator_reproduces_fit_smc_exactly(history, exp_model, exp_prior):
    """The core test: same seed, same blocks, same numbers to the last bit."""
    reference = fit_smc(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        history,
        blocks=2,
        n_particles=PARTICLES,
        rng=0,
    )
    estimator = HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, blocks=2, rng=0).fit(
        history
    )

    np.testing.assert_array_equal(estimator.cloud_.theta, reference.cloud.theta)
    np.testing.assert_array_equal(estimator.cloud_.log_weights, reference.cloud.log_weights)


def test_partial_fit_over_blocks_matches_one_fit(history, exp_model, exp_prior):
    """Blocking is a matter of when answers arrive, not of what answer arrives."""
    batch = HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, blocks=4, rng=0).fit(
        history
    )

    online = HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, rng=0)
    previous = history.start
    for upto in block_boundaries(history, 4):
        block = history.times[(history.times > previous) & (history.times <= upto)]
        online.partial_fit(block, end=float(upto))
        previous = upto

    np.testing.assert_array_equal(online.cloud_.theta, batch.cloud_.theta)
    np.testing.assert_array_equal(online.cloud_.log_weights, batch.cloud_.log_weights)
    assert online.n_events_ == batch.n_events_


def test_score_equals_the_next_blocks_log_evidence_increment(history, exp_model, exp_prior):
    """`score` and the evidence increment are the same quantity, so they agree exactly."""
    cut = float(history.times[7])
    estimator = HawkesEstimator(exp_model, exp_prior, n_particles=PARTICLES, blocks=2, rng=0).fit(
        history.upto(cut)
    )

    ahead = history.times[history.times > cut]
    scored = estimator.score(ahead, end=history.end)
    estimator.partial_fit(ahead, end=history.end)

    assert scored == estimator.diagnostics_.steps[-1].log_evidence_increment


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_marginalises_rather_than_plugging_in(fitted, exp_model, history):
    """The decision this file exists to pin.

    Two particles with well-separated decay rates, and a grid where the
    difference shows. The marginal is the weighted mean of the two curves; the
    plug-in at the mean parameter sits strictly *below* it at every point,
    because the intensity is convex in `beta`. A `predict` that quietly became a
    plug-in would still return a smooth, plausible curve.
    """
    theta = np.array([[2.0, 0.5, 0.6], [2.0, 0.5, 4.0]])
    fitted.sampler_.cloud = ParticleCloud(theta, np.log([0.5, 0.5]), exp_model.spec)
    grid = np.array([1.5, 3.0, 6.0])

    curves = []
    for row in theta:
        process = exp_model(row)
        _bind_history(process, history)
        curves.append([process._conditional_intensity(float(t)) for t in grid])
    expected = 0.5 * (np.array(curves[0]) + np.array(curves[1]))

    plug_in = exp_model(theta.mean(axis=0))
    _bind_history(plug_in, history)
    plugged = np.array([plug_in._conditional_intensity(float(t)) for t in grid])

    np.testing.assert_array_equal(fitted.predict(grid), expected)
    assert np.all(plugged < expected), "the plug-in must differ, or this test proves nothing"


def test_predict_at_an_event_time_is_the_left_limit(fitted, history):
    """The intensity hook excludes events *at* `t`, which is the likelihood's convention."""
    event = float(history.times[4])
    assert fitted.predict([event])[0] < fitted.predict([event + 1e-9])[0]


def test_predict_returns_one_value_per_requested_time(fitted, history):
    """Guards against reaching for `intensity_over_interval`, which merges the grid."""
    assert fitted.predict(history.times).shape == history.times.shape


def test_predict_accepts_a_scikit_learn_column(fitted):
    """An ``(n, 1)`` column is ravelled, not read as one event in n-1 dimensions."""
    grid = np.array([1.5, 3.0, 6.0])
    np.testing.assert_array_equal(fitted.predict(grid.reshape(-1, 1)), fitted.predict(grid))


def test_predict_refuses_a_two_dimensional_shape_it_cannot_read(fitted):
    with pytest.raises(ValueError, match=r"shape \(n,\) or \(n, 1\)"):
        fitted.predict(np.zeros((2, 3)))


@pytest.mark.parametrize("outside", [[10.5], [-0.1], [1.0, 12.0]])
def test_predict_refuses_times_outside_the_observation_window(fitted, outside):
    """Past the window the intensity is biased low and nothing says so."""
    with pytest.raises(ValueError, match="observation window"):
        fitted.predict(outside)


def test_predict_admits_both_ends_of_the_window(fitted, history):
    values = fitted.predict([history.start, history.end])
    assert np.all(values > 0.0)


def test_the_intensity_band_brackets_the_mean(fitted):
    grid = np.array([1.5, 3.0, 6.0])
    lower, median, upper = fitted.predict_intensity_band(grid)
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    assert lower.shape == median.shape == upper.shape == grid.shape


def test_the_intensity_band_refuses_an_impossible_level(fitted):
    with pytest.raises(ValueError, match="level must lie"):
        fitted.predict_intensity_band([1.0], level=1.0)


# ---------------------------------------------------------------------------
# fit and partial_fit contracts
# ---------------------------------------------------------------------------


def test_fit_refuses_an_array_without_a_window(estimator, history):
    with pytest.raises(ValueError, match="needs the observation window"):
        estimator.fit(history.times)


def test_fit_refuses_both_a_history_and_a_window(estimator, history):
    with pytest.raises(ValueError, match="History and an explicit window"):
        estimator.fit(history, end=history.end)


def test_fit_accepts_an_array_with_an_explicit_window(estimator, history):
    estimator.fit(history.times, end=history.end)
    np.testing.assert_array_equal(estimator.history_.times, history.times)
    assert estimator.history_.end == history.end


def test_fit_rebuilds_the_sampler_so_a_second_fit_is_not_a_continuation(estimator, history):
    """A likelihood cannot run backwards, so a refit starts over rather than resuming."""
    first = estimator.fit(history).cloud_.theta.copy()
    second = estimator.fit(history).cloud_.theta

    np.testing.assert_array_equal(first, second)
    assert estimator.n_events_ == history.n_events


def test_fit_after_partial_fit_discards_the_online_state(estimator, history):
    estimator.partial_fit(history.times[:4], end=float(history.times[4]))
    estimator.fit(history)
    assert estimator.n_events_ == history.n_events
    assert len(estimator.diagnostics_.steps) == 2


@pytest.mark.parametrize("target", [0, np.zeros(3), "y"])
def test_a_target_is_refused_rather_than_ignored(estimator, history, target):
    with pytest.raises(ValueError, match="no target"):
        estimator.fit(history, y=target)


def test_partial_fit_refuses_events_already_seen(estimator, history):
    cut = float(history.times[5])
    estimator.partial_fit(history.times[history.times <= cut], end=cut)
    with pytest.raises(ValueError, match="only the events in"):
        estimator.partial_fit(history.times, end=history.end)


def test_partial_fit_refuses_a_window_that_does_not_advance(estimator, history):
    cut = float(history.times[5])
    estimator.partial_fit(history.times[history.times <= cut], end=cut)
    with pytest.raises(ValueError, match="does not advance"):
        estimator.partial_fit([], end=cut)


def test_partial_fit_refuses_a_history_object(estimator, history):
    with pytest.raises(ValueError, match="not a History"):
        estimator.partial_fit(history, end=history.end)


def test_partial_fit_accepts_an_empty_block(estimator, history):
    """An empty block is data: it says nothing happened."""
    estimator.partial_fit(history.times, end=history.end)
    before = estimator.n_events_
    estimator.partial_fit([], end=history.end + 5.0)

    assert estimator.n_events_ == before
    assert estimator.history_.end == history.end + 5.0
    assert len(estimator.diagnostics_.steps) == 2


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_does_not_advance_the_fit(fitted, history):
    events = fitted.n_events_
    steps = len(fitted.diagnostics_.steps)
    fitted.score([history.end + 1.0], end=history.end + 2.0)

    assert fitted.n_events_ == events
    assert fitted.history_.end == history.end
    assert len(fitted.diagnostics_.steps) == steps


def test_score_refuses_a_window_that_does_not_advance(fitted, history):
    with pytest.raises(ValueError, match="does not advance"):
        fitted.score([], end=history.end)


def test_score_refuses_events_inside_the_fitted_window(fitted, history):
    with pytest.raises(ValueError, match="only the events in"):
        fitted.score([history.times[0]], end=history.end + 1.0)


# ---------------------------------------------------------------------------
# forecasting and sampling
# ---------------------------------------------------------------------------


def test_forecast_starts_at_the_end_of_the_window(fitted, history):
    paths = fitted.forecast(horizon=4.0, n_paths=8, rng=0)
    assert len(paths) == 8
    for path in paths:
        assert np.all(path > history.end)
        assert np.all(path <= history.end + 4.0)


def test_predict_counts_returns_one_count_per_path(fitted):
    counts = fitted.predict_counts(horizon=4.0, n_paths=8, rng=0)
    assert counts.shape == (8,)
    assert counts.dtype.kind == "i"


def test_predict_interval_is_ordered_and_refuses_the_fitted_window(fitted, history):
    grid = history.end + np.array([1.0, 2.0, 3.0])
    lower, median, upper = fitted.predict_interval(grid, n_paths=8, rng=0)

    assert np.all(lower <= median)
    assert np.all(median <= upper)
    assert np.all(np.diff(median) >= 0), "a cumulative count cannot fall"
    with pytest.raises(ValueError, match="must exceed"):
        fitted.predict_interval([history.end], n_paths=4)


def test_sample_posterior_draws_parameter_vectors(fitted):
    draws = fitted.sample_posterior(5, rng=0)
    assert draws.shape == (5, len(fitted.parameter_names_))
    assert np.all(fitted.model_.support(draws))
    with pytest.raises(ValueError, match="at least 1"):
        fitted.sample_posterior(0)


def test_residuals_and_report_reach_the_existing_diagnostics(fitted, history):
    assert fitted.residuals().shape == (history.n_events,)
    assert "mu" in fitted.report()


# ---------------------------------------------------------------------------
# construction, parameters and the fitted surface
# ---------------------------------------------------------------------------


def test_the_constructor_stores_every_argument_unmodified(exp_model, exp_prior):
    """`clone`'s own check: it reconstructs and asserts every value is the same object.

    So a `int(n_particles)` or a `tuple(blocks)` here would break cloning rather
    than catch anything -- which is why validation lives in `fit` instead.
    """
    blocks = [3.0, 10.0]
    particles = np.int64(16)
    estimator = HawkesEstimator(exp_model, exp_prior, n_particles=particles, blocks=blocks, rng=0)

    assert estimator.n_particles is particles
    assert estimator.blocks is blocks
    assert estimator.model is exp_model
    assert estimator.prior is exp_prior


def test_get_params_round_trips_through_the_constructor(estimator):
    params = estimator.get_params()
    rebuilt = type(estimator)(**params)
    for name, value in params.items():
        assert getattr(rebuilt, name) is value


def test_get_params_returns_sorted_names_including_keyword_only(estimator):
    names = list(estimator.get_params())
    assert names == sorted(names)
    assert {"model", "prior", "n_particles", "rng", "on_invalid"} <= set(names)


def test_set_params_changes_the_fit(estimator, history):
    estimator.set_params(n_particles=8, blocks=1)
    estimator.fit(history)
    assert estimator.cloud_.n_particles == 8
    assert len(estimator.diagnostics_.steps) == 1


def test_set_params_rejects_an_unknown_name(estimator):
    with pytest.raises(ValueError, match="invalid parameter 'nope'"):
        estimator.set_params(nope=1)


def test_repr_shows_only_the_parameters_that_differ_from_their_defaults(estimator):
    text = repr(estimator)
    assert text.startswith("HawkesEstimator(")
    assert "n_particles=" in text
    assert "on_invalid=" not in text, "an untouched default should not be printed"
    assert "family='exponential'" in text, "the model repr should stay readable"


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (exponential_model, ExponentialLogLikelihood),
        (monotone_model, TemporalLogLikelihood),
        (bell_shape_model, TemporalLogLikelihood),
    ],
)
def test_the_likelihood_is_auto_selected_by_family(history, factory, expected):
    """A speed choice, not a modelling one: the two temporal paths agree numerically."""
    model = factory()
    prior = ConstrainedPrior(
        IndependentPrior(tuple(LogNormal(-0.5, 0.5) for _ in model.spec.names)),
        model.support,
    )
    estimator = HawkesEstimator(model, prior, n_particles=8, blocks=1, rng=0).fit(history)
    assert isinstance(estimator.likelihood_, expected)


def test_a_string_names_a_temporal_family(history, exp_prior):
    estimator = HawkesEstimator("exponential", exp_prior, n_particles=8, blocks=1, rng=0)
    estimator.fit(history)
    assert estimator.model_.spec.names == ("mu", "alpha", "beta")


@pytest.mark.parametrize("name", ["spatio_temporal", "nope", 3])
def test_a_string_model_refuses_anything_but_a_temporal_family(exp_prior, history, name):
    with pytest.raises(ValueError, match="model must be a ProcessModel"):
        HawkesEstimator(name, exp_prior).fit(history)


def test_an_explicit_likelihood_is_used_as_given(history, exp_model, exp_prior):
    likelihood = TemporalLogLikelihood(exp_model)
    estimator = HawkesEstimator(
        exp_model, exp_prior, likelihood=likelihood, n_particles=8, blocks=1, rng=0
    ).fit(history)
    assert estimator.likelihood_ is likelihood


def test_the_fitted_surface_reads_off_the_sampler(fitted, history, exp_model):
    assert fitted.cloud_ is fitted.sampler_.cloud
    assert fitted.diagnostics_ is fitted.sampler_.diagnostics
    assert fitted.spec_ is exp_model.spec
    assert fitted.parameter_names_ == ("mu", "alpha", "beta")
    assert fitted.n_events_ == history.n_events
    assert fitted.theta_.shape == (3,)
    assert np.isfinite(fitted.log_evidence_)


@pytest.mark.parametrize(
    "call",
    [
        lambda est: est.predict([1.0]),
        lambda est: est.score([1.0], end=2.0),
        lambda est: est.forecast(horizon=1.0),
        lambda est: est.predict_counts(horizon=1.0),
        lambda est: est.predict_interval([1.0]),
        lambda est: est.sample_posterior(),
        lambda est: est.residuals(),
        lambda est: est.report(),
        lambda est: est.cloud_,
        lambda est: est.theta_,
        lambda est: est.diagnostics_,
        lambda est: est.spec_,
        lambda est: est.n_events_,
        lambda est: est.log_evidence_,
    ],
)
def test_unfitted_methods_raise(estimator, call):
    assert not estimator.__sklearn_is_fitted__()
    with pytest.raises(RuntimeError, match="not fitted"):
        call(estimator)


def test_a_bad_hyperparameter_raises_before_the_data_is_touched(exp_model, exp_prior, history):
    """`fit` builds the sampler first, so this arrives immediately rather than late."""
    estimator = HawkesEstimator(exp_model, exp_prior, n_particles=1)
    with pytest.raises(ValueError, match="n_particles must be at least 2"):
        estimator.fit(history)


# ---------------------------------------------------------------------------
# spatio-temporal
# ---------------------------------------------------------------------------


def test_a_spatio_temporal_history_fits_scores_and_forecasts(
    spatial_history, st_model_1d, st_prior
):
    """The same class, on the path where the geometry cache is the thing that can rot."""
    estimator = HawkesEstimator(st_model_1d, st_prior, n_particles=8, blocks=2, rng=0)
    estimator.fit(spatial_history)
    assert estimator.likelihood_.backend_used == "cached"
    assert estimator.theta_.shape == (4,)

    block = np.array([[spatial_history.end + 0.4], [0.7]])
    scored = estimator.score(block, end=spatial_history.end + 1.0)
    assert np.isfinite(scored)

    estimator.partial_fit(block, end=spatial_history.end + 1.0)
    assert estimator.n_events_ == spatial_history.n_events + 1
    assert estimator.likelihood_.backend_used == "cached"


def test_a_spatio_temporal_fit_refuses_predict(spatial_history, st_model_1d, st_prior):
    estimator = HawkesEstimator(st_model_1d, st_prior, n_particles=8, blocks=1, rng=0)
    estimator.fit(spatial_history)
    with pytest.raises(ValueError, match="predict is temporal"):
        estimator.predict([1.0])


def test_a_spatio_temporal_fit_refuses_a_bare_time_array(spatial_history, st_prior):
    model = spatio_temporal_model(
        hp.Circle(),
        base=ConstantBase(),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=32,
    )
    estimator = HawkesEstimator(model, st_prior, n_particles=8, blocks=1, rng=0)
    with pytest.raises(ValueError, match=r"record of shape \(2, n\)"):
        estimator.fit(spatial_history.times, end=spatial_history.end)


def test_the_spatio_temporal_likelihood_is_auto_selected(spatial_history, st_model_1d, st_prior):
    estimator = HawkesEstimator(st_model_1d, st_prior, n_particles=8, blocks=1, rng=0)
    estimator.fit(spatial_history)
    assert isinstance(estimator.likelihood_, SpatioTemporalLogLikelihood)


# ---------------------------------------------------------------------------
# the dependency claim
# ---------------------------------------------------------------------------


def test_importing_the_estimator_does_not_import_scikit_learn():
    """In a subprocess, because an in-process check depends on what ran before it.

    The module names scikit-learn in exactly one method body. If that import ever
    migrates to the top of the file, scikit-learn becomes a runtime dependency of
    a package that promises numpy and scipy only -- and nothing else would notice.
    """
    code = "import hawkes_package.inference, sys; assert 'sklearn' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# The edges that are only reachable deliberately
# ---------------------------------------------------------------------------


def test_a_zero_weight_particle_is_skipped_rather_than_built(fitted, exp_model, history):
    """An abandoned particle can sit outside the support, where the process cannot exist.

    It carries no weight, so it contributes nothing -- but building it to discover
    that would raise from `ProcessModel.__call__`, turning a healthy cloud into a
    crash. Both `predict` and `score` step over it instead.
    """
    theta = np.array([[2.0, 0.5, 1.0], [2.0, 5.0, 1.0]])  # the second has alpha > beta
    assert not fitted.model_.support(theta)[1], "the second particle must be unbuildable"
    # Written out rather than np.log([1.0, 0.0]): filterwarnings = ["error"], and
    # log(0) is a RuntimeWarning even where -inf is exactly what is wanted.
    weights = np.array([0.0, -np.inf])
    fitted.sampler_.cloud = ParticleCloud(theta, weights, exp_model.spec)

    survivor = exp_model(theta[0])
    _bind_history(survivor, history)
    expected = survivor._conditional_intensity(3.0)

    assert fitted.predict([3.0])[0] == expected
    assert np.isfinite(fitted.score([history.end + 0.5], end=history.end + 1.0))


def test_set_params_with_nothing_to_set_is_a_no_op(estimator):
    assert estimator.set_params() is estimator


def test_predict_interval_needs_at_least_one_time(fitted):
    with pytest.raises(ValueError, match="at least one time"):
        fitted.predict_interval([])


def test_score_refuses_a_history_object(fitted, history):
    with pytest.raises(ValueError, match="not a History"):
        fitted.score(history, end=history.end + 1.0)


def test_a_spatio_temporal_fit_accepts_an_empty_block(spatial_history, st_model_1d, st_prior):
    estimator = HawkesEstimator(st_model_1d, st_prior, n_particles=8, blocks=1, rng=0)
    estimator.fit(spatial_history)
    estimator.partial_fit([], end=spatial_history.end + 2.0)

    assert estimator.n_events_ == spatial_history.n_events
    assert estimator.history_.end == spatial_history.end + 2.0
