"""Posterior predictive simulation.

Two things here can go wrong without failing. The forecast can start from the
last event rather than from the end of the observation window, which makes the
first predicted event arrive too early every time; and the observed history can
be mutated by the paths that condition on it, which corrupts every later block
of an online fit.
"""

import numpy as np
import pytest

from hawkes_package.inference import (
    History,
    ParticleCloud,
    posterior_predictive,
    predictive_counts,
    predictive_interval,
)


@pytest.fixture
def cloud(exp_model, rng):
    """A tight cloud around a known parameter, so the paths are predictable."""
    theta = np.tile(np.array([2.0, 0.5, 1.0]), (24, 1)) * rng.lognormal(0.0, 0.05, size=(24, 3))
    return ParticleCloud(theta, np.full(24, -np.log(24)), exp_model.spec)


def test_paths_lie_strictly_inside_the_forecast_window(exp_model, cloud, history):
    paths = posterior_predictive(exp_model, cloud, history, horizon=6.0, n_paths=20, rng=0)
    assert len(paths) == 20
    for path in paths:
        assert np.all(path > history.end)
        assert np.all(path <= history.end + 6.0)
        assert np.all(np.diff(path) > 0)


def test_the_forecast_starts_at_the_window_not_the_last_event(exp_model, cloud):
    """The gap between them is data: it says nothing happened there.

    Continuing from the last event replays that gap as unobserved, so the first
    predicted event arrives too early -- systematically, not on average.
    """
    times = np.array([1.0, 2.0, 3.0])
    history = History(times, None, 0.0, end=20.0)
    paths = posterior_predictive(exp_model, cloud, history, horizon=5.0, n_paths=30, rng=1)
    assert all(path.size == 0 or path.min() > 20.0 for path in paths)


def test_the_observed_history_is_not_mutated(exp_model, cloud, history):
    before_times = history.times.copy()
    before_count = history.n_events
    posterior_predictive(exp_model, cloud, history, horizon=8.0, n_paths=10, rng=2)
    np.testing.assert_array_equal(history.times, before_times)
    assert history.n_events == before_count


def test_an_empty_path_is_a_legitimate_outcome(exp_model):
    """The outcome a fixed-count simulation cannot produce."""
    quiet = np.array([[0.02, 0.001, 1.0]] * 8)
    cloud = ParticleCloud(quiet, np.full(8, -np.log(8)), exp_model.spec)
    history = History(np.array([1.0]), None, 0.0, end=2.0)
    paths = posterior_predictive(exp_model, cloud, history, horizon=1.0, n_paths=20, rng=3)
    assert any(path.size == 0 for path in paths)


def test_counts_agree_with_the_paths(exp_model, cloud, history):
    paths = posterior_predictive(exp_model, cloud, history, horizon=5.0, n_paths=12, rng=4)
    counts = predictive_counts(paths)
    assert counts.shape == (12,)
    np.testing.assert_array_equal(counts, [path.size for path in paths])


def test_more_excitation_forecasts_more_events(exp_model, history):
    """A sanity check with a direction: the forecast must respond to the parameter."""
    calm = ParticleCloud(np.array([[1.0, 0.1, 2.0]] * 8), np.full(8, -np.log(8)), exp_model.spec)
    busy = ParticleCloud(np.array([[1.0, 1.8, 2.0]] * 8), np.full(8, -np.log(8)), exp_model.spec)
    calm_counts = predictive_counts(
        posterior_predictive(exp_model, calm, history, horizon=10.0, n_paths=40, rng=5)
    )
    busy_counts = predictive_counts(
        posterior_predictive(exp_model, busy, history, horizon=10.0, n_paths=40, rng=5)
    )
    assert busy_counts.mean() > 1.5 * calm_counts.mean()


def test_particles_are_drawn_in_proportion_to_their_weights(exp_model, history):
    """Otherwise the predictive is a plug-in one wearing a posterior's clothes."""
    theta = np.array([[0.02, 0.001, 1.0], [8.0, 0.1, 1.0]])
    heavy_on_quiet = ParticleCloud(theta, np.log(np.array([0.99, 0.01])), exp_model.spec)
    counts = predictive_counts(
        posterior_predictive(exp_model, heavy_on_quiet, history, horizon=5.0, n_paths=60, rng=6)
    )
    assert counts.mean() < 2.0, "the busy particle carries 1% of the weight"


def test_the_predictive_band_is_ordered_and_monotone(exp_model, cloud, history):
    paths = posterior_predictive(exp_model, cloud, history, horizon=10.0, n_paths=40, rng=7)
    grid = np.linspace(history.end, history.end + 10.0, 25)
    low, mid, high = predictive_interval(paths, grid)
    assert np.all(low <= mid)
    assert np.all(mid <= high)
    # A running count cannot decrease.
    for band in (low, mid, high):
        assert np.all(np.diff(band) >= 0)
    assert low[0] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0.0}, "horizon must be positive"),
        ({"horizon": 1.0, "n_paths": 0}, "at least 1"),
    ],
)
def test_bad_forecast_arguments_are_refused(exp_model, cloud, history, kwargs, match):
    with pytest.raises(ValueError, match=match):
        posterior_predictive(exp_model, cloud, history, rng=0, **kwargs)


def test_a_bad_band_level_is_refused(exp_model, cloud, history):
    paths = posterior_predictive(exp_model, cloud, history, horizon=2.0, n_paths=4, rng=0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        predictive_interval(paths, [history.end + 1.0], level=1.0)


@pytest.mark.slow
def test_spatio_temporal_paths_carry_locations(spatial_history, st_model_1d, st_theta):
    cloud = ParticleCloud(np.tile(st_theta, (6, 1)), np.full(6, -np.log(6)), st_model_1d.spec)
    paths = posterior_predictive(st_model_1d, cloud, spatial_history, horizon=2.0, n_paths=3, rng=0)
    for path in paths:
        assert path.shape[0] == 2, "one row of times above one of coordinates"
        if path.shape[1]:
            assert np.all(path[0] > spatial_history.end)
            assert np.all(np.abs(path[1]) <= np.pi + 1e-9)
    np.testing.assert_array_equal(predictive_counts(paths), [p.shape[1] for p in paths])
