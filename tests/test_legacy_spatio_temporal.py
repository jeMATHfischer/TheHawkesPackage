"""Tests for the frozen LegacySpatioTemporalHawkesProcess.

This class is kept only so results published with it stay reproducible; it is
removed in 0.4.0. These tests pin the behaviour that must not drift until then,
plus the three bugs fixed in 0.2.0.
"""

import subprocess
import sys

import numpy as np
import pytest

from hawkes_package import LegacySpatioTemporalHawkesProcess


@pytest.fixture
def legacy(legacy_kernels):
    base, spatial, temporal = legacy_kernels

    def _make(**kw):
        kw.setdefault("rng", 0)
        return LegacySpatioTemporalHawkesProcess(base, spatial, temporal, **kw)

    return _make


def test_smoke(legacy):
    p = legacy()
    p.simulate(5)
    assert p.Events.shape == (2, 5)
    assert np.all(np.diff(p.Events[0, :]) > 0)


def test_no_crash_with_duplicate_spatial_coordinates(legacy):
    """Regression: two events at identical coordinates must not raise."""
    p = legacy()
    p.simulate(3)
    dup_time = p.Events[0, -1] + 0.1
    dup_space = p.Events[1, 0]
    p.Events = np.append(p.Events, np.array([[dup_time], [dup_space]]), axis=1)
    p.simulate(1)  # must not raise
    assert p.Events.shape[1] == 5


def test_coordinates_stay_in_the_default_interval(legacy):
    p = legacy()
    p.simulate(5)
    assert np.all(p.Events[1, :] >= -np.pi - 1e-9)
    assert np.all(p.Events[1, :] <= np.pi + 1e-9)


def test_custom_space_is_honoured(legacy):
    """Regression: before 0.2.0 the spatial sampler hard-coded [-pi, pi].

    A non-default ``space`` was accepted and then silently ignored, so events
    landed outside the interval the caller asked for.
    """
    p = legacy(space=(-1.0, 1.0))
    assert p.space == (-1.0, 1.0)
    p.simulate(5)
    assert np.all(p.Events[1, :] >= -1.0 - 1e-9)
    assert np.all(p.Events[1, :] <= 1.0 + 1e-9)


def test_integration_uses_the_custom_space(legacy):
    """The space-integrated intensity must span the requested interval."""
    narrow = legacy(space=(-1.0, 1.0))
    wide = legacy(space=(-np.pi, np.pi))
    # base is the constant 0.5, so the integral is 0.5 * interval width
    assert narrow._integrated_intensity(1.0) == pytest.approx(0.5 * 2.0, rel=1e-6)
    assert wide._integrated_intensity(1.0) == pytest.approx(0.5 * 2 * np.pi, rel=1e-6)


def test_reproducible_with_seed(legacy):
    a, b = legacy(rng=5), legacy(rng=5)
    a.simulate(4)
    b.simulate(4)
    np.testing.assert_array_equal(a.Events, b.Events)


def test_intensity_matches_the_hand_rolled_formula(legacy, legacy_kernels):
    """The new accessor must reproduce what the old notebook computed by hand.

    Before 0.2.0 this class exposed no intensity accessor at all, so the
    notebook re-implemented the field intensity inline. That formula is the
    reference here.
    """
    base, spatial, temporal = legacy_kernels
    p = legacy()
    p.Events = np.array([[1.0, 2.0], [0.5, -1.0]])

    def periodize(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    t, x = 3.0, 0.2
    expected = base(x)
    for time, coord in [(1.0, 0.5), (2.0, -1.0)]:
        expected += float(temporal(t - time)) * float(spatial(periodize(x - coord)))

    assert p.intensity(t, x) == pytest.approx(max(0.0, expected))


def test_intensity_over_interval_orientation(legacy):
    p = legacy()
    p.simulate(3)
    times, points, intensity = p.intensity_over_interval(np.linspace(0, 2, 6))
    assert points.shape == (200,)
    assert intensity.shape == (points.shape[0], times.shape[0])
    assert np.all(intensity >= 0)
    assert np.isin(p.Events[0, :], times).all()


def test_import_has_no_random_seed_side_effect():
    """Regression: the module used to call random.seed(42) at import time.

    That silently reseeded the *caller's* global `random` module as a side
    effect of an import, so a user's unrelated random stream changed.
    """
    script = (
        "import random; random.seed(0); before = random.random();"
        "import hawkes_package.spatio_temporal.legacy;"
        "random.seed(0); after = random.random();"
        "print(before == after)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"


def test_import_does_not_disturb_the_global_random_stream():
    """Stronger form: importing must not consume draws from `random` either."""
    script = (
        "import random; random.seed(7); expected = [random.random() for _ in range(3)];"
        "random.seed(7);"
        "import hawkes_package.spatio_temporal.legacy;"
        "got = [random.random() for _ in range(3)];"
        "print(expected == got)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"
