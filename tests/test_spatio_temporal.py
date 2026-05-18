"""
Tests for the legacy Spatio_Temporal_Hawkes_Process and the new
SpatioTemporalHawkesProcess with domain support.

Regression test: duplicate spatial coordinates must not raise ValueError.
"""

import numpy as np
import pytest
import TheHawkesPackage as THP
from TheHawkesPackage.spatio_temporal.domains import Circle


def _make_legacy_process():
    def base(x): return 0.5
    def spatial(space):
        b = np.pi
        Ind = ((space + b/2) >= 0) & ((space + b/2) <= b)
        return (504/(5*np.pi**4)*space**4 - 146/(5*np.pi**2)*space**2 + 1) * Ind
    def temporal(dt):
        a, b = 0.9, 2.0
        InRise = (dt < b/2) & (dt > 0)
        InDecay = (dt >= b/2) & (dt < b)
        return 2*a/b * dt * InRise + ((-2*a/b)*dt + 2*a) * InDecay
    return THP.Spatio_Temporal_Hawkes_Process(base, spatial, temporal)


def test_legacy_process_smoke(tmp_path):
    """Basic smoke test: simulate a small number of events without error."""
    np.random.seed(0)
    G = _make_legacy_process()
    G.simulate(5)
    assert G.Events.shape[1] == 5


def test_legacy_no_crash_with_duplicate_spatial():
    """Regression: injecting two events at identical spatial coords must not crash."""
    np.random.seed(0)
    G = _make_legacy_process()
    G.simulate(3)
    # Manually inject a duplicate spatial location
    dup_time = G.Events[0, -1] + 0.1
    dup_space = G.Events[1, 0]  # same x as first event
    G.Events = np.append(G.Events, np.array([[dup_time], [dup_space]]), axis=1)
    # Simulate one more event — must not raise ValueError
    G.simulate(1)


def test_legacy_propagate_alias():
    np.random.seed(1)
    G = _make_legacy_process()
    G.propagate_by_amount(3)  # fixed name
    assert G.Events.shape[1] == 3


def test_legacy_deprecated_alias():
    np.random.seed(2)
    G = _make_legacy_process()
    G.propogate_by_amount(3)  # deprecated alias must still work
    assert G.Events.shape[1] == 3


def test_new_process_circle_smoke():
    np.random.seed(3)

    def base(x): return 0.5
    def spatial(d): return max(0.0, 1 - d / np.pi)
    def temporal(dt): return 0.9 * np.exp(-5 * dt)

    P = THP.SpatioTemporalHawkesProcess(base, spatial, temporal,
                                         domain=Circle(),
                                         monotone_temporal_kernel=True)
    P.simulate(5)
    assert P.Events.shape[0] == 2  # time + 1 spatial coord
    assert P.Events.shape[1] == 5
