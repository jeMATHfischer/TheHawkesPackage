"""Processes with a seeded record, rather than a simulated one.

Simulating even ten events on a Klein bottle takes seconds, and simulating on
the projective plane takes longer still -- the cost is `domain.distance`, at
142 us and 327 us a call. None of these tests is about the simulator, so the
record is written directly through the `events` setter, which is the sanctioned
way to seed a realisation and is exactly what `inference._bind_history` does.

That also makes every assertion independent of the RNG, so `pytest-randomly`
reshuffling the suite cannot change what is being measured.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.spatio_temporal import FundamentalDomain, Sphere, Torus2D

WIDTH, HEIGHT = 3.0, 5.0


def seeded_process(domain, *, n_events=6, seed=1, base=None, spatial=None, horizon=6.0):
    """A process on `domain` carrying `n_events` events, without simulating any."""
    process = hp.SpatioTemporalHawkesProcess(
        base=base if base is not None else (lambda x: 0.3),
        spatial=spatial if spatial is not None else (lambda d: 0.8 * np.exp(-2.0 * d)),
        temporal=lambda s: 1.5 * np.exp(-1.0 * s),
        domain=domain,
        monotone_temporal_kernel=True,
        rng=0,
    )
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, horizon, n_events))
    # Reshaped explicitly: an empty list comprehension gives shape (0,), not
    # (0, ndim), and the record setter rejects the result for having two rows.
    ndim = domain.bounds.shape[0]
    points = np.array([domain.sample_uniform(rng) for _ in range(n_events)], dtype=float).reshape(
        n_events, ndim
    )
    process.events = np.vstack([times[None, :], points.T])
    process.n_simulated = n_events
    return process


@pytest.fixture
def torus():
    return seeded_process(Torus2D(WIDTH, HEIGHT))


@pytest.fixture
def sphere():
    return seeded_process(Sphere())


@pytest.fixture
def bottle():
    return seeded_process(FundamentalDomain.klein_bottle(WIDTH, HEIGHT), n_events=5)


@pytest.fixture
def projective():
    return seeded_process(FundamentalDomain.projective_plane(), n_events=5)


@pytest.fixture
def frame_times():
    """Deliberately not aligned to any event time; the boundary gets its own test."""
    return np.linspace(0.5, 6.5, 5)
