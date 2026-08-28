"""Fixtures for the inference unit suite.

Local rather than in the root ``conftest.py`` on purpose: these are only wanted
by one directory, and the root conftest is imported for every collection in the
project.

The `history` fixture is **injected**, not simulated. A unit test that begins by
simulating is testing the simulator too, so a failure in it points at two
subsystems at once -- and it costs a second per test to say so.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstantBase,
    ConstrainedPrior,
    ExponentialKernel,
    GaussianSpatial,
    History,
    IndependentPrior,
    LogNormal,
    exponential_model,
    spatio_temporal_model,
)

#: A fixed set of event times, irregular enough that no accidental symmetry can
#: make a wrong compensator look right, and short enough to keep a full
#: likelihood evaluation free.
TIMES = np.array([0.31, 0.47, 1.05, 1.09, 2.30, 2.88, 3.02, 4.51, 5.60, 5.93, 7.15, 8.02, 9.44])


@pytest.fixture
def history():
    """A temporal history observed on ``[0, 10]``, with an empty tail."""
    return History(TIMES, None, 0.0, end=10.0)


@pytest.fixture
def spatial_history():
    """A one-dimensional spatio-temporal history on a circle."""
    rng = np.random.default_rng(4)
    points = rng.uniform(-np.pi, np.pi, size=(1, TIMES.size))
    return History(TIMES, points, 0.0, end=10.0)


@pytest.fixture
def exp_model():
    """The linear exponential-kernel model, ``(mu, alpha, beta)``."""
    return exponential_model()


@pytest.fixture
def exp_prior(exp_model):
    """A vague log-normal prior, truncated to the model's support."""
    base = IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0)))
    return ConstrainedPrior(base, exp_model.support)


@pytest.fixture
def st_model_1d():
    """A spatio-temporal model on the unit circle, cheap enough for a unit test."""
    return spatio_temporal_model(
        hp.Circle(),
        base=ConstantBase(),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=32,
    )


@pytest.fixture
def st_theta():
    """A parameter vector inside `st_model_1d`'s support."""
    return np.array([0.5, 0.6, 1.5, 0.6])


@pytest.fixture
def theta():
    """A parameter vector inside `exp_model`'s support."""
    return np.array([2.0, 0.5, 1.0])
