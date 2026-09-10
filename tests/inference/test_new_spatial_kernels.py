"""The new spatial families in a spatio-temporal fit, through both backends.

`SpatioTemporalLogLikelihood` computes one number two ways, and the two must
agree. How *closely* they agree is a property of the kernel rather than of the
code: both are quadrature on the same rule, so the gap between them is the
rule's own error on that shape, and it shrinks when the rule is refined.

Measured on a 13-event history on a unit `Circle`:

============  =========  ==========  ==========  ==========
family        n_quad=64  n_quad=128  n_quad=256  n_quad=512
============  =========  ==========  ==========  ==========
Gaussian      2.2e-10    2.9e-11     2.2e-11     3.9e-12
Pareto        2.3e-07    2.1e-08     1.5e-08     2.7e-09
compact       3.1e-05    1.4e-06     1.6e-06     2.8e-07
============  =========  ==========  ==========  ==========

Five orders between the smooth kernel and the one with a kink at its radius,
and that ordering is the point: a compact kernel is harder to integrate than a
heavy-tailed one, which is harder than a Gaussian. The numbers are what a caller
needs in order to choose `n_quad`, and the fact that they converge is what says
neither backend is wrong.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    CompactSpatial,
    ExponentialKernel,
    GaussianSpatial,
    History,
    ParetoSpatial,
    SpatioTemporalLogLikelihood,
    spatio_temporal_model,
)

TIMES = np.array([0.31, 0.47, 1.05, 1.09, 2.30, 2.88, 3.02, 4.51, 5.60, 5.93, 7.15, 8.02, 9.44])

#: family, theta, and the agreement measured at ``n_quad = 64``.
CASES = {
    "gaussian": (GaussianSpatial(1), np.array([0.5, 0.6, 1.5, 0.6]), 1e-9),
    "pareto": (ParetoSpatial(1), np.array([0.5, 0.6, 1.5, 0.5, 1.4]), 1e-6),
    "compact": (CompactSpatial(1), np.array([0.5, 0.6, 1.5, 1.2]), 1e-4),
}


@pytest.fixture
def spatial_history():
    points = np.random.default_rng(4).uniform(-np.pi, np.pi, size=(1, TIMES.size))
    return History(TIMES, points, 0.0, end=10.0)


def model_with(family, n_quad=64):
    return spatio_temporal_model(
        hp.Circle(), spatial=family, temporal=ExponentialKernel(), n_quad=n_quad
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_both_backends_compute_the_same_number(name, spatial_history):
    """To the accuracy of the quadrature, which is stated per family above."""
    family, theta, tolerance = CASES[name]
    model = model_with(family)
    cached = SpatioTemporalLogLikelihood(model, backend="cached").total(theta, spatial_history)
    hooks = SpatioTemporalLogLikelihood(model, backend="hooks").total(theta, spatial_history)

    assert np.isfinite(cached)
    assert cached == pytest.approx(hooks, rel=tolerance)


@pytest.mark.parametrize("name", ["pareto", "compact"])
def test_refining_the_rule_closes_the_gap(name, spatial_history):
    """Which is what says the difference is quadrature error and not a defect.

    Two implementations that disagreed for a *structural* reason would disagree
    by the same amount however fine the rule.
    """
    family, theta, _ = CASES[name]

    def gap(n_quad):
        model = model_with(family, n_quad=n_quad)
        cached = SpatioTemporalLogLikelihood(model, backend="cached").total(theta, spatial_history)
        hooks = SpatioTemporalLogLikelihood(model, backend="hooks").total(theta, spatial_history)
        return abs(cached - hooks) / abs(hooks)

    assert gap(256) < 0.1 * gap(64)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_cached_backend_is_the_one_that_runs(name, spatial_history):
    """None of the three trips the non-negativity precondition.

    `CompactSpatial` is the one that could: it is zero over most of a large
    domain and the cached form's separability identity holds only where the
    pre-floor integrand is non-negative at every node. Zero is non-negative, so
    the fast path survives -- but it is worth an assertion rather than an
    assumption, because the fallback is a warning and a hundredfold slowdown.
    """
    family, theta, _ = CASES[name]
    likelihood = SpatioTemporalLogLikelihood(model_with(family), backend="auto")
    likelihood.total(theta, spatial_history)
    assert likelihood.backend_used == "cached"


def test_a_compact_kernel_below_the_resolution_floor_is_refused():
    """`min_scale` is the radius, and the support bounds it by the panel width.

    Below `2 * width / n_quad` a kernel can fall entirely between two nodes: the
    quadrature then reports the background integral alone, the excitation is
    invisible to the thinning, and the process degenerates towards Poisson in
    time while the location sampler still sees the kernel.
    """
    model = model_with(CompactSpatial(1), n_quad=32)
    floor = 2.0 * (2.0 * np.pi) / 32
    inside = np.array([[0.5, 0.6, 1.5, floor * 2.0]])
    outside = np.array([[0.5, 0.6, 1.5, floor * 0.5]])
    assert bool(model.support(inside)[0])
    assert not bool(model.support(outside)[0])
