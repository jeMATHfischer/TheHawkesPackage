"""The Ogata thinning invariant, checked on every process class at once.

Ogata's algorithm is only correct while ``M >= lambda`` holds at every
acceptance test. Two ways to get this wrong, and the assertions below catch
both:

* **M too tight** — the historical ``MonotoneKernelHawkes`` bug, where the bound
  excluded the most recent event. Every candidate is then accepted and the
  output is a Poisson process wearing a Hawkes costume. Caught by requiring
  that some candidates are actually rejected.
* **M invalid** — the bound simply fails to dominate. Caught directly.

Before 0.2.0 this lived in ``test_monotone.py`` as a hand-copied
re-implementation of the simulation loop, which could only ever check the one
class it duplicated. Naming the two hooks in the base class turned it into a
single parametrized test over all five.
"""

import numpy as np
import pytest

import hawkes_package as hp


def instrument(proc, lam_name):
    """Record every ``(M, lambda)`` pair the acceptance test actually compares.

    ``_upper_bound`` calls the intensity hook internally on some classes, so a
    depth counter suppresses those nested calls; only the loop's own evaluation
    is recorded.
    """
    orig_bound = proc._upper_bound
    orig_lam = getattr(proc, lam_name)
    state = {"m": None, "depth": 0, "pairs": []}

    def bound(t):
        state["depth"] += 1
        try:
            value = orig_bound(t)
        finally:
            state["depth"] -= 1
        state["m"] = value
        return value

    def lam(t, **kwargs):
        value = orig_lam(t, **kwargs)
        # Record only the loop's own evaluation: not the nested call made while
        # computing the bound, and not a bound-mode evaluation.
        if state["depth"] == 0 and not kwargs.get("bound") and state["m"] is not None:
            state["pairs"].append((state["m"], value))
        return value

    proc._upper_bound = bound
    setattr(proc, lam_name, lam)
    return state


def _check(state, *, label):
    pairs = np.array(state["pairs"], dtype=float)
    assert len(pairs) > 0, f"{label}: no acceptance tests were recorded"

    m, lam = pairs[:, 0], pairs[:, 1]
    violations = int(np.sum(lam > m + 1e-9))
    assert violations == 0, (
        f"{label}: thinning invariant M >= lambda violated {violations}/{len(pairs)} times; "
        f"worst excess {float(np.max(lam - m)):.3e}"
    )

    ratio = lam / m
    assert np.all(ratio <= 1.0 + 1e-9)
    assert float(np.min(ratio)) < 0.99, (
        f"{label}: every candidate was accepted (min ratio {float(np.min(ratio)):.4f}); "
        "the bound is degenerate and this is a Poisson process, not a Hawkes one"
    )


TEMPORAL = ["ExponentialHawkes", "MonotoneKernelHawkes", "BellShapeHawkes"]


@pytest.fixture
def build(exp_kernel, triangular_kernel, legacy_kernels, bump_spatial):
    base, spatial, temporal = legacy_kernels

    def _build(name, seed):
        if name == "ExponentialHawkes":
            return hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=seed)
        if name == "MonotoneKernelHawkes":
            return hp.MonotoneKernelHawkes(exp_kernel, rng=seed)
        if name == "BellShapeHawkes":
            return hp.BellShapeHawkes(triangular_kernel, rng=seed)
        if name == "SpatioTemporalHawkesProcess":
            return hp.SpatioTemporalHawkesProcess(
                lambda x: 0.5,
                bump_spatial,
                exp_kernel,
                domain=hp.Circle(),
                monotone_temporal_kernel=True,
                rng=seed,
            )
        return hp.LegacySpatioTemporalHawkesProcess(base, spatial, temporal, rng=seed)

    return _build


@pytest.mark.statistical
@pytest.mark.parametrize("name", TEMPORAL)
@pytest.mark.parametrize("seed", [11, 23, 47])
def test_temporal_thinning_invariant(build, name, seed):
    proc = build(name, seed)
    state = instrument(proc, "_conditional_intensity")
    proc.simulate(300)
    _check(state, label=f"{name}(seed={seed})")


@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.parametrize(
    "name", ["SpatioTemporalHawkesProcess", "LegacySpatioTemporalHawkesProcess"]
)
def test_spatio_temporal_thinning_invariant(build, name):
    """Same invariant, but thinning runs against the space-integrated intensity."""
    proc = build(name, 11)
    state = instrument(proc, "_integrated_intensity")
    proc.simulate(15)
    _check(state, label=name)


@pytest.mark.statistical
def test_instrumentation_detects_a_broken_bound(exp_kernel):
    """Guard the guard: a deliberately too-tight bound must be caught.

    Without this, a bug in `instrument` could make every invariant test pass
    vacuously.
    """
    proc = hp.MonotoneKernelHawkes(exp_kernel, rng=3)
    proc.simulate(20)

    # Reproduce the pre-0.2.0 bug: exclude the most recent event from the bound.
    def broken_bound(t):
        past = proc.Events[proc.Events < t]
        return float(proc.nonlinearity(np.sum(proc.temporal(t - past))))

    proc._upper_bound = broken_bound
    state = instrument(proc, "_conditional_intensity")
    proc.simulate(50)
    with pytest.raises(AssertionError, match="violated|every candidate was accepted"):
        _check(state, label="deliberately-broken")
