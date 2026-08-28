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
from _pytest.mark.structures import ParameterSet

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


#: Stopping rules the loop can be driven by. The invariant is a property of the
#: *bound*, not of what ends the run, so every case below is checked under both:
#: `simulate_until` reaches the acceptance test through a second loop, and a
#: bound that only dominates on the path `simulate` happens to take would be a
#: bound that does not dominate.
STOPPING = ["count", "horizon"]


def last_time(proc):
    """Time of the most recent event, whichever record layout `proc` uses."""
    record = proc.events
    return float(record[-1] if record.ndim == 1 else record[0, -1])


def stopping_rule(build, name, seed, size, stop):
    """Return a process and a no-argument callable that drives it.

    The driver is built before `instrument` wraps the hooks, so the work of
    sizing the horizon is not recorded as a comparison the loop never made.

    A horizon cannot be guessed from the intensity here. Several cases in the
    tables below are deliberately **supercritical** -- the bump kernel on a unit
    circle integrates to pi over the domain and the temporal kernel carries mass
    0.45, so its branching ratio is 1.41 -- and they stay finite only because
    `simulate` stops counting. Given a horizon instead, such a process explodes
    and the run ends in the stall guard rather than in an assertion about the
    bound.

    So the horizon is read off a reference realisation of the same length --
    ``simulate_until(t_k)`` reproduces ``simulate(k)`` exactly, which
    `test_simulate_until` pins, so the horizon run then covers the same ground
    through the other loop.

    The reference runs the full `size` rather than half of it, and the halving
    is not a saving worth having: `_check` insists that some candidate was
    *rejected*, and the counts below are already the smallest at which that is
    reliable. At four events on a torus instead of eight, the smallest recorded
    ratio was 0.9935 against the 0.99 the check wants -- a bound that is fine,
    reported as degenerate because the run was too short to see it work.
    """
    proc = build(name, seed)
    if stop == "count":
        return proc, lambda: proc.simulate(size)

    reference = build(name, seed)
    reference.simulate(size)
    horizon = last_time(reference)
    return proc, lambda: proc.simulate_until(horizon)


TEMPORAL = [
    "ExponentialHawkes",
    "MonotoneKernelHawkes",
    "BellShapeHawkes",
    # A kernel that is flat at lag 0 and peaks later. The old fmin-from-zero
    # search returned 0 here, collapsing the peak value to 0 and silently
    # disabling the bell-shaped bound: 46% of steps violated M >= lambda.
    "DelayedBellShapeHawkes",
]

#: Spatio-temporal cases. Before 0.2.0 only `st-circle` was covered, which is
#: why every other configuration below shipped with a broken bound. The
#: `legacy` case went with the frozen class it exercised, in 0.4.0.
SPATIO_TEMPORAL = [
    "st-circle",
    pytest.param("st-torus", marks=pytest.mark.slow),
    "st-signed",
    "st-delayed",  # monotone_temporal_kernel=False with a delayed kernel
    "st-periodic",
    # A domain that is a proper subset of its bounding box, so the quadrature
    # rule is masked. Masking is exactly the kind of change this harness exists
    # to police: it alters the node set the bound and the acceptance test share,
    # and that shared node set is the whole reason M >= lambda holds.
    "st-hexagon",
    "st-hexagon-periodic",  # the same, through make_periodic's orbit branch
    pytest.param("st-rectangle", marks=pytest.mark.slow),
    # The curved domains. Each one changes something the domination argument
    # depends on, and the argument only ever needed the bound and the acceptance
    # test to share one node set with strictly positive weights -- so each is a
    # case where that could quietly stop being true.
    #
    # `volume_element` is no longer 1, so the weights are rescaled per node and
    # the location sampler is handed a different density than the intensity.
    "st-sphere",
    # A deck group containing an orientation-reversing element, so `wrap` is not
    # a translation and the quotient distance is not a lattice reduction.
    "st-klein",
    # A domain on a curved model space, where `contains` masks the rule *and*
    # the measure varies across it.
    pytest.param("st-projective", marks=pytest.mark.slow),
    # Negative curvature: an infinite deck group, a chart whose bounding box
    # reaches outside the model space, and a distance that is certified rather
    # than truncated by word length.
    pytest.param("st-crosscaps", marks=pytest.mark.slow),
]


@pytest.fixture
def build(
    exp_kernel,
    triangular_kernel,
    bump_spatial,
    delayed_bump_kernel,
    signed_spatial,
):
    def _spatio_temporal(**kwargs):
        defaults = {
            "base": lambda x: 0.5,
            "spatial": bump_spatial,
            "temporal": exp_kernel,
            "domain": hp.Circle(),
            "monotone_temporal_kernel": True,
        }
        defaults.update(kwargs)
        return hp.SpatioTemporalHawkesProcess(**defaults)

    def _build(name, seed):
        if name == "ExponentialHawkes":
            return hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=seed)
        if name == "MonotoneKernelHawkes":
            return hp.MonotoneKernelHawkes(exp_kernel, rng=seed)
        if name == "BellShapeHawkes":
            return hp.BellShapeHawkes(triangular_kernel, rng=seed)
        if name == "DelayedBellShapeHawkes":
            return hp.BellShapeHawkes(delayed_bump_kernel, rng=seed)
        if name == "st-circle":
            return _spatio_temporal(rng=seed)
        if name == "st-torus":
            return _spatio_temporal(domain=hp.Torus2D(), rng=seed)
        if name == "st-signed":
            return _spatio_temporal(spatial=signed_spatial, rng=seed)
        if name == "st-delayed":
            return _spatio_temporal(
                temporal=delayed_bump_kernel, monotone_temporal_kernel=False, rng=seed
            )
        if name == "st-periodic":
            return _spatio_temporal(spatial=hp.make_periodic(bump_spatial, hp.Circle()), rng=seed)
        if name == "st-hexagon":
            return _spatio_temporal(domain=hp.FundamentalDomain.hexagon(1.0), rng=seed)
        if name == "st-hexagon-periodic":
            hexagon = hp.FundamentalDomain.hexagon(1.0)
            return _spatio_temporal(
                domain=hexagon, spatial=hp.make_periodic(bump_spatial, hexagon), rng=seed
            )
        if name == "st-rectangle":
            return _spatio_temporal(domain=hp.FundamentalDomain.rectangle(), rng=seed)
        if name == "st-sphere":
            return _spatio_temporal(domain=hp.Sphere(), rng=seed)
        if name == "st-klein":
            return _spatio_temporal(domain=hp.FundamentalDomain.klein_bottle(3.0, 3.0), rng=seed)
        if name == "st-projective":
            return _spatio_temporal(domain=hp.FundamentalDomain.projective_plane(), rng=seed)
        if name == "st-crosscaps":
            # Three crosscaps rather than genus two: the smallest hyperbolic
            # surface here, and the only one whose quadrature is affordable to
            # run a simulation on at all.
            return _spatio_temporal(
                domain=hp.FundamentalDomain.crosscaps(3),
                spatial=lambda d: max(0.0, 1.0 - d / 1.5),
                rng=seed,
            )
        raise AssertionError(f"unknown process {name!r}")

    return _build


@pytest.mark.statistical
@pytest.mark.parametrize("stop", STOPPING)
@pytest.mark.parametrize("name", TEMPORAL)
@pytest.mark.parametrize("seed", [11, 23, 47])
def test_temporal_thinning_invariant(build, name, seed, stop):
    proc, drive = stopping_rule(build, name, seed, 300, stop)
    state = instrument(proc, "_conditional_intensity")
    drive()
    _check(state, label=f"{name}(seed={seed}, stop={stop})")


#: The two-dimensional domains cost ~1000 kernel evaluations per integration, so
#: they run shorter. Long enough that candidates are still rejected, which
#: `_check` insists on.
TWO_DIMENSIONAL = {
    "st-torus",
    "st-rectangle",
    "st-hexagon",
    "st-hexagon-periodic",
    "st-sphere",
    "st-klein",
}

#: The curved domains cost more again: a hyperbolic `distance` searches a
#: deck-group window per quadrature node per past event, so four events is
#: already a minute of work. Still long enough that candidates are rejected.
EXPENSIVE = {"st-projective", "st-crosscaps"}


def case_size(name):
    """Events to simulate for a spatio-temporal case, by how much one costs."""
    if name in EXPENSIVE:
        return 4
    return 8 if name in TWO_DIMENSIONAL else 15


def spatio_temporal_cases():
    """Every case crossed with every stopping rule, carrying its own marks.

    The horizon rule costs twice its case -- a reference realisation and the
    instrumented one -- so on the domains that are already the expensive ones it
    is marked `slow` and runs in the coverage job rather than in the ten-job
    matrix. The count rule keeps every case in the fast suite, and the cheap
    domains keep both.
    """
    for case in SPATIO_TEMPORAL:
        name = case.values[0] if isinstance(case, ParameterSet) else case
        inherited = list(case.marks) if isinstance(case, ParameterSet) else []
        for stop in STOPPING:
            costly = stop == "horizon" and name in TWO_DIMENSIONAL | EXPENSIVE
            marks = [*inherited, *([pytest.mark.slow] if costly else [])]
            yield pytest.param(name, stop, marks=marks, id=f"{name}-{stop}")


@pytest.mark.statistical
@pytest.mark.parametrize(("name", "stop"), list(spatio_temporal_cases()))
def test_spatio_temporal_thinning_invariant(build, name, stop):
    """Same invariant, but thinning runs against the space-integrated intensity."""
    proc, drive = stopping_rule(build, name, 11, case_size(name), stop)
    state = instrument(proc, "_integrated_intensity")
    drive()
    _check(state, label=f"{name}(stop={stop})")


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
        past = proc.events[proc.events < t]
        return float(proc.nonlinearity(np.sum(proc.temporal(t - past))))

    proc._upper_bound = broken_bound
    state = instrument(proc, "_conditional_intensity")
    proc.simulate(50)
    with pytest.raises(AssertionError, match=r"violated|every candidate was accepted"):
        _check(state, label="deliberately-broken")
