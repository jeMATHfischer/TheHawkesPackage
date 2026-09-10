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
from hawkes_package.base import TemporalHawkesProcess
from hawkes_package.exponential import _DecayCursor
from hawkes_package.inference import CompactSpatial, OmoriUtsuKernel, ParetoSpatial


def total(value):
    """Reduce a vector intensity to the scalar its single bound dominates.

    ``cumsum(v)[-1]`` rather than ``sum(v)``, and the difference is not
    pedantry: the multivariate loop partitions ``(0, M]`` with the cumulative
    intensities and compares its draw against their last entry, while ``sum``
    reduces pairwise and is free to land a bit away. Recording one while the
    loop compared the other would mean the harness checks a number the loop
    never used.
    """
    return float(np.cumsum(np.asarray(value, dtype=float))[-1])


def instrument(proc, lam_name, *, reduce=None):
    """Record every ``(M, lambda)`` pair the acceptance test actually compares.

    ``_upper_bound`` calls the intensity hook internally on some classes, so a
    depth counter suppresses those nested calls; only the loop's own evaluation
    is recorded.

    `reduce` maps the hook's return value to the scalar the bound dominates. It
    is ``None`` for a univariate class, whose hook already returns that scalar,
    and :func:`total` for a multivariate one, whose hook returns the per-type
    intensities against whose *sum* the one bound is drawn. Recording a single
    component instead would check a weaker inequality than the loop relies on,
    and would pass on a bound too small by a factor approaching the number of
    types.
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
            state["pairs"].append((state["m"], value if reduce is None else reduce(value)))
        return value

    proc._upper_bound = bound
    setattr(proc, lam_name, lam)

    if carries_a_cursor(proc):
        # The loop reads a class like `ExponentialHawkes` through a cursor and
        # never calls the two hooks patched above, so patching them alone
        # records nothing at all. That is not hypothetical: when the cursor
        # landed, every `ExponentialHawkes` case here failed on
        # ``assert len(pairs) > 0`` rather than passing vacuously -- the one
        # assertion in `_check` that exists to catch a harness gone blind.
        original = proc._cursor

        def make_cursor(start, _original=original):
            return _RecordingCursor(_original(start), state, reduce)

        proc._cursor = make_cursor

    return state


def carries_a_cursor(proc):
    """Whether `proc` reads the intensity through a cursor of its own."""
    return (
        isinstance(proc, TemporalHawkesProcess)
        and type(proc)._cursor is not TemporalHawkesProcess._cursor
    )


class _RecordingCursor:
    """Delegates to a real cursor and records what the loop actually compared.

    The pair is ``(the bound drawn against, the intensity tested against it)``,
    the same pair the hook-level instrumentation records -- so a class with a
    cursor is held to exactly the inequality every other class is.
    """

    def __init__(self, inner, state, reduce=None):
        self.inner = inner
        self.state = state
        self.reduce = reduce

    @property
    def time(self):
        return self.inner.time

    def move_to(self, t):
        self.inner.move_to(t)

    def bound(self):
        value = self.inner.bound()
        self.state["m"] = value
        return value

    def intensity(self):
        value = self.inner.intensity()
        self.state["pairs"].append(
            (self.state["m"], value if self.reduce is None else self.reduce(value))
        )
        return value

    def accept(self):
        self.inner.accept()


def as_vector_hook(proc, lam_name):
    """Make `proc`'s intensity hook return a length-1 vector instead of a scalar.

    No multivariate class exists yet, so this is how the reducing path above is
    exercised: the same process, the same seed, the same numbers -- only the
    hook's *shape* changes. A size-1 array multiplies, compares and tests truthy
    exactly as the scalar did, so the loop consumes an identical stream of draws
    and the recorded pairs must come out identical too. That equality is the
    whole point: it pins `reduce` as inert on the path it does not apply to.
    """
    orig = getattr(proc, lam_name)

    def vector(t, **kwargs):
        return np.array([orig(t, **kwargs)], dtype=float)

    setattr(proc, lam_name, vector)


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
    # A power-law kernel, 1.0.0. Monotone, so its bound is the value at the
    # current time -- but the *reason* it needs a case is the tail: every past
    # event still contributes at every later step, where an exponential's
    # contribution has underflowed to nothing. A bound that quietly dropped old
    # events would pass every exponential case here and fail this one.
    "OmoriKernelHawkes",
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
    # The two spatial families added in 1.0.0. The power law puts real mass at
    # every distance the domain reaches, so the space-integrated bound cannot
    # rely on the kernel having decayed by the boundary; the compact one is
    # exactly zero over most of the domain, which is the opposite stress -- a
    # quadrature rule that misses its support entirely would report the
    # background integral and accept everything.
    "st-pareto",
    "st-compact",
    # A background that varies in *time*, 1.0.0. The one case here whose bound
    # is not a statement about the kernel: a candidate is drawn ahead of the
    # moment the bound is computed, so the background contribution has to be the
    # supremum over every later time and not the value at the current one. The
    # two differ by the whole amplitude of the cycle, and the error is in the
    # direction that accepts everything.
    "st-periodic-background",
    # The same at an amplitude that takes the schedule to zero for part of the
    # cycle, where the intensity is the excitation alone and the acceptance
    # ratio is at its most extreme.
    "st-periodic-deep",
    pytest.param("st-rectangle", marks=pytest.mark.slow),
    # Bounded, non-periodic. Every other case here is a closed surface.
    pytest.param("st-bounded-rect", marks=pytest.mark.slow),
    "st-bounded-interval",
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
        if name == "marked-light":
            # A mark law far heavier than the productivity that feeds it, so
            # E[g] = b/(b-a) is close to one and the process is barely marked.
            return hp.ExponentialMarkedHawkes(
                mu=1.0, alpha=0.3, beta=2.0, scale=0.2, b_value=2.0, rng=seed
            )
        if name == "marked-heavy":
            # scale just below b_value: E[g] = 10, and the realised marks reach
            # far enough that a handful of events dominate the whole intensity.
            # This is the configuration most likely to expose an under-bound and
            # exactly the kind the 0.2.0 harness did not reach -- and it is safe
            # only because the bound sums over marks already drawn.
            return hp.ExponentialMarkedHawkes(
                mu=1.0, alpha=0.09, beta=2.0, scale=0.9, b_value=1.0, rng=seed
            )
        if name == "marked-bell":
            # A rising kernel *and* a mark: each event is bounded by its own
            # future supremum, scaled by its own productivity, and the two
            # per-event factors have to be applied to the same event.
            return hp.MarkedHawkes(
                mu=0.5,
                temporal=triangular_kernel,
                productivity=lambda m: np.exp(0.5 * np.asarray(m, dtype=float)),
                mark_sampler=lambda rng_: float(rng_.exponential(1.0)),
                monotone_temporal_kernel=False,
                rng=seed,
            )
        if name == "mv-exp-d2":
            # Linear, asymmetric: type 1 excites type 0 harder than the reverse,
            # so the two components are genuinely different functions and a bound
            # that quietly used one of them for both would show.
            return hp.MultivariateHawkes(
                mu=[0.5, 0.2],
                excitation=[[0.6, 0.9], [0.3, 0.5]],
                temporal=exp_kernel,
                rng=seed,
            )
        if name == "mv-exp-d3-asym":
            # Three types, no symmetry and two zero entries: a component that
            # receives nothing from a source still has to be bounded, and the
            # cumulative slices have to stay ordered when one of them is empty.
            return hp.MultivariateHawkes(
                mu=[0.4, 0.1, 0.3],
                excitation=[[0.5, 0.2, 0.0], [0.3, 0.4, 0.2], [0.0, 0.3, 0.5]],
                temporal=exp_kernel,
                rng=seed,
            )
        if name == "mv-monotone-d2":
            # A nonlinearity on top, so the bound is only valid because phi is
            # monotone increasing -- the same condition the univariate class rests
            # on, now applied to a vector.
            return hp.MultivariateHawkes(
                mu=[0.0, 0.0],
                excitation=[[0.7, 0.4], [0.2, 0.6]],
                temporal=exp_kernel,
                nonlinearity=lambda x: x + 2,
                rng=seed,
            )
        if name == "mv-bell-d2":
            # Non-monotone kernel: each event is bounded by its own future
            # supremum, and with two types several events can be rising at once
            # from different sources. That is the configuration the univariate
            # bell-shaped bound was originally wrong about.
            return hp.MultivariateHawkes(
                mu=[0.4, 0.2],
                excitation=[[0.6, 0.5], [0.4, 0.7]],
                temporal=triangular_kernel,
                monotone_temporal_kernel=False,
                rng=seed,
            )
        if name == "mv-delayed-d2":
            # The kernel that is flat at lag 0, which collapsed the univariate
            # peak search to zero and silently disabled the bell-shaped bound.
            return hp.MultivariateHawkes(
                mu=[0.4, 0.2],
                excitation=[[0.6, 0.5], [0.4, 0.7]],
                temporal=delayed_bump_kernel,
                monotone_temporal_kernel=False,
                rng=seed,
            )
        if name == "mv-d1":
            # The degenerate multivariate process: one type, and by construction
            # the same intensity `MonotoneKernelHawkes` has. It earns its place
            # because it reaches that intensity through the *vector* hook and
            # the summed bound, so a defect in the reduction or in the type draw
            # shows up here against a case whose correct answer is already known.
            return hp.MultivariateHawkes(
                mu=[0.0],
                excitation=[[1.0]],
                temporal=exp_kernel,
                nonlinearity=lambda x: x + 2,
                rng=seed,
            )
        if name == "OmoriKernelHawkes":
            # Mass 0.9 * 0.3**-0.8 / 0.8 = 2.99 at these parameters, which with
            # the default phi(x) = x + 2 is supercritical -- and deliberately
            # so: `simulate` stops counting, and a heavily excited run is where
            # an under-bound shows.
            return hp.MonotoneKernelHawkes(
                OmoriUtsuKernel().build(np.array([0.9, 0.3, 1.8])), rng=seed
            )
        if name == "st-pareto":
            return _spatio_temporal(spatial=ParetoSpatial(1).build(np.array([0.5, 1.4])), rng=seed)
        if name == "st-compact":
            # The radius is comfortably above the quadrature's resolution floor
            # on a unit circle; below it the rule would miss the support and the
            # process would silently degenerate towards Poisson in time, which
            # `check_resolution` warns about at construction.
            return _spatio_temporal(spatial=CompactSpatial(1).build(np.array([1.2])), rng=seed)
        if name == "st-periodic-background":
            schedule = hp.PeriodicSchedule([0.6], [0.3], period=4.0)
            return _spatio_temporal(base=hp.PeriodicBackground(lambda x: 0.5, schedule), rng=seed)
        if name == "st-periodic-deep":
            # Amplitude 1.0 with one harmonic: the raw series touches zero, so
            # the floor bites for an instant each period.
            schedule = hp.PeriodicSchedule([1.0], [0.0], period=3.0)
            return _spatio_temporal(base=hp.PeriodicBackground(lambda x: 0.6, schedule), rng=seed)
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
        if name == "st-bounded-rect":
            # The first domain with an *edge*. Nothing wraps, so the kernel mass
            # over the domain depends on where an event sits -- and the bound is
            # integrated over a rule whose nodes fill the box exactly, as on a
            # torus, but whose geometry is not a quotient. The invariant has to
            # hold before any edge correction is applied, so that the correction
            # can be shown not to have broken it.
            return _spatio_temporal(domain=hp.Rectangle(4.0, 3.0), rng=seed)
        if name == "st-bounded-interval":
            # One dimension, the non-periodic counterpart of `st-circle`. A bump
            # kernel of reach pi on an interval of length 4 loses real mass off
            # both ends, which is the whole phenomenon in its simplest form.
            return _spatio_temporal(domain=hp.Rectangle(4.0), rng=seed)
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


#: Multivariate cases. The bound is one number dominating the *total* of a
#: vector intensity, so these are instrumented on the vector hook and reduced --
#: recording a single component would check a weaker inequality than the loop
#: relies on.
#: Marked cases. The productivity multiplies each event's kernel by a number
#: drawn from an unbounded law, which sounds like a threat to the bound and is
#: not: both the intensity and the bound sum over marks that have already been
#: observed, so the supremum of `g` over the mark *distribution* never enters.
MARKED = ["marked-light", "marked-heavy", "marked-bell"]


@pytest.mark.statistical
@pytest.mark.parametrize("stop", STOPPING)
@pytest.mark.parametrize("name", MARKED)
@pytest.mark.parametrize("seed", [11, 23, 47])
def test_marked_thinning_invariant(build, name, seed, stop):
    """The same invariant with a per-event productivity in front of the kernel."""
    proc, drive = stopping_rule(build, name, seed, 300, stop)
    state = instrument(proc, "_conditional_intensity")
    drive()
    _check(state, label=f"{name}(seed={seed}, stop={stop})")


MULTIVARIATE = [
    "mv-d1",
    "mv-exp-d2",
    "mv-exp-d3-asym",
    "mv-monotone-d2",
    "mv-bell-d2",
    "mv-delayed-d2",
]


@pytest.mark.statistical
@pytest.mark.parametrize("stop", STOPPING)
@pytest.mark.parametrize("name", MULTIVARIATE)
@pytest.mark.parametrize("seed", [11, 23, 47])
def test_multivariate_thinning_invariant(build, name, seed, stop):
    """Same invariant, against the summed intensity one bound has to dominate."""
    proc, drive = stopping_rule(build, name, seed, 300, stop)
    state = instrument(proc, "_component_intensities", reduce=total)
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
    "st-bounded-rect",
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


#: The temporal cases whose loop reads the intensity *hook*, which is what
#: `as_vector_hook` can reshape. `ExponentialHawkes` reads a cursor instead, so
#: wrapping its hook changes nothing the loop sees -- a fact worth stating here
#: rather than discovering as a mysteriously passing test.
THROUGH_THE_HOOK = [name for name in TEMPORAL if name != "ExponentialHawkes"]


@pytest.mark.statistical
@pytest.mark.parametrize("name", THROUGH_THE_HOOK)
def test_reducing_a_vector_intensity_records_the_same_pairs(build, name):
    """`reduce` must not disturb what the harness records.

    The multivariate loop draws one bound against the *total* of a vector
    intensity, so the harness has to reduce before comparing. This pins that the
    reduction is the only difference between the two paths: same seed, same
    draws, same pairs, exactly. Without it a reducing harness could quietly
    record something other than what the loop compared, and every multivariate
    invariant test below would be checking the wrong inequality.
    """
    plain, drive_plain = stopping_rule(build, name, 11, 60, "count")
    plain_state = instrument(plain, "_conditional_intensity")
    drive_plain()

    reduced, drive_reduced = stopping_rule(build, name, 11, 60, "count")
    as_vector_hook(reduced, "_conditional_intensity")
    reduced_state = instrument(reduced, "_conditional_intensity", reduce=total)
    drive_reduced()

    assert len(plain_state["pairs"]) > 0
    np.testing.assert_array_equal(plain_state["pairs"], reduced_state["pairs"])
    np.testing.assert_array_equal(plain.events, reduced.events)
    _check(reduced_state, label=f"{name}(reduced)")


@pytest.mark.statistical
@pytest.mark.parametrize("reduce", [None, total], ids=["scalar", "reduced"])
def test_instrumentation_detects_a_broken_bound(exp_kernel, reduce):
    """Guard the guard: a deliberately too-tight bound must be caught.

    Without this, a bug in `instrument` could make every invariant test pass
    vacuously -- and that has to stay true on the reducing path, which is the
    one the multivariate cases are checked through.
    """
    proc = hp.MonotoneKernelHawkes(exp_kernel, rng=3)
    proc.simulate(20)

    # Reproduce the pre-0.2.0 bug: exclude the most recent event from the bound.
    def broken_bound(t):
        past = proc.events[proc.events < t]
        return float(proc.nonlinearity(np.sum(proc.temporal(t - past))))

    proc._upper_bound = broken_bound
    if reduce is not None:
        as_vector_hook(proc, "_conditional_intensity")
    state = instrument(proc, "_conditional_intensity", reduce=reduce)
    proc.simulate(50)
    with pytest.raises(AssertionError, match=r"violated|every candidate was accepted"):
        _check(state, label="deliberately-broken")


@pytest.mark.statistical
def test_instrumentation_detects_a_broken_bound_inside_a_cursor():
    """The same guard, on the path where the bound lives in carried state.

    A class with a cursor computes its bound from a running sum, so a defect
    there never touches `_upper_bound` and the hook-level instrumentation cannot
    see it. This breaks the cursor the way the pre-0.2.0 bug broke the hook --
    the most recent event left out of the bound -- and requires the harness to
    catch it, which is what makes the `ExponentialHawkes` cases above worth
    anything.
    """
    proc = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=3)

    class NeverAbsorbs(_DecayCursor):
        """The bound stops seeing events as they are accepted."""

        def accept(self):
            self.n_events += 1  # the count keeps up; the sum does not

    proc._cursor = lambda start: NeverAbsorbs(proc, start)
    state = instrument(proc, "_conditional_intensity")
    proc.simulate(60)
    with pytest.raises(AssertionError, match=r"violated|every candidate was accepted"):
        _check(state, label="deliberately-broken-cursor")
