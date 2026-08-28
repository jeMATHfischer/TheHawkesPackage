"""Every name that was dated for removal must actually be gone.

The global ``filterwarnings = ["error"]`` in pyproject.toml means an *accidental*
deprecation path anywhere fails the suite. This file no longer needs
``pytest.warns`` to opt anything back in, because **the package no longer carries
a single deprecation**: 0.5.0 removed the last of them and the ``_deprecation``
module with them.

0.4.0 emptied this file and refilled it; 0.5.0 has emptied it again. Everything
0.4.0 deprecated -- ``Events``, ``Sim_num``, ``L1``, ``L2`` and the ``n_images``
argument of :class:`~hawkes_package.FundamentalDomain` -- was dated for removal
in 0.5.0 and is gone. So the tests that used to pin the warnings now pin the
absence, which is the same discipline the file has always applied: a name that
was supposed to disappear and quietly did not is exactly as bad as one that
disappeared early.

Two absences are worth more than the rest. ``Events`` had a *setter*, so its
descriptor could not simply be deleted and forgotten -- an assignment to a name
that no longer exists binds a plain attribute and raises nothing, which is why
`test_assigning_a_removed_alias_does_not_silently_shadow_the_real_one` checks
the record rather than the exception. And ``_deprecation`` is gone entirely: an
unused deprecation helper reads as supported machinery, which is worse than none.
"""

import importlib
import subprocess
import sys

import numpy as np
import pytest

import hawkes_package as hp

#: Removed in 0.5.0, having been deprecated since 0.4.0. Kept in lockstep with
#: the removals table in docs/migration.md.
REMOVED_IN_0_5_0 = [
    ("Events", "events"),
    ("Sim_num", "n_simulated"),
]

#: The same, for the two side lengths of a rectangular domain -- removed as
#: constructor keywords *and* as attributes.
REMOVED_SIDES = [("L1", "width"), ("L2", "height")]

#: Removed in 0.4.0. Each was deprecated for two releases first.
REMOVED_IN_0_4_0 = [
    "propagate_by_amount",
    "propagate_by_k_events",
    "propogate_by_amount",
    "Spatio_Temporal_Hawkes_Process",
    "LegacySpatioTemporalHawkesProcess",
]

PROCESS_CLASSES = [
    "ExponentialHawkes",
    "MonotoneKernelHawkes",
    "BellShapeHawkes",
    "SpatioTemporalHawkesProcess",
]


@pytest.fixture
def process(exp_kernel):
    return hp.MonotoneKernelHawkes(exp_kernel, rng=0)


@pytest.fixture
def processes(exp_kernel, triangular_kernel, bump_spatial):
    return {
        "ExponentialHawkes": hp.ExponentialHawkes(np.array([1.0, 0.4, 2.0]), rng=0),
        "MonotoneKernelHawkes": hp.MonotoneKernelHawkes(exp_kernel, rng=0),
        "BellShapeHawkes": hp.BellShapeHawkes(triangular_kernel, rng=0),
        "SpatioTemporalHawkesProcess": hp.SpatioTemporalHawkesProcess(
            lambda x: 0.5,
            bump_spatial,
            exp_kernel,
            domain=hp.Circle(),
            monotone_temporal_kernel=True,
            rng=0,
        ),
    }


# ---------------------------------------------------------------------------
# The event record: the aliases are gone, the real names work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("old", "new"), REMOVED_IN_0_5_0)
@pytest.mark.parametrize("cls_name", PROCESS_CLASSES)
def test_the_old_spelling_is_gone_and_the_new_one_works(processes, cls_name, old, new):
    proc = processes[cls_name]
    proc.simulate(2)

    assert not hasattr(proc, old)
    assert not hasattr(type(proc), old)
    with pytest.raises(AttributeError, match=old):
        getattr(proc, old)

    canonical = getattr(proc, new)
    assert canonical is not None


def test_assigning_a_removed_alias_does_not_silently_shadow_the_real_one(process):
    """The setter was the reason the alias survived 0.4.0, so its absence is checked here.

    ``process.Events = history`` was how a caller conditioned a realisation on
    events it did not simulate. Now that the descriptor is gone, that assignment
    binds a plain instance attribute and raises nothing -- Python has no way to
    refuse it. What must not happen is the *record* changing: the simulator has
    to keep reading ``events``, so a caller still on the old spelling gets a
    realisation that visibly ignored them rather than one that quietly half-worked.
    """
    process.events = np.array([0.5, 1.5])
    process.Events = np.array([99.0, 100.0, 101.0])  # binds a plain attribute

    np.testing.assert_array_equal(process.events, [0.5, 1.5])
    process.simulate(3)
    assert len(process.events) == 5
    assert process.events[0] == 0.5


def test_the_real_names_do_not_warn(process, recwarn):
    process.simulate(2)
    process.events
    process.n_simulated
    process.events = np.array([1.0])
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# The side lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("old", "new"), REMOVED_SIDES)
def test_the_old_side_keyword_is_refused(old, new):
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.Torus2D(**{old: 7.0})
    assert getattr(hp.Torus2D(**{new: 7.0}), new) == 7.0


@pytest.mark.parametrize(("old", "_new"), REMOVED_SIDES)
def test_the_old_side_attribute_is_gone(old, _new):
    torus = hp.Torus2D(width=3.0, height=5.0)
    assert not hasattr(torus, old)
    assert not hasattr(hp.Torus2D, old)


def test_the_old_side_keyword_is_refused_by_both_rectangle_builders():
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.FundamentalDomain.rectangle(L1=3.0, L2=5.0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.FundamentalDomain.klein_bottle(L1=3.0, L2=5.0)

    assert hp.FundamentalDomain.rectangle(3.0, 5.0).volume == pytest.approx(15.0)
    assert hp.FundamentalDomain.klein_bottle(3.0, 5.0).volume == pytest.approx(15.0)


def test_an_unknown_keyword_still_raises():
    """Or a typo would be swallowed and the default silently used.

    This used to be raised by the ``**deprecated`` resolver; now it is Python's
    own, and the message still has to say so.
    """
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.Torus2D(widht=3.0)


def test_the_real_side_names_do_not_warn(recwarn):
    hp.Torus2D(width=3.0, height=5.0)
    hp.FundamentalDomain.rectangle(3.0, 5.0)
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# The retired parameter
# ---------------------------------------------------------------------------


def test_n_images_is_no_longer_a_constructor_argument():
    """It tuned a truncation that now bounds itself, so nothing replaced it.

    The third parameter is keyword-only now, so a caller who passed it
    positionally gets a ``TypeError`` about the count rather than a silently
    misread ``model``.
    """
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    pairings = [
        [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    ]
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.FundamentalDomain(square, pairings, n_images=4)
    with pytest.raises(TypeError, match="positional argument"):
        hp.FundamentalDomain(square, pairings, 4)


def test_the_n_images_attribute_survived_the_argument():
    """`orbit` reads it as its default word length, so it is still API."""
    domain = hp.FundamentalDomain.hexagon(1.0)
    assert domain.n_images == 3
    assert len(domain.orbit([0.0, 0.0], 2)) > 1


def test_building_a_domain_does_not_warn(recwarn):
    hp.FundamentalDomain.hexagon(1.0)
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# What earlier releases removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REMOVED_IN_0_4_0)
def test_the_names_removed_in_0_4_0_are_gone(name):
    """Not merely undocumented: absent, and absent from every process too.

    A name kept alive "just in case" after its removal version is the reason the
    next removal gets argued about instead of executed.
    """
    assert not hasattr(hp, name)
    assert not hasattr(hp.ExponentialHawkes, name)


def test_the_import_shim_is_gone():
    """``import TheHawkesPackage`` was two releases deprecated and is now an error."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("TheHawkesPackage")


def test_the_legacy_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hawkes_package.spatio_temporal.legacy")


def test_the_deprecation_module_is_gone():
    """It lost its last caller when 0.5.0 removed the aliases, so it went too.

    0.4.0 deleted ``DeprecatedAlias`` and ``deprecated_module_getattr`` on this
    exact reasoning -- an unused deprecation helper reads as supported machinery,
    which is worse than none -- and the whole module is now in that position.
    The next deprecation recreates it, which is cheap and honest.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("hawkes_package._deprecation")


def test_an_unknown_top_level_attribute_raises_without_warning(recwarn):
    with pytest.raises(AttributeError):
        hp.does_not_exist
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# Documentation cannot drift from behaviour
# ---------------------------------------------------------------------------


def test_the_package_carries_no_deprecations_at_all(process):
    """The claim docs/migration.md's 0.5.0 section makes, checked rather than asserted.

    Every name in the two removal tables must be absent, and every replacement
    present. If a sixth deprecation is ever added, this test is what forces the
    tables -- and so the migration guide -- to be updated with it.
    """
    for old, new in REMOVED_IN_0_5_0:
        assert not hasattr(hp.HawkesProcess, old)
        assert hasattr(process, new)
    for old, new in REMOVED_SIDES:
        assert not hasattr(hp.Torus2D, old)
        assert hasattr(hp.Torus2D(), new)


def test_no_module_emits_a_deprecation_on_import():
    """Nothing left to warn about, so an import that warns is a regression.

    In a subprocess, and emphatically **not** via ``importlib.reload``. Reloading
    ``spatio_temporal.domains`` rebinds `Circle` and `Torus2D` to fresh class
    objects, and ``kernels.image_distance_fn`` resolves them with a late
    ``from .domains import`` -- so every domain built before the reload stops
    satisfying ``isinstance`` and silently takes the generic deck-group branch
    instead of its own. That poisons the rest of the session: the symptom was
    `test_torus_sums_over_the_full_image_lattice` failing on a truncation warning
    a `Torus2D` should never reach, several files later.
    """
    code = "import hawkes_package, hawkes_package.inference"
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
