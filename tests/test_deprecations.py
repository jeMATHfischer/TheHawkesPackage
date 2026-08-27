"""Every deprecated name must warn, still work, and forward to the same object.

The global ``filterwarnings = ["error"]`` in pyproject.toml means an
*accidental* deprecation path anywhere fails the suite; ``pytest.warns`` below
opts the intentional ones back in.

0.4.0 emptied this file and refilled it. Everything it used to pin — the
``TheHawkesPackage`` import shim, the three ``propagate_*`` spellings, the
ambiguous ``Spatio_Temporal_Hawkes_Process`` name and the frozen legacy class —
was dated for removal in 0.4.0 and is gone, and what replaced it is the naming
sweep that release carried. The tests that guarded the removals now guard the
*absence*: a name that was supposed to disappear and quietly did not is exactly
as bad as one that disappeared early.
"""

import importlib

import numpy as np
import pytest

import hawkes_package as hp

#: (old spelling, new spelling, removal version) — kept in lockstep with the
#: rename table in docs/migration.md.
RENAMES = [
    ("Events", "events", "0.5.0"),
    ("Sim_num", "n_simulated", "0.5.0"),
]

#: The same, for the two side lengths of a rectangular domain.
SIDE_RENAMES = [("L1", "width"), ("L2", "height")]

#: (what is going away, why, removal version). Distinct from RENAMES: nothing
#: replaces these, they simply stopped being needed.
RETIREMENTS = [
    ("the n_images argument of FundamentalDomain", "truncated by displacement radius", "0.5.0"),
]

#: Removed in 0.4.0. Each was deprecated for two releases first.
REMOVED = [
    "propagate_by_amount",
    "propagate_by_k_events",
    "propogate_by_amount",
    "Spatio_Temporal_Hawkes_Process",
    "LegacySpatioTemporalHawkesProcess",
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
# The renamed event record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("old", "new", "version"), RENAMES)
@pytest.mark.parametrize(
    "cls_name",
    [
        "ExponentialHawkes",
        "MonotoneKernelHawkes",
        "BellShapeHawkes",
        "SpatioTemporalHawkesProcess",
    ],
)
def test_the_old_spelling_warns_and_reads_the_new_one(processes, cls_name, old, new, version):
    proc = processes[cls_name]
    proc.simulate(2)
    with pytest.warns(DeprecationWarning, match=rf"{old}.*removed in the-hawkes-package {version}"):
        through_alias = getattr(proc, old)
    canonical = getattr(proc, new)
    if isinstance(canonical, np.ndarray):
        np.testing.assert_array_equal(through_alias, canonical)
    else:
        assert through_alias == canonical


def test_assigning_through_the_old_spelling_still_seeds_the_record(process):
    """The setter is not optional.

    ``process.Events = history`` is how a caller conditions a realisation on
    events it did not simulate, and a descriptor with only ``__get__`` would let
    that assignment bind a plain instance attribute *over* the descriptor: no
    warning, no error, and a simulator that never sees the history.
    """
    with pytest.warns(DeprecationWarning, match="Events"):
        process.Events = np.array([0.5, 1.5])
    np.testing.assert_array_equal(process.events, [0.5, 1.5])
    process.simulate(3)
    assert len(process.events) == 5
    assert process.events[0] == 0.5


def test_the_new_spelling_does_not_warn(process, recwarn):
    process.simulate(2)
    process.events
    process.n_simulated
    process.events = np.array([1.0])
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


def test_the_alias_is_readable_on_the_class_without_warning(recwarn):
    """``help()`` and ``inspect`` must not fire a deprecation at import time."""
    assert hp.HawkesProcess.Events.__doc__.startswith("Deprecated alias")
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# The renamed side lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("old", "new"), SIDE_RENAMES)
def test_the_old_side_keyword_warns_and_still_sizes_the_domain(old, new):
    with pytest.warns(DeprecationWarning, match=rf"the {old} argument.*use {new}"):
        torus = hp.Torus2D(**{old: 7.0})
    assert getattr(torus, new) == 7.0


@pytest.mark.parametrize(("old", "new"), SIDE_RENAMES)
def test_the_old_side_attribute_warns_and_reads_the_new_one(old, new):
    torus = hp.Torus2D(width=3.0, height=5.0)
    with pytest.warns(DeprecationWarning, match=rf"Torus2D.{old}"):
        assert getattr(torus, old) == getattr(torus, new)


def test_the_old_side_keyword_works_on_both_rectangle_builders():
    """Both spellings at once, so both warnings have to be expected."""
    with pytest.warns(DeprecationWarning, match="L1|L2"):
        rectangle = hp.FundamentalDomain.rectangle(L1=3.0, L2=5.0)
    with pytest.warns(DeprecationWarning, match="L1|L2"):
        bottle = hp.FundamentalDomain.klein_bottle(L1=3.0, L2=5.0)
    assert rectangle.volume == pytest.approx(15.0)
    assert bottle.volume == pytest.approx(15.0)


def test_an_unknown_keyword_still_raises():
    """Or a typo would be swallowed and the default silently used."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.Torus2D(widht=3.0)


def test_the_new_side_names_do_not_warn(recwarn):
    hp.Torus2D(width=3.0, height=5.0)
    hp.FundamentalDomain.rectangle(3.0, 5.0)
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# Retired parameters
# ---------------------------------------------------------------------------


def test_n_images_on_a_fundamental_domain_warns_and_still_works():
    """It tuned a truncation that now bounds itself, so nothing replaces it."""
    square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    pairings = [
        [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
    ]
    with pytest.warns(DeprecationWarning, match="n_images.*removed in the-hawkes-package 0.5.0"):
        domain = hp.FundamentalDomain(square, pairings, n_images=4)
    assert domain.n_images == 4
    assert len(domain.orbit([0.0, 0.0], 2)) > 1


def test_not_passing_n_images_does_not_warn(recwarn):
    hp.FundamentalDomain.hexagon(1.0)
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# What 0.4.0 removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REMOVED)
def test_the_removed_names_are_gone(name):
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


def test_the_deprecation_helpers_that_lost_their_callers_are_gone():
    """An unused deprecation helper is worse than none: it reads as supported."""
    from hawkes_package import _deprecation

    assert not hasattr(_deprecation, "DeprecatedAlias")
    assert not hasattr(_deprecation, "deprecated_module_getattr")


def test_an_unknown_top_level_attribute_raises_without_warning(recwarn):
    with pytest.raises(AttributeError):
        hp.does_not_exist
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# Documentation cannot drift from behaviour
# ---------------------------------------------------------------------------


def test_every_documented_rename_is_real(process):
    """The rename table in docs/migration.md is generated from these lists.

    Checked on an *instance*: ``n_simulated`` is set in ``__init__``, so the
    class carries only the deprecated spelling, which is the descriptor.
    """
    for old, new, version in RENAMES:
        assert version == "0.5.0"
        assert hasattr(hp.HawkesProcess, old)
        assert hasattr(process, new)
    for old, new in SIDE_RENAMES:
        assert hasattr(hp.Torus2D, old)
        assert hasattr(hp.Torus2D(), new)


def test_every_documented_retirement_names_its_version():
    for _what, _why, version in RETIREMENTS:
        assert version == "0.5.0"


def test_everything_still_deprecated_names_one_removal_version():
    """`REMOVED_IN` is the single source for the date, and it has moved on."""
    from hawkes_package._deprecation import REMOVED_IN

    assert REMOVED_IN == "0.5.0"
    assert {version for _, _, version in RENAMES} == {REMOVED_IN}
