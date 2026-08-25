"""Every deprecated name must warn, still work, and forward to the same object.

The global ``filterwarnings = ["error"]`` in pyproject.toml means an
*accidental* deprecation path anywhere fails the suite; ``pytest.warns`` below
opts the intentional ones back in.
"""

import importlib
import subprocess
import sys

import numpy as np
import pytest

import hawkes_package as hp

ALIASES = ["propagate_by_amount", "propagate_by_k_events", "propogate_by_amount"]

#: (old name, new name, removal version) — kept in lockstep with docs/migration.md.
RENAMES = [
    ("propagate_by_amount", "simulate", "0.4.0"),
    ("propagate_by_k_events", "simulate", "0.4.0"),
    ("propogate_by_amount", "simulate", "0.4.0"),
    ("Spatio_Temporal_Hawkes_Process", "LegacySpatioTemporalHawkesProcess", "0.4.0"),
]


def _processes(exp_kernel, triangular_kernel, legacy_kernels):
    base, spatial, temporal = legacy_kernels
    return {
        "ExponentialHawkes": hp.ExponentialHawkes(np.array([1.0, 0.4, 2.0]), rng=0),
        "MonotoneKernelHawkes": hp.MonotoneKernelHawkes(exp_kernel, rng=0),
        "BellShapeHawkes": hp.BellShapeHawkes(triangular_kernel, rng=0),
        "SpatioTemporalHawkesProcess": hp.SpatioTemporalHawkesProcess(
            base,
            lambda d: max(0.0, 1 - d / np.pi),
            exp_kernel,
            domain=hp.Circle(),
            monotone_temporal_kernel=True,
            rng=0,
        ),
        "LegacySpatioTemporalHawkesProcess": hp.LegacySpatioTemporalHawkesProcess(
            base, spatial, temporal, rng=0
        ),
    }


@pytest.fixture
def processes(exp_kernel, triangular_kernel, legacy_kernels):
    return _processes(exp_kernel, triangular_kernel, legacy_kernels)


# ---------------------------------------------------------------------------
# Method aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize(
    "cls_name",
    [
        "ExponentialHawkes",
        "MonotoneKernelHawkes",
        "BellShapeHawkes",
        "SpatioTemporalHawkesProcess",
        "LegacySpatioTemporalHawkesProcess",
    ],
)
def test_alias_warns_and_forwards(processes, cls_name, alias):
    proc = processes[cls_name]
    with pytest.warns(DeprecationWarning, match=rf"{alias}\(\).*use .*simulate\(\)"):
        getattr(proc, alias)(2)
    assert proc.Sim_num == 2


def test_alias_names_the_removal_version(processes):
    with pytest.warns(DeprecationWarning, match="removed in the-hawkes-package 0.4.0"):
        processes["MonotoneKernelHawkes"].propagate_by_amount(1)


def test_alias_is_accessible_on_the_class(processes):
    """Class-level access must not blow up (it did not go through an instance)."""
    assert callable(hp.MonotoneKernelHawkes.propagate_by_amount)


def test_canonical_method_does_not_warn(processes, recwarn):
    processes["MonotoneKernelHawkes"].simulate(2)
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------------------------------------------------------
# The renamed legacy class
# ---------------------------------------------------------------------------


def test_top_level_legacy_name_warns_and_is_the_legacy_class():
    """The top-level name must keep meaning the LEGACY class.

    Pointing it at the domain-aware class would silently change both the
    algorithm and the shape of `Events` for existing callers.
    """
    with pytest.warns(DeprecationWarning, match="LegacySpatioTemporalHawkesProcess"):
        cls = hp.Spatio_Temporal_Hawkes_Process
    assert cls is hp.LegacySpatioTemporalHawkesProcess


def test_ambiguous_subpackage_name_is_gone():
    """In the subpackage the same name used to mean the *new* class.

    It is removed rather than repointed, so the ambiguity surfaces as an
    AttributeError instead of a silent change of meaning.
    """
    import hawkes_package.spatio_temporal as st

    with pytest.raises(AttributeError):
        st.Spatio_Temporal_Hawkes_Process


def test_unknown_top_level_attribute_raises_without_warning(recwarn):
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        hp.does_not_exist
    assert [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)] == []


def test_legacy_space_keyword_warns(legacy_kernels):
    base, spatial, temporal = legacy_kernels
    with pytest.warns(DeprecationWarning, match=r"Space=\.\.\..*use.*space=\.\.\."):
        p = hp.LegacySpatioTemporalHawkesProcess(base, spatial, temporal, Space=[-1.0, 1.0], rng=0)
    assert p.space == (-1.0, 1.0)


def test_legacy_space_attribute_warns(legacy_kernels):
    base, spatial, temporal = legacy_kernels
    p = hp.LegacySpatioTemporalHawkesProcess(base, spatial, temporal, rng=0)
    with pytest.warns(DeprecationWarning, match=r"\.Space.*use.*\.space"):
        assert p.Space == p.space


def test_unexpected_keyword_still_raises(legacy_kernels):
    base, spatial, temporal = legacy_kernels
    with pytest.raises(TypeError, match="unexpected keyword"):
        hp.LegacySpatioTemporalHawkesProcess(base, spatial, temporal, nonsense=1)


# ---------------------------------------------------------------------------
# The TheHawkesPackage import shim
# ---------------------------------------------------------------------------


def _purge_shim():
    """Drop the shim from sys.modules so a re-import actually re-executes it."""
    for name in [m for m in list(sys.modules) if m.split(".")[0] == "TheHawkesPackage"]:
        del sys.modules[name]


def test_importing_the_shim_warns():
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="use 'import hawkes_package' instead"):
        importlib.import_module("TheHawkesPackage")


def test_shim_forwards_attributes_to_the_same_objects():
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        shim = importlib.import_module("TheHawkesPackage")
    assert shim.ExponentialHawkes is hp.ExponentialHawkes
    assert shim.Circle is hp.Circle
    assert shim.mcmc_sampler is hp.mcmc_sampler
    assert shim.__version__ == hp.__version__


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [
        ("TheHawkesPackage.MCMC_sampler", "hawkes_package.mcmc"),
        ("TheHawkesPackage.ExponentialHawkes", "hawkes_package.exponential"),
        ("TheHawkesPackage.MonotoneKernelHawkes", "hawkes_package.monotone"),
        ("TheHawkesPackage.BellShapeHawkes", "hawkes_package.bell_shape"),
        (
            "TheHawkesPackage.SpatioTemporal_Hawkes_Monotone",
            "hawkes_package.spatio_temporal.legacy",
        ),
        ("TheHawkesPackage.spatio_temporal", "hawkes_package.spatio_temporal"),
        (
            "TheHawkesPackage.spatio_temporal.domains",
            "hawkes_package.spatio_temporal.domains",
        ),
        ("TheHawkesPackage.spatio_temporal.sampler", "hawkes_package.mcmc"),
    ],
)
def test_shim_submodule_imports_resolve_to_the_same_module(old_path, new_path):
    """A PEP 562 __getattr__ never fires for `import pkg.sub`, so the shim also
    aliases sys.modules. Both mechanisms are required."""
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        importlib.import_module("TheHawkesPackage")
    assert importlib.import_module(old_path) is importlib.import_module(new_path)


def test_shim_dotted_from_import_yields_identical_classes():
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        importlib.import_module("TheHawkesPackage")
    mod = importlib.import_module("TheHawkesPackage.spatio_temporal.domains")
    assert mod.Circle is hp.Circle


def test_shim_attribute_is_the_class_not_the_module():
    """`THP.ExponentialHawkes` has always been the class; that must not change.

    Binding the aliased submodules into globals() would have shadowed it with
    the module object and broken every existing notebook.
    """
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        shim = importlib.import_module("TheHawkesPackage")
    assert isinstance(shim.ExponentialHawkes, type)


def test_shim_dir_lists_names_and_submodules():
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        shim = importlib.import_module("TheHawkesPackage")
    listed = dir(shim)
    assert "ExponentialHawkes" in listed
    assert "spatio_temporal" in listed
    assert "__version__" in listed
    assert listed == sorted(listed)


def test_shim_unknown_attribute_raises_attribute_error():
    _purge_shim()
    with pytest.warns(DeprecationWarning, match="import hawkes_package"):
        shim = importlib.import_module("TheHawkesPackage")
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        shim.nope


def test_canonical_import_emits_no_warning():
    """`import hawkes_package` under -W error must succeed in a clean process."""
    subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import hawkes_package"],
        check=True,
        capture_output=True,
    )


def test_shim_import_fails_under_werror():
    """Conversely, the shim's warning must be real enough to trip -W error."""
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import TheHawkesPackage"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "DeprecationWarning" in result.stderr


# ---------------------------------------------------------------------------
# Documentation cannot drift from behaviour
# ---------------------------------------------------------------------------


def test_every_documented_rename_is_real():
    """The rename table in docs/migration.md is generated from this list."""
    for old, _new, version in RENAMES:
        assert version == "0.4.0"
        assert old in ALIASES or hasattr(hp, "LegacySpatioTemporalHawkesProcess")
