"""The declared public surface must stay coherent with what is installed."""

import importlib
import importlib.metadata as md
import importlib.util
import re
from pathlib import Path

import pytest

import hawkes_package as hp
import hawkes_package.spatio_temporal as st

DISTRIBUTION = "the-hawkes-package"

# PEP 440, restricted to the release/pre-release forms this project uses.
PEP440 = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")

MODULES = [
    "hawkes_package",
    "hawkes_package.base",
    "hawkes_package.bell_shape",
    "hawkes_package.exponential",
    "hawkes_package.mcmc",
    "hawkes_package.monotone",
    "hawkes_package._deprecation",
    "hawkes_package.spatio_temporal",
    "hawkes_package.spatio_temporal.domains",
    "hawkes_package.spatio_temporal._model",
    "hawkes_package.spatio_temporal._gluing",
    "hawkes_package.spatio_temporal._integration",
    "hawkes_package.spatio_temporal.kernels",
    "hawkes_package.spatio_temporal.process",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("name", hp.__all__)
def test_every_exported_name_resolves(name):
    assert hasattr(hp, name), f"{name} is in __all__ but missing from the module"


@pytest.mark.parametrize(("mod", "label"), [(hp, "hawkes_package"), (st, "spatio_temporal")])
def test_all_is_sorted_and_unique(mod, label):
    assert mod.__all__ == sorted(mod.__all__), f"{label}.__all__ is not sorted"
    assert len(mod.__all__) == len(set(mod.__all__)), f"{label}.__all__ has duplicates"


def test_star_import_matches_all():
    namespace = {}
    exec("from hawkes_package import *", namespace)
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(hp.__all__) - {"__version__"}


# ---------------------------------------------------------------------------
# Distribution metadata
# ---------------------------------------------------------------------------


def test_version_is_pep440():
    assert PEP440.match(hp.__version__), f"{hp.__version__!r} is not a PEP 440 version"


def test_version_matches_installed_metadata():
    """The single source of truth in __init__.py must be what hatchling shipped."""
    assert md.version(DISTRIBUTION) == hp.__version__


def test_requires_python_floor_matches_the_ci_matrix():
    """The floor here and the oldest interpreter in ci.yml must agree.

    If one moves without the other, CI either tests an unsupported version or
    silently stops covering the declared minimum.
    """
    requires = md.metadata(DISTRIBUTION)["Requires-Python"]
    assert requires == ">=3.10"


def test_package_ships_type_information():
    """PEP 561: without py.typed beside the package, downstream mypy ignores us.

    Checked on the filesystem rather than through ``metadata.files()``, which
    lists only the ``.pth`` redirect under an editable install. That the file
    also lands *in the wheel* is asserted by the ``build`` job in CI.
    """
    assert (Path(hp.__file__).parent / "py.typed").is_file()


def test_only_one_top_level_package_ships():
    """0.4.0 removed the second one.

    ``TheHawkesPackage`` was a real top-level package in the wheel for two
    releases, not merely an attribute alias, so its removal is a packaging
    change as much as a source one: the ``packages`` entry, the coverage source
    and two tool overrides all had to go with it.
    """
    assert importlib.util.find_spec("TheHawkesPackage") is None
