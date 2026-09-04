"""The optional extra must be genuinely optional, and must say so when it is missing.

Nothing here imports plotly, so all of it runs in every CI job -- including the
ones that never install the `[viz]` extra. That is the point: this file is what
a user without the extra actually meets.
"""

import subprocess
import sys

import pytest

import hawkes_package.viz as viz
from hawkes_package.viz import _plotly


def test_importing_viz_does_not_import_the_backend():
    """The extra must not become a runtime dependency by accident.

    Checked in a subprocess because another test in the same session may quite
    legitimately have imported plotly already, which would make an in-process
    check pass or fail for reasons unrelated to this module.
    """
    code = (
        "import sys, hawkes_package.viz; "
        "leaked = [m for m in sys.modules if m == 'plotly' or m.startswith('plotly.')]; "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_missing_backend_says_what_to_install(monkeypatch):
    """Setting the module to None in `sys.modules` makes a real import raise."""
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match=r"the-hawkes-package\[viz\]"):
        _plotly._backend()


def test_the_message_points_at_the_part_that_needs_nothing(monkeypatch):
    """Building the frames is numpy-only, and a user hitting this should be told."""
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match="intensity_frames"):
        _plotly._backend()


def test_all_is_sorted_and_unique():
    assert viz.__all__ == sorted(viz.__all__)
    assert len(viz.__all__) == len(set(viz.__all__))


@pytest.mark.parametrize("name", viz.__all__)
def test_every_exported_name_resolves(name):
    assert hasattr(viz, name), f"{name} is in viz.__all__ but missing from the module"
