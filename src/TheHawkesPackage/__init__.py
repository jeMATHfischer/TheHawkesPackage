"""Deprecated import shim for the old ``TheHawkesPackage`` import name.

Importing this package emits a :class:`DeprecationWarning`. Every attribute and
submodule is forwarded to :mod:`hawkes_package`; the objects are *identical*
(same module objects, same classes), so ``isinstance`` checks are unaffected.

This shim is present in 0.2.x and 0.3.x and removed in 0.4.0. Migrate with::

    import hawkes_package as hp        # instead of: import TheHawkesPackage

Implementation note
-------------------
Two mechanisms are needed, and neither alone suffices:

* ``sys.modules`` aliasing handles ``import TheHawkesPackage.MCMC_sampler`` and
  ``from TheHawkesPackage.spatio_temporal.domains import Circle`` — a PEP 562
  module ``__getattr__`` never fires for those.
* ``__getattr__`` forwarding handles plain attribute access. The aliased modules
  are deliberately *not* bound into ``globals()``: in the original layout
  ``TheHawkesPackage.ExponentialHawkes`` resolved to the *class* (the
  ``from .X import X`` in ``__init__`` shadowed the submodule attribute), and
  routing through ``getattr(_hp, ...)`` reproduces that exactly.
"""

import importlib
import sys
import warnings
from typing import Any, List

import hawkes_package as _hp

__version__ = _hp.__version__
__all__ = list(_hp.__all__)

_MODULE_ALIASES = {
    "ExponentialHawkes": "hawkes_package.exponential",
    "MonotoneKernelHawkes": "hawkes_package.monotone",
    "BellShapeHawkes": "hawkes_package.bell_shape",
    "MCMC_sampler": "hawkes_package.mcmc",
    "SpatioTemporal_Hawkes_Monotone": "hawkes_package.spatio_temporal.legacy",
    "spatio_temporal": "hawkes_package.spatio_temporal",
    "spatio_temporal.domains": "hawkes_package.spatio_temporal.domains",
    "spatio_temporal.kernels": "hawkes_package.spatio_temporal.kernels",
    "spatio_temporal.process": "hawkes_package.spatio_temporal.process",
    "spatio_temporal.legacy": "hawkes_package.spatio_temporal.legacy",
    "spatio_temporal.sampler": "hawkes_package.mcmc",
}

warnings.warn(
    "The import name 'TheHawkesPackage' is deprecated and will be removed in "
    "the-hawkes-package 0.4.0; use 'import hawkes_package' instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _old, _new in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_old}"] = importlib.import_module(_new)
del _old, _new


def __getattr__(name: str) -> Any:
    """Forward attribute access to :mod:`hawkes_package`."""
    try:
        return getattr(_hp, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> List[str]:
    """List forwarded attributes and aliased submodules."""
    return sorted(set(__all__) | set(_MODULE_ALIASES) | {"__version__"})
