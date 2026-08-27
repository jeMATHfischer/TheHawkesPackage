"""Helpers for emitting :class:`DeprecationWarning` from renamed public APIs.

Everything the package deprecates routes through here so the wording, the
removal version and the ``stacklevel`` stay consistent across call sites.

.. versionchanged:: 0.4.0
   ``DeprecatedAlias`` and ``deprecated_module_getattr`` are gone, along with
   the last names that used them. :class:`DeprecatedAttribute` replaces the
   first: an alias for a renamed *attribute* cannot be a method wrapper, which
   is all ``DeprecatedAlias`` could produce.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["DeprecatedAttribute", "warn_removed", "warn_renamed"]

#: Version in which everything currently deprecated is removed.
REMOVED_IN = "0.5.0"


def warn_renamed(
    old: str,
    new: str,
    *,
    removed_in: str = REMOVED_IN,
    stacklevel: int = 3,
) -> None:
    """Emit a uniform rename warning pointing from `old` to `new`.

    Parameters
    ----------
    old, new : str
        Human-readable names, e.g. ``"HawkesProcess.Events"``.
    removed_in : str
        Version in which `old` stops working.
    stacklevel : int
        Passed through to :func:`warnings.warn`.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in the-hawkes-package "
        f"{removed_in}; use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_removed(
    what: str,
    because: str,
    *,
    removed_in: str = REMOVED_IN,
    stacklevel: int = 3,
) -> None:
    """Warn that something is going away without being replaced by a new spelling.

    Distinct from :func:`warn_renamed`, which points at a new name. Some things
    are deprecated because they stopped being *needed* — a tuning parameter for
    a search that now bounds itself — and telling the caller to use a
    replacement would be wrong.

    Parameters
    ----------
    what : str
        Human-readable name of the thing going away.
    because : str
        Why it is no longer needed, phrased so the caller can act on it.
    removed_in : str
        Version in which `what` stops working.
    stacklevel : int
        Passed through to :func:`warnings.warn`.
    """
    warnings.warn(
        f"{what} is deprecated and will be removed in the-hawkes-package {removed_in}; {because}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class DeprecatedAttribute:
    """Descriptor exposing a deprecated spelling of a renamed attribute.

    Reads *and writes* forward to `target`, both warning. Assignment is not
    optional: ``process.Events = history`` is how a caller seeds a realisation
    before simulating, and dropping the setter would turn that from a
    deprecation into a silent shadowing — the assignment would bind a plain
    instance attribute over the descriptor and the simulator would never see it.

    The alias learns its own name from :meth:`__set_name__`, so declaring one
    costs a single line and the warning text cannot drift from reality.

    Parameters
    ----------
    target : str
        Name of the attribute this alias forwards to.
    removed_in : str
        Version in which the alias stops working.

    Examples
    --------
    >>> class Process:
    ...     def __init__(self):
    ...         self.events = []
    ...
    ...     Events = DeprecatedAttribute("events")
    """

    def __init__(self, target: str, *, removed_in: str = REMOVED_IN) -> None:
        self.target = target
        self.removed_in = removed_in
        self.public_name = target

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name this descriptor was bound to."""
        self.public_name = name
        self.__doc__ = (
            f"Deprecated alias for :attr:`~{owner.__module__}.{owner.__qualname__}.{self.target}`."
        )

    def _warn(self, owner: type) -> None:
        """Point from this spelling to the current one."""
        warn_renamed(
            f"{owner.__name__}.{self.public_name}",
            f"{owner.__name__}.{self.target}",
            removed_in=self.removed_in,
            stacklevel=4,
        )

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Warn, then read the attribute this one was renamed to."""
        if obj is None:
            # Class-level access: hand back the descriptor so `help()` and
            # `inspect` can see the docstring without a warning fired at import.
            return self
        self._warn(objtype if objtype is not None else type(obj))
        return getattr(obj, self.target)

    def __set__(self, obj: Any, value: Any) -> None:
        """Warn, then write through to the attribute this one was renamed to."""
        self._warn(type(obj))
        setattr(obj, self.target, value)
