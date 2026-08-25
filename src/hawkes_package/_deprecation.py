"""Helpers for emitting :class:`DeprecationWarning` from renamed public APIs.

Everything the package deprecates routes through here so the wording, the
removal version and the ``stacklevel`` stay consistent across call sites.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = ["DeprecatedAlias", "deprecated_module_getattr", "warn_renamed"]

#: Version in which everything currently deprecated is removed.
REMOVED_IN = "0.4.0"


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
        Human-readable names, e.g. ``"BellShapeHawkes.propagate_by_amount()"``.
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


class DeprecatedAlias:
    """Descriptor exposing a deprecated alias of another method.

    The alias learns its own name from :meth:`__set_name__`, so declaring one
    costs a single line and the warning text can never drift from reality.

    Parameters
    ----------
    target : str
        Name of the method this alias forwards to.
    removed_in : str
        Version in which the alias stops working.

    Examples
    --------
    >>> class Process:
    ...     def simulate(self, k):
    ...         return k
    ...     propagate_by_amount = DeprecatedAlias("simulate")
    """

    def __init__(self, target: str, *, removed_in: str = REMOVED_IN) -> None:
        self.target = target
        self.removed_in = removed_in
        self.public_name = target

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name this descriptor was bound to."""
        self.public_name = name
        self.__doc__ = (
            f"Deprecated alias for :meth:`~{owner.__module__}."
            f"{owner.__qualname__}.{self.target}`."
        )

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Callable[..., Any]:
        """Return a wrapper that warns and then forwards to the target method."""
        owner = objtype if objtype is not None else type(obj)
        bound = getattr(obj if obj is not None else owner, self.target)
        old = f"{owner.__name__}.{self.public_name}()"
        new = f"{owner.__name__}.{self.target}()"
        removed_in = self.removed_in

        @functools.wraps(bound)
        def _alias(*args: Any, **kwargs: Any) -> Any:
            warn_renamed(old, new, removed_in=removed_in, stacklevel=2)
            return bound(*args, **kwargs)

        _alias.__name__ = self.public_name
        _alias.__qualname__ = f"{owner.__qualname__}.{self.public_name}"
        _alias.__doc__ = self.__doc__
        return _alias


def deprecated_module_getattr(
    mapping: Dict[str, Tuple[str, Any]],
    *,
    module: str,
    removed_in: str = REMOVED_IN,
) -> Callable[[str], Any]:
    """Build a :pep:`562` ``__getattr__`` that warns for renamed module globals.

    Parameters
    ----------
    mapping : dict
        ``{old_name: (new_name, new_object)}``.
    module : str
        ``__name__`` of the calling module, used in the message and in the
        :class:`AttributeError` raised for unknown names.
    removed_in : str
        Version in which the old names stop working.

    Returns
    -------
    callable
        A function suitable for assignment to a module-level ``__getattr__``.
    """

    def __getattr__(name: str) -> Any:
        try:
            new_name, obj = mapping[name]
        except KeyError:
            raise AttributeError(f"module {module!r} has no attribute {name!r}") from None
        warn_renamed(
            f"{module}.{name}",
            f"{module}.{new_name}",
            removed_in=removed_in,
            stacklevel=3,
        )
        return obj

    return __getattr__
