"""Spatial kernel utilities for spatio-temporal Hawkes processes.

Provides :func:`make_periodic`, which wraps an isotropic spatial kernel so it
respects the periodic structure of a :class:`SpatialDomain` by summing the
contributions of image points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .domains import SpatialDomain

__all__ = ["make_periodic"]

KernelFn = Callable[[Any], Any]
PeriodicKernel = Callable[[Any, Any], float]


def make_periodic(kernel_fn: KernelFn, domain: SpatialDomain, n_images: int = 3) -> PeriodicKernel:
    r"""Return a periodised version of `kernel_fn` that sums over image points.

    For a domain with discrete translational symmetry (:class:`Circle`,
    :class:`Torus2D`) the true kernel between ``x`` and ``y`` is

    .. math::

        K(x, y) = \sum_{\text{offset} \in \text{images}}
                  \mathrm{kernel\_fn}\!\left( \| x - y - \text{offset} \| \right).

    With ``n_images=3`` the sum covers offsets up to three periods in each
    direction, which is ample for a rapidly decaying kernel.

    Parameters
    ----------
    kernel_fn : callable
        Isotropic kernel evaluated at a non-negative scalar distance.
    domain : SpatialDomain
        The domain whose periodicity structure to respect. A domain that is not
        a :class:`Circle` or :class:`Torus2D` falls back to a single evaluation
        at ``domain.distance(x, y)``.
    n_images : int
        Number of image periods in each direction.

    Returns
    -------
    callable
        The periodised kernel, taking two domain points.

    Examples
    --------
    >>> kernel = make_periodic(lambda d: np.exp(-(d**2)), Circle())
    >>> round(kernel(0.3, -1.2), 6)
    0.105399
    """
    # Imported here rather than at module scope: domains.py is imported by
    # process.py, which also imports this module.
    from .domains import Circle, Torus2D

    if isinstance(domain, Circle):
        period = domain.volume

        def circle_kernel(x: Any, y: Any) -> float:
            # Accept scalars or shape-(1,) arrays: process.py passes event
            # coordinates, which are always 1-element arrays.
            xs = float(np.ravel(x)[0])
            ys = float(np.ravel(y)[0])
            total = 0.0
            for n in range(-n_images, n_images + 1):
                total += float(kernel_fn(abs(xs - ys - n * period)))
            return total

        return circle_kernel

    if isinstance(domain, Torus2D):
        length_1, length_2 = domain.L1, domain.L2

        def torus_kernel(x: Any, y: Any) -> float:
            x, y = np.asarray(x), np.asarray(y)
            total = 0.0
            for n1 in range(-n_images, n_images + 1):
                for n2 in range(-n_images, n_images + 1):
                    offset = np.array([n1 * length_1, n2 * length_2])
                    total += float(kernel_fn(float(np.linalg.norm(x - y - offset))))
            return total

        return torus_kernel

    def generic_kernel(x: Any, y: Any) -> float:
        return float(kernel_fn(domain.distance(x, y)))

    return generic_kernel
