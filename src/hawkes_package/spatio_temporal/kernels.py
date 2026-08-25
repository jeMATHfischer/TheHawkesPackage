"""Spatial kernel utilities for spatio-temporal Hawkes processes.

Provides :func:`make_periodic`, which wraps an isotropic spatial kernel so it
respects the periodic structure of a :class:`SpatialDomain` by summing the
contributions of image points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._numerics import as_float, as_point
from .domains import SpatialDomain

__all__ = ["PairwiseKernel", "make_periodic"]


class PairwiseKernel:
    """A spatial kernel that consumes both endpoints rather than a distance.

    :class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`
    normally reduces a pair of points to a geodesic distance before calling
    `spatial`. An image sum cannot be written that way -- on a torus the
    geodesic distance does not determine the lattice sum -- so such a kernel
    declares itself by carrying ``pairwise = True`` and is then called with the
    two points.

    Any callable may opt in the same way; it need not be an instance of this
    class.
    """

    pairwise = True

    def __init__(self, fn: Callable[[Any, Any], Any], /) -> None:
        self._fn = fn

    def __call__(self, x: Any, y: Any) -> float:
        """Evaluate the kernel between two domain points."""
        return as_float(self._fn(x, y))


KernelFn = Callable[[Any], Any]


def make_periodic(kernel_fn: KernelFn, domain: SpatialDomain, n_images: int = 3) -> PairwiseKernel:
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
            # Reduce to the canonical offset first. Summing images about a raw
            # difference leaves the nearest image outside the window once
            # |x - y| exceeds n_images * period, so the kernel would decay to
            # zero instead of staying periodic.
            offset = (as_point(x, 1)[0] - as_point(y, 1)[0] + period / 2) % period - period / 2
            total = 0.0
            for n in range(-n_images, n_images + 1):
                total += as_float(kernel_fn(abs(offset - n * period)))
            return total

        return PairwiseKernel(circle_kernel)

    if isinstance(domain, Torus2D):
        length_1, length_2 = domain.L1, domain.L2

        periods = np.array([length_1, length_2])

        def torus_kernel(x: Any, y: Any) -> float:
            # Canonical offset per axis, for the same reason as on the circle.
            delta = as_point(x, 2) - as_point(y, 2)
            delta = (delta + periods / 2) % periods - periods / 2
            total = 0.0
            for n1 in range(-n_images, n_images + 1):
                for n2 in range(-n_images, n_images + 1):
                    offset = np.array([n1 * length_1, n2 * length_2])
                    total += as_float(kernel_fn(float(np.linalg.norm(delta - offset))))
            return total

        return PairwiseKernel(torus_kernel)

    def generic_kernel(x: Any, y: Any) -> float:
        return as_float(kernel_fn(domain.distance(x, y)))

    return PairwiseKernel(generic_kernel)
