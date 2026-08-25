#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial kernel utilities for spatio-temporal Hawkes processes.

Provides `make_periodic` which wraps an isotropic spatial kernel so that it
correctly accounts for the periodic structure of a SpatialDomain by summing
contributions from image points.
"""

import numpy as np
from .domains import SpatialDomain


def make_periodic(kernel_fn, domain: SpatialDomain, n_images: int = 3):
    """Return a periodised version of kernel_fn that sums over image points.

    For a domain with discrete translational symmetry (Circle, Torus2D), the
    true kernel between x and y is:

        K(x, y) = Σ_{offset ∈ images} kernel_fn(||x - y - offset||)

    For n_images=3 the sum covers offsets up to ±3 periods in each direction,
    which is sufficient for rapidly decaying kernels.

    Parameters
    ----------
    kernel_fn : callable(distance: float) -> float
        Isotropic kernel evaluated at a non-negative scalar distance.
    domain : SpatialDomain
        The spatial domain whose periodicity structure to respect.
    n_images : int
        Number of image periods in each direction.

    Returns
    -------
    callable(x, y) -> float
        Periodised kernel evaluated at two domain points.
    """
    from .domains import Circle, Torus2D

    if isinstance(domain, Circle):
        period = domain.volume

        def periodic_kernel(x, y):
            total = 0.0
            for n in range(-n_images, n_images + 1):
                offset = n * period
                total += kernel_fn(abs(float(x) - float(y) - offset))
            return total

        return periodic_kernel

    elif isinstance(domain, Torus2D):
        L1, L2 = domain.L1, domain.L2

        def periodic_kernel(x, y):
            x, y = np.asarray(x), np.asarray(y)
            total = 0.0
            for n1 in range(-n_images, n_images + 1):
                for n2 in range(-n_images, n_images + 1):
                    offset = np.array([n1 * L1, n2 * L2])
                    dist = float(np.linalg.norm(x - y - offset))
                    total += kernel_fn(dist)
            return total

        return periodic_kernel

    else:
        # Generic fallback: use the domain's own distance metric
        def periodic_kernel(x, y):
            return kernel_fn(domain.distance(x, y))

        return periodic_kernel
