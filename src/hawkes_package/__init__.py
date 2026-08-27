"""Simulation of self-exciting (Hawkes) point processes.

Temporal processes
------------------
:class:`ExponentialHawkes`, :class:`MonotoneKernelHawkes`,
:class:`BellShapeHawkes`

Spatio-temporal processes
-------------------------
:class:`SpatioTemporalHawkesProcess`

Spatial domains
---------------
:class:`SpatialDomain`, :class:`Circle`, :class:`Torus2D`, :class:`Sphere` and
:class:`FundamentalDomain` — a convex geodesic polygon with side pairings, which
between them reach **every closed surface**: the sphere and the projective plane,
the torus and the Klein bottle, and every orientable or non-orientable surface of
higher genus. Plus :func:`make_periodic` for summing a kernel over a domain's
image points.

Notes
-----
Every process takes ``rng=``, accepting ``None``, an ``int`` seed or an existing
:class:`numpy.random.Generator`. As of 0.2.0 :func:`numpy.random.seed` no longer
influences simulations.
"""

from .base import HawkesProcess, TemporalHawkesProcess
from .bell_shape import BellShapeHawkes
from .exponential import ExponentialHawkes
from .mcmc import mcmc_sampler
from .monotone import MonotoneKernelHawkes
from .spatio_temporal import (
    Circle,
    FundamentalDomain,
    SpatialDomain,
    SpatioTemporalHawkesProcess,
    Sphere,
    Torus2D,
    make_periodic,
)

__version__ = "0.4.0rc2"

__all__ = [
    "BellShapeHawkes",
    "Circle",
    "ExponentialHawkes",
    "FundamentalDomain",
    "HawkesProcess",
    "MonotoneKernelHawkes",
    "SpatialDomain",
    "SpatioTemporalHawkesProcess",
    "Sphere",
    "TemporalHawkesProcess",
    "Torus2D",
    "__version__",
    "make_periodic",
    "mcmc_sampler",
]
