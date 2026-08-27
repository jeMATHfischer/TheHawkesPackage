"""Simulation of self-exciting (Hawkes) point processes.

Temporal processes
------------------
:class:`ExponentialHawkes`, :class:`MonotoneKernelHawkes`,
:class:`BellShapeHawkes`

Spatio-temporal processes
-------------------------
:class:`SpatioTemporalHawkesProcess` (domain-aware) and
:class:`LegacySpatioTemporalHawkesProcess` (frozen, removed in 0.4.0)

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

from ._deprecation import deprecated_module_getattr
from .base import HawkesProcess, TemporalHawkesProcess
from .bell_shape import BellShapeHawkes
from .exponential import ExponentialHawkes
from .mcmc import mcmc_sampler
from .monotone import MonotoneKernelHawkes
from .spatio_temporal import (
    Circle,
    FundamentalDomain,
    LegacySpatioTemporalHawkesProcess,
    SpatialDomain,
    SpatioTemporalHawkesProcess,
    Sphere,
    Torus2D,
    make_periodic,
)

__version__ = "0.3.0"

__all__ = [
    "BellShapeHawkes",
    "Circle",
    "ExponentialHawkes",
    "FundamentalDomain",
    "HawkesProcess",
    "LegacySpatioTemporalHawkesProcess",
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

# Deprecated top-level names. Deliberately absent from __all__ and never bound
# as globals, so PEP 562 __getattr__ fires and warns on first access.
#
# `Spatio_Temporal_Hawkes_Process` resolves to the LEGACY class: that is what
# the name has always meant at top level, and silently switching callers to the
# domain-aware class would change both the algorithm and the shape of `Events`.
__getattr__ = deprecated_module_getattr(
    {
        "Spatio_Temporal_Hawkes_Process": (
            "LegacySpatioTemporalHawkesProcess",
            LegacySpatioTemporalHawkesProcess,
        ),
    },
    module=__name__,
)
