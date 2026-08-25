"""Simulation of self-exciting (Hawkes) point processes.

Temporal processes
------------------
:class:`~hawkes_package.exponential.ExponentialHawkes`,
:class:`~hawkes_package.monotone.MonotoneKernelHawkes`,
:class:`~hawkes_package.bell_shape.BellShapeHawkes`

Spatio-temporal processes
-------------------------
:class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`
(domain-aware) and
:class:`~hawkes_package.spatio_temporal.legacy.LegacySpatioTemporalHawkesProcess`

Spatial domains
---------------
:class:`~hawkes_package.spatio_temporal.domains.SpatialDomain`,
:class:`~hawkes_package.spatio_temporal.domains.Circle`,
:class:`~hawkes_package.spatio_temporal.domains.Torus2D`
"""

from .bell_shape import BellShapeHawkes
from .exponential import ExponentialHawkes
from .mcmc import mcmc_sampler
from .monotone import MonotoneKernelHawkes
from .spatio_temporal import (
    Circle,
    SpatialDomain,
    SpatioTemporalHawkesProcess,
    Torus2D,
    make_periodic,
)
from .spatio_temporal.legacy import Spatio_Temporal_Hawkes_Process

__version__ = "0.2.0"

__all__ = [
    "BellShapeHawkes",
    "Circle",
    "ExponentialHawkes",
    "MonotoneKernelHawkes",
    "SpatialDomain",
    "SpatioTemporalHawkesProcess",
    "Spatio_Temporal_Hawkes_Process",
    "Torus2D",
    "__version__",
    "make_periodic",
    "mcmc_sampler",
]
