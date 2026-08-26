"""Spatio-temporal Hawkes processes and the spatial domains they live on.

Note
----
``Spatio_Temporal_Hawkes_Process`` is deliberately **not** exported from this
subpackage. Historically the name meant the domain-aware class here but the
legacy periodic-interval class at top level — two different algorithms behind
one identifier. Accessing it from this module now raises :class:`AttributeError`
rather than silently resolving to whichever class the import path happens to
pick. Use :class:`SpatioTemporalHawkesProcess` or, for the frozen legacy
implementation, ``hawkes_package.spatio_temporal.legacy``.
"""

from ..mcmc import mcmc_sampler
from .domains import Circle, FundamentalDomain, SpatialDomain, Torus2D
from .kernels import make_periodic
from .legacy import LegacySpatioTemporalHawkesProcess
from .process import SpatioTemporalHawkesProcess

__all__ = [
    "Circle",
    "FundamentalDomain",
    "LegacySpatioTemporalHawkesProcess",
    "SpatialDomain",
    "SpatioTemporalHawkesProcess",
    "Torus2D",
    "make_periodic",
    "mcmc_sampler",
]
