#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain-aware spatio-temporal Hawkes process simulation.

The intensity is separable:

    λ(t, x | H_t) = μ(x) + Σ_{i: t_i < t} κ_t(t - t_i) · κ_s(d(x, x_i))

where d(·, ·) is the geodesic distance on `domain`.

Temporal thinning uses Ogata's algorithm; spatial coordinates are drawn by
Metropolis-Hastings on the conditional spatial density.
"""

import numpy as np
import random as rand
from scipy.integrate import quad
from scipy.optimize import fmin

from .domains import SpatialDomain, Circle
from .sampler import mcmc_sampler


class SpatioTemporalHawkesProcess:
    """Spatio-temporal Hawkes process on an arbitrary SpatialDomain.

    Parameters
    ----------
    base : callable(x) -> float
        Background intensity as a function of spatial coordinate.
    spatial : callable(distance) -> float
        Isotropic spatial kernel evaluated at a non-negative distance.
    temporal : callable(dt) -> float
        Temporal kernel evaluated at a non-negative time lag.
    domain : SpatialDomain
        Spatial domain (default: unit circle, equivalent to [0, 2π)).
    monotone_temporal_kernel : bool
        Set True if the temporal kernel is monotone decreasing (enables a
        tighter upper bound and avoids the fmin search for the extremum).
    """

    def __init__(self, base, spatial, temporal,
                 domain: SpatialDomain = None,
                 monotone_temporal_kernel: bool = False):
        self.base = base
        self.spatial = spatial
        self.temporal = temporal
        self.domain = domain if domain is not None else Circle()
        self.monotone_temporal_kernel = monotone_temporal_kernel

        # Events stored as (ndim+1, n) array: row 0 = times, rows 1.. = coords
        ndim = self.domain.bounds.shape[0]
        seed_coord = np.zeros(ndim)
        self.Events = np.vstack([np.array([[0.0]]),
                                 seed_coord.reshape(-1, 1)])
        self.PoissEvent = np.array([])
        self.Sim_num = 0
        self._rng = np.random.default_rng()

        if not monotone_temporal_kernel:
            self.temporal_extremum = float(
                fmin(lambda t: -self.temporal(t), 0, disp=False)
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _past_event_indices(self, t: float):
        """Column indices of events strictly before time t (excluding seed)."""
        return [i for i in range(self.Events.shape[1])
                if 0 < self.Events[0, i] < t]

    def _dist_temporal(self, t: float) -> np.ndarray:
        idx = self._past_event_indices(t)
        return np.array([self.temporal(t - self.Events[0, i]) for i in idx])

    def _dist_spatial(self, x, t: float) -> np.ndarray:
        idx = self._past_event_indices(t)
        return np.array([self.spatial(self.domain.distance(x, self.Events[1:, i]))
                         for i in idx])

    def _full_intensity(self, x, t: float) -> float:
        contrib = np.multiply(self._dist_temporal(t), self._dist_spatial(x, t)).sum()
        return max(0.0, float(self.base(x)) + contrib)

    def _integrated_intensity(self, t: float) -> float:
        """Intensity integrated over the spatial domain at time t."""
        bounds = self.domain.bounds
        if bounds.shape[0] == 1:
            val, _ = quad(lambda x: self._full_intensity(x, t),
                          bounds[0, 0], bounds[0, 1])
        else:
            # For higher dimensions: Monte Carlo estimate over domain volume
            n_mc = 500
            rng = self._rng
            pts = np.column_stack([rng.uniform(bounds[d, 0], bounds[d, 1], n_mc)
                                   for d in range(bounds.shape[0])])
            val = self.domain.volume * np.mean(
                [self._full_intensity(pts[j], t) for j in range(n_mc)]
            )
        return float(val)

    def _upper_bound(self, t: float) -> float:
        base = self._integrated_intensity(t)
        if self.monotone_temporal_kernel:
            return base
        # Bell-shaped kernel: add headroom for kernel still rising
        last_event_time = self.Events[0, -1]
        if t - last_event_time < self.temporal_extremum:
            return base + self.temporal(self.temporal_extremum)
        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate_by_amount(self, k: int):
        """Simulate k new events and append them to self.Events."""
        for _ in range(k):
            self.PoissEvent = np.append(self.PoissEvent, rand.expovariate(1))

        poiss_times = np.cumsum(self.PoissEvent)

        for _time in poiss_times[self.Sim_num:]:
            T = float(self.Events[0, -1])

            # --- Temporal thinning ---
            while True:
                upper_bd = self._upper_bound(T)
                u = self._rng.uniform()
                tau = -np.log(u) / upper_bd
                T += tau
                s = self._rng.uniform()
                if s <= self._integrated_intensity(T) / upper_bd:
                    event_time = T
                    break

            # --- Spatial sampling via MCMC ---
            def spatial_density(x):
                return self._full_intensity(x, event_time)

            event_coord, _ = mcmc_sampler(spatial_density, self.domain.bounds,
                                          return_diagnostics=True)
            event_coord = self.domain.wrap(event_coord)

            new_event = np.vstack([np.array([[event_time]]),
                                   np.asarray(event_coord).reshape(-1, 1)])
            self.Events = np.append(self.Events, new_event, axis=1)

        if self.Sim_num == 0:
            self.Events = self.Events[:, 1:]

        self.Sim_num += k

    def simulate(self, k: int):
        self.propagate_by_amount(k)
