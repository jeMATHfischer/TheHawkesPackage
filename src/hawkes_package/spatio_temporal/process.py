#!/usr/bin/env python3
r"""Domain-aware spatio-temporal Hawkes process simulation.

The intensity is separable,

.. math::

    \lambda(t, x \mid H_t) = \mu(x)
        + \sum_{t_i < t} \kappa_t(t - t_i)\, \kappa_s\!\left( d(x, x_i) \right),

where :math:`d(\cdot, \cdot)` is the geodesic distance on the domain.

Event times come from Ogata's thinning applied to the space-integrated
intensity; the location of each accepted event is then drawn by
Metropolis-Hastings from the conditional spatial density at that time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import quad

from .._numerics import as_float, locate_peak
from ..base import HawkesProcess, SeedLike, _stalled_message
from ..mcmc import mcmc_sampler
from .domains import Circle, SpatialDomain

__all__ = ["SpatioTemporalHawkesProcess"]

#: Sample count for the Monte Carlo spatial integral used when ndim >= 2.
_MC_SAMPLES = 500


class SpatioTemporalHawkesProcess(HawkesProcess):
    """Spatio-temporal Hawkes process on an arbitrary :class:`SpatialDomain`.

    Parameters
    ----------
    base : callable
        Background intensity ``mu(x)`` as a function of spatial coordinate.
    spatial : callable
        Isotropic spatial kernel evaluated at a non-negative distance.
    temporal : callable
        Temporal kernel evaluated at a non-negative time lag.
    domain : SpatialDomain, optional
        Spatial domain. Defaults to the unit :class:`Circle`.
    monotone_temporal_kernel : bool
        Set ``True`` when `temporal` is monotone decreasing. This permits a
        tighter thinning bound and skips the numerical search for the kernel's
        maximum.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

        .. versionadded:: 0.2.0

    Attributes
    ----------
    Events : numpy.ndarray
        Shape ``(ndim + 1, n)``. Row 0 holds event times, rows 1.. hold
        coordinates.

    Examples
    --------
    >>> process = SpatioTemporalHawkesProcess(
    ...     base=lambda x: 0.5,
    ...     spatial=lambda d: max(0.0, 1 - d / np.pi),
    ...     temporal=lambda dt: 0.9 * np.exp(-5 * dt),
    ...     domain=Circle(),
    ...     monotone_temporal_kernel=True,
    ...     rng=0,
    ... )
    >>> process.simulate(5)
    >>> process.Events.shape
    (2, 5)
    """

    def __init__(
        self,
        base: Callable[[Any], float],
        spatial: Callable[[float], float],
        temporal: Callable[[float], float],
        domain: SpatialDomain | None = None,
        monotone_temporal_kernel: bool = False,
        rng: SeedLike = None,
        *,
        peak_lag: float | None = None,
        peak_value: float | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.base = base
        self.spatial = spatial
        self.temporal = temporal
        self.domain = domain if domain is not None else Circle()
        self.monotone_temporal_kernel = monotone_temporal_kernel

        ndim = self.domain.bounds.shape[0]
        # Empty: `Events` holds only real events at every moment. See
        # TemporalHawkesProcess.__init__ for why the bootstrap column is gone.
        self.Events = np.empty((ndim + 1, 0), dtype=float)

        if not monotone_temporal_kernel:
            if peak_lag is None:
                located = locate_peak(temporal, name="temporal kernel")
                self.temporal_extremum, self.temporal_peak = located.lag, located.value
            else:
                self.temporal_extremum = float(peak_lag)
                self.temporal_peak = as_float(
                    peak_value if peak_value is not None else temporal(self.temporal_extremum)
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _past_event_indices(self, t: float, inclusive: bool = False) -> list[int]:
        """Column indices of events before time `t`.

        `inclusive` also admits an event at exactly `t`, which the thinning
        bound needs: at the start of a step `t` *is* the most recent event time.

        There is no ``0 < ...`` guard: it used to hide the bootstrap column, but
        it also silently discarded any user-supplied event at a non-positive
        time for the whole life of the object.
        """
        if inclusive:
            return [i for i in range(self.Events.shape[1]) if self.Events[0, i] <= t]
        return [i for i in range(self.Events.shape[1]) if self.Events[0, i] < t]

    def _temporal_factors(self, t: float, bound: bool = False) -> np.ndarray:
        """Temporal kernel factors at time `t`, or their future suprema.

        With ``bound=True`` each event contributes the largest value its kernel
        can still reach at any later time -- the peak if it has not yet peaked,
        its current value if it has. For a monotone-decreasing kernel the two
        coincide, so only the inclusive event set differs.
        """
        idx = self._past_event_indices(t, inclusive=bound)
        lags = np.array([t - self.Events[0, i] for i in idx])
        if lags.size == 0:
            return lags
        values = np.array([float(self.temporal(lag)) for lag in lags])
        if bound and not self.monotone_temporal_kernel:
            values = np.where(lags < self.temporal_extremum, self.temporal_peak, values)
        return values

    def _dist_temporal(self, t: float) -> np.ndarray:
        return self._temporal_factors(t)

    def _dist_spatial(self, x: Any, t: float, bound: bool = False) -> np.ndarray:
        idx = self._past_event_indices(t, inclusive=bound)
        return np.array([self.spatial(self.domain.distance(x, self.Events[1:, i])) for i in idx])

    def _full_intensity(self, x: Any, t: float, bound: bool = False) -> float:
        contrib = np.multiply(
            self._temporal_factors(t, bound=bound), self._dist_spatial(x, t, bound=bound)
        ).sum()
        return max(0.0, float(self.base(x)) + float(contrib))

    def _integrated_intensity(self, t: float, bound: bool = False) -> float:
        """Intensity integrated over the spatial domain at time `t`."""
        bounds = self.domain.bounds
        if bounds.shape[0] == 1:
            val, _ = quad(
                lambda x: self._full_intensity(x, t, bound=bound), bounds[0, 0], bounds[0, 1]
            )
        else:
            # No product quadrature in higher dimensions: Monte Carlo instead.
            pts = np.column_stack(
                [
                    self.rng.uniform(bounds[d, 0], bounds[d, 1], _MC_SAMPLES)
                    for d in range(bounds.shape[0])
                ]
            )
            val = self.domain.volume * float(
                np.mean([self._full_intensity(pts[j], t, bound=bound) for j in range(_MC_SAMPLES)])
            )
        return float(val)

    def _upper_bound(self, t: float) -> float:
        # Integrating the per-event suprema dominates the integrated intensity
        # at every later time, for monotone and bell-shaped kernels alike.
        return self._integrated_intensity(t, bound=True)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _propagate(self, k: int) -> None:
        for done in range(k):
            t = float(self.Events[0, -1]) if self.Events.shape[1] else 0.0

            # --- Temporal thinning on the space-integrated intensity ---
            while True:
                bound = self._upper_bound(t)
                if not bound > 0:
                    raise RuntimeError(
                        f"Non-positive thinning bound M={bound!r} at t={t!r}; the "
                        "background intensity must be positive somewhere."
                    )
                advanced = t + self.rng.exponential() / bound
                if not advanced > t:
                    raise RuntimeError(_stalled_message(t, bound, done, k))
                t = advanced
                if self.rng.uniform() * bound <= self._integrated_intensity(t):
                    break
            event_time = t

            # --- Spatial coordinate from the conditional density at that time ---
            coord = mcmc_sampler(
                lambda x: self._full_intensity(x, event_time),  # noqa: B023
                self.domain.bounds,
                seed=self.rng,
            )
            coord = self.domain.wrap(coord)

            new_event = np.vstack(
                [np.array([[event_time]]), np.asarray(coord, dtype=float).reshape(-1, 1)]
            )
            self.Events = np.append(self.Events, new_event, axis=1)
            self.Sim_num += 1

    # ------------------------------------------------------------------
    # Intensity accessors
    # ------------------------------------------------------------------

    def intensity(self, t: float, x: Any) -> float:
        r"""Conditional intensity :math:`\lambda(t, x \mid H_t)` at one point.

        Parameters
        ----------
        t : float
            Time.
        x : array_like
            Spatial coordinate, of length ``ndim``.

        Returns
        -------
        float
            The intensity, including the background term and floored at zero.

        .. versionadded:: 0.2.0
        """
        return self._full_intensity(np.asarray(x, dtype=float), float(t))

    def intensity_over_interval(
        self,
        times: Any,
        points: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Evaluate the intensity on the tensor grid ``times`` x ``points``.

        Parameters
        ----------
        times : array_like, shape (n_t,)
            Times to evaluate at. The realised event times are merged in and
            the result is sorted and de-duplicated.
        points : array_like, shape (n_x, ndim), optional
            Spatial evaluation points. Defaults to 200 equispaced points across
            the domain when ``ndim == 1``. **Required** when ``ndim >= 2``,
            because no canonical ordering of a multi-dimensional grid exists.

        Returns
        -------
        times : numpy.ndarray, shape (n_t',)
            The sorted evaluation times, events merged in.
        points : numpy.ndarray, shape (n_x, ndim)
            The spatial points actually used.
        intensity : numpy.ndarray, shape (n_x, n_t')
            Rows index space, columns index time. This is the orientation
            :func:`matplotlib.pyplot.contourf` expects, so a field plot is
            ``plt.contourf(times, points[:, 0], intensity)``.

        Raises
        ------
        ValueError
            If `points` is omitted on a domain of two or more dimensions.

        Notes
        -----
        `times` and `points` are both returned because `times` is modified
        (events merged in) and `points` may have been defaulted; without them
        the caller cannot label the axes of `intensity`.

        .. versionadded:: 0.2.0
           Before 0.2.0 there was no way to evaluate the field intensity
           without re-implementing it by hand.
        """
        ndim = self.domain.bounds.shape[0]

        if points is None:
            if ndim != 1:
                raise ValueError(
                    f"points is required for a {ndim}-dimensional domain; there is no "
                    "canonical default grid above one dimension."
                )
            lo, hi = self.domain.bounds[0]
            points_arr = np.linspace(lo, hi, 200).reshape(-1, 1)
        else:
            points_arr = np.asarray(points, dtype=float)
            if points_arr.ndim == 1:
                points_arr = points_arr.reshape(-1, 1)
            if points_arr.shape[1] != ndim:
                raise ValueError(
                    f"points must have shape (n_x, {ndim}) for this domain, got {points_arr.shape}"
                )

        event_times = self.Events[0, :]
        times_arr = np.unique(np.append(np.asarray(times, dtype=float).ravel(), event_times))

        intensity = np.array(
            [
                [self._full_intensity(points_arr[i], float(t)) for t in times_arr]
                for i in range(points_arr.shape[0])
            ]
        )
        return times_arr, points_arr, intensity
