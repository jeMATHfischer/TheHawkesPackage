#!/usr/bin/env python3
"""Legacy spatio-temporal Hawkes process on a periodic interval.

Originally written in 2018 to simulate on ``[-pi, pi] x [0, inf)`` with periodic
boundary conditions in space. It predates the :class:`SpatialDomain` abstraction
and is kept so that results published with it remain reproducible.

.. deprecated:: 0.2.0
   Use :class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`,
   which supports arbitrary domains. This class is removed in 0.4.0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import quad

from .._deprecation import warn_renamed
from .._numerics import as_float, locate_peak
from ..base import HawkesProcess, SeedLike, _stalled_message
from ..mcmc import mcmc_sampler

__all__ = ["LegacySpatioTemporalHawkesProcess"]


class LegacySpatioTemporalHawkesProcess(HawkesProcess):
    """Spatio-temporal Hawkes process on a periodic one-dimensional interval.

    Parameters
    ----------
    Base : callable
        Background intensity as a function of the spatial coordinate.
    spatial : callable
        Spatial kernel, evaluated at a signed offset (not a distance).
    temporal : callable
        Temporal kernel, evaluated at a non-negative time lag.
    space : tuple of float
        Interval endpoints, periodic at the boundary. Defaults to
        ``(-pi, pi)``.

        .. versionchanged:: 0.2.0
           Renamed from ``Space`` and changed from a mutable list default to a
           tuple. The old keyword still works with a ``DeprecationWarning``.
           The value is now actually honoured: before 0.2.0 the spatial sampler
           hard-coded ``[-pi, pi]`` and silently ignored anything else.
    monotone_temporal_kernel : bool
        Set ``True`` when `temporal` is monotone decreasing.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

        .. versionadded:: 0.2.0

    Attributes
    ----------
    Events : numpy.ndarray
        Shape ``(2, n)``. Row 0 holds times, row 1 holds spatial coordinates.
    """

    def __init__(
        self,
        Base: Callable[[Any], float],
        spatial: Callable[[Any], Any],
        temporal: Callable[[Any], Any],
        space: tuple[float, float] = (-np.pi, np.pi),
        monotone_temporal_kernel: bool = False,
        rng: SeedLike = None,
        peak_lag: float | None = None,
        peak_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        if "Space" in kwargs:
            warn_renamed(
                f"{type(self).__name__}(Space=...)",
                f"{type(self).__name__}(space=...)",
                stacklevel=3,
            )
            space = tuple(kwargs.pop("Space"))
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

        super().__init__(rng=rng)
        self.Base = Base
        self.spatial = spatial
        self.temporal = temporal
        self.space = (float(space[0]), float(space[1]))
        self.monotone_temporal_kernel = monotone_temporal_kernel
        self.Events = np.array([[0.0], [0.0]])

        if monotone_temporal_kernel is not True:
            if peak_lag is None:
                located = locate_peak(temporal, name="temporal kernel")
                self.temporal_extremum, self.temporal_peak = located.lag, located.value
            else:
                self.temporal_extremum = float(peak_lag)
                self.temporal_peak = as_float(
                    peak_value if peak_value is not None else temporal(self.temporal_extremum)
                )

    @property
    def Space(self) -> tuple[float, float]:
        """Deprecated alias for :attr:`space`."""
        warn_renamed(
            f"{type(self).__name__}.Space",
            f"{type(self).__name__}.space",
            stacklevel=3,
        )
        return self.space

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _period(self) -> float:
        return self.space[1] - self.space[0]

    def _periodize(self, x: Any) -> Any:
        """Map `x` back into the periodic interval."""
        lo = self.space[0]
        return (np.asarray(x, dtype=float) - lo) % self._period + lo

    def _periodized_spatial(self, x: Any) -> Any:
        return self.spatial(self._periodize(x))

    def _past_columns(self, t: float, inclusive: bool = False) -> list[int]:
        if inclusive:
            return [i for i in range(self.Events.shape[1]) if 0 < self.Events[0, i] <= t]
        return [i for i in range(self.Events.shape[1]) if 0 < self.Events[0, i] < t]

    def _temporal_factors(self, t: float, bound: bool = False) -> np.ndarray:
        """Temporal kernel factors at `t`, or their suprema over all later times."""
        idx = self._past_columns(t, inclusive=bound)
        lags = np.array([t - self.Events[0, i] for i in idx])
        if lags.size == 0:
            return lags
        values = np.array([float(self.temporal(lag)) for lag in lags])
        if bound and self.monotone_temporal_kernel is not True:
            values = np.where(lags < self.temporal_extremum, self.temporal_peak, values)
        return values

    def _dist_temporal(self, t: float) -> np.ndarray:
        return self._temporal_factors(t)

    def _dist_spatial(self, x: Any, t: float, bound: bool = False) -> np.ndarray:
        return np.array(
            [
                self._periodized_spatial(x - self.Events[1, i])
                for i in self._past_columns(t, inclusive=bound)
            ]
        )

    def _full_intensity(self, x: Any, t: float, bound: bool = False) -> float:
        contrib = np.multiply(
            self._temporal_factors(t, bound=bound), self._dist_spatial(x, t, bound=bound)
        ).sum()
        return max(0.0, float(self.Base(x)) + float(contrib))

    def _integrated_intensity(self, t: float, bound: bool = False) -> float:
        """Intensity integrated over the spatial interval at time `t`."""
        val, _ = quad(
            lambda x: self._full_intensity(x, t, bound=bound), self.space[0], self.space[1]
        )
        return float(val)

    def _upper_bound(self, t: float) -> float:
        return self._integrated_intensity(t, bound=True)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _propagate(self, k: int) -> None:
        for done in range(k):
            t = float(self.Events[0, -1])

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

            coord = mcmc_sampler(
                lambda x: self._full_intensity(self._periodize(x), event_time),
                np.array([self.space]),
                seed=self.rng,
            )
            event_space = float(np.ravel(self._periodize(coord))[0])

            self.Events = np.append(self.Events, np.array([[event_time], [event_space]]), axis=1)

        if self.Sim_num == 0:
            self.Events = self.Events[:, 1:]
        self.Sim_num += k

    # ------------------------------------------------------------------
    # Intensity accessors
    # ------------------------------------------------------------------

    def intensity(self, t: float, x: Any) -> float:
        r"""Conditional intensity :math:`\lambda(t, x \mid H_t)` at one point.

        .. versionadded:: 0.2.0
        """
        return self._full_intensity(self._periodize(x), float(t))

    def intensity_over_interval(
        self,
        times: Any,
        points: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the intensity on the tensor grid ``times`` x ``points``.

        Parameters
        ----------
        times : array_like, shape (n_t,)
            Times to evaluate at. Event times are merged in and the result is
            sorted and de-duplicated.
        points : array_like, shape (n_x,), optional
            Spatial points. Defaults to 200 equispaced points across
            :attr:`space`.

        Returns
        -------
        times : numpy.ndarray, shape (n_t',)
        points : numpy.ndarray, shape (n_x,)
        intensity : numpy.ndarray, shape (n_x, n_t')
            Rows index space, columns index time, so a field plot is
            ``plt.contourf(times, points, intensity)``.

        .. versionadded:: 0.2.0
        """
        if points is None:
            points_arr = np.linspace(self.space[0], self.space[1], 200)
        else:
            points_arr = np.asarray(points, dtype=float).ravel()

        times_arr = np.unique(np.append(np.asarray(times, dtype=float).ravel(), self.Events[0, :]))
        intensity = np.array(
            [[self._full_intensity(x, float(t)) for t in times_arr] for x in points_arr]
        )
        return times_arr, points_arr, intensity
