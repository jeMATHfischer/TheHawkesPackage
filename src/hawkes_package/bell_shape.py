"""Hawkes process for kernels with a single interior maximum."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import fmin

from .base import SeedLike, TemporalHawkesProcess

__all__ = ["BellShapeHawkes"]


class BellShapeHawkes(TemporalHawkesProcess):
    r"""Hawkes process whose kernel rises before it decays.

    The conditional intensity has the same form as
    :class:`~hawkes_package.monotone.MonotoneKernelHawkes`,

    .. math::

        \lambda(t \mid H_t) = \varphi\!\left( \sum_{t_i < t} \kappa(t - t_i) \right),

    but :math:`\kappa` is no longer monotone: excitation ramps up to a peak at
    lag :attr:`ext` before decaying. The bound used by
    :class:`~hawkes_package.monotone.MonotoneKernelHawkes` is therefore invalid
    while the kernel is still rising, so each event is instead bounded by its
    own future supremum -- the peak value if that event has not yet peaked, its
    current value if it has.

    .. versionchanged:: 0.2.0
       Before 0.2.0 the bound added a single peak's worth of headroom to the
       whole intensity. That is not enough when two or more events are rising
       at once, and the thinning invariant ``M >= lambda`` failed in a few
       percent of steps -- silently biasing the simulated process.

    Parameters
    ----------
    temporal : callable
        The kernel :math:`\kappa`, with one global maximum on ``[0, inf)``.
    nonlinearity : callable
        The monotone-increasing :math:`\varphi`. Defaults to ``x + 2``.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Attributes
    ----------
    ext : float
        Lag at which `temporal` attains its maximum, located numerically with
        :func:`scipy.optimize.fmin`.

    Examples
    --------
    >>> def triangular(s):
    ...     s = np.asarray(s, dtype=float)
    ...     return 2 * s * ((s > 0) & (s < 0.5)) + (-2 * s + 2) * ((s >= 0.5) & (s < 1))
    >>> process = BellShapeHawkes(triangular, rng=0)
    >>> process.simulate(20)
    >>> len(process.Events)
    20
    """

    def __init__(
        self,
        temporal: Callable[[Any], Any],
        nonlinearity: Callable[[Any], Any] = lambda x: x + 2,
        rng: SeedLike = None,
    ) -> None:
        super().__init__(rng=rng)
        self.temporal = temporal
        self.nonlinearity = nonlinearity
        self.ext = float(fmin(lambda s: -self.temporal(s), 0, disp=False)[0])
        self._peak = float(self.temporal(self.ext))

    def _conditional_intensity(self, t: float) -> float:
        past = self.Events[self.Events < t]
        return float(self.nonlinearity(np.sum(self.temporal(t - past))))

    def _upper_bound(self, t: float) -> float:
        # Bound each event's future contribution by its own supremum: an event
        # that has not yet peaked can still climb to the peak, one that has is
        # already decaying. Events at exactly `t` count -- at the start of a
        # thinning step `t` *is* the most recent event time.
        past = self.Events[self.Events <= t]
        if past.size == 0:
            return float(self.nonlinearity(0.0))
        lags = t - past
        factors = np.where(lags < self.ext, self._peak, self.temporal(lags))
        return float(self.nonlinearity(factors.sum()))
