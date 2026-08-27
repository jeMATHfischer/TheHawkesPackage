"""Linear Hawkes process with an exponential excitation kernel."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import SeedLike, TemporalHawkesProcess

__all__ = ["ExponentialHawkes"]


class ExponentialHawkes(TemporalHawkesProcess):
    r"""Linear Hawkes process with kernel :math:`\kappa(s) = \alpha e^{-\beta s}`.

    The conditional intensity is

    .. math::

        \lambda(t \mid H_t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)}.

    Parameters
    ----------
    param : array_like of shape (3,)
        ``[mu, alpha, beta]`` — background rate, excitation size and decay rate.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

    Raises
    ------
    ValueError
        If `param` does not have exactly three entries, if ``beta <= 0``, if
        `mu` or `alpha` is negative, or if ``alpha / beta >= 1``. The last is
        the stationarity condition: the branching ratio is ``alpha / beta``, so
        at or above 1 each event spawns at least one offspring on average and
        the simulation would not terminate.

    Examples
    --------
    >>> process = ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
    >>> process.simulate(100)
    >>> len(process.events)
    100
    """

    def __init__(self, param: Any, rng: SeedLike = None) -> None:
        param = np.asarray(param, dtype=float).ravel()
        if param.size != 3:
            raise ValueError(
                f"param must have exactly 3 entries [mu, alpha, beta], got {param.size}"
            )
        mu, alpha, beta = (float(v) for v in param)
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if mu < 0 or alpha < 0:
            raise ValueError(f"mu and alpha must be non-negative, got mu={mu}, alpha={alpha}")
        if alpha / beta >= 1:
            raise ValueError(
                f"Stability condition violated: alpha/beta = {alpha / beta:.4f} >= 1. "
                "The process will not be stationary."
            )

        super().__init__(rng=rng)
        self.param = param
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.temporal = lambda s: alpha * np.exp(-beta * np.asarray(s, dtype=float))

    def _conditional_intensity(self, t: float) -> float:
        record = self.events
        past = record[record < t]
        return float(self.mu + self.alpha * np.exp(-self.beta * (t - past)).sum())

    def _upper_bound(self, t: float) -> float:
        # The kernel decreases and no event can arrive before the next accepted
        # one, so the intensity at `t` dominates the whole interval. Bounding at
        # `t` rather than at the last event is both tighter after a rejection
        # and well defined when no events have occurred yet.
        return float(self.mu + self.alpha * np.exp(-self.beta * (t - self.events)).sum())
