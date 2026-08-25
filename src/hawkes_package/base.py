"""Shared machinery for every Hawkes process in this package.

:class:`HawkesProcess` owns the bookkeeping that is identical everywhere — the
random stream, the simulated-event counter, the canonical :meth:`simulate`
entry point and the deprecated aliases for it.

:class:`TemporalHawkesProcess` adds the purely temporal Ogata thinning loop.
Concrete classes supply two hooks, :meth:`~TemporalHawkesProcess._conditional_intensity`
and :meth:`~TemporalHawkesProcess._upper_bound`, and inherit both the simulator
and the intensity accessor. Defining the accessor in terms of the *same*
function the simulator thins against is what keeps the two from diverging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np

from ._deprecation import DeprecatedAlias

__all__ = ["HawkesProcess", "TemporalHawkesProcess"]

#: Anything :func:`numpy.random.default_rng` accepts.
SeedLike = Union[None, int, np.random.Generator, np.random.SeedSequence]


class HawkesProcess(ABC):
    """Base class for all Hawkes processes.

    Parameters
    ----------
    rng : None, int or numpy.random.Generator
        Source of randomness. Pass an ``int`` for a reproducible run, or an
        existing :class:`~numpy.random.Generator` to share one stream with
        other components. ``None`` draws from fresh OS entropy.

        .. versionadded:: 0.2.0
           Simulations are no longer affected by :func:`numpy.random.seed`.

    Attributes
    ----------
    Events : numpy.ndarray
        Simulated events. Shape ``(n,)`` for temporal processes and
        ``(ndim + 1, n)`` for spatio-temporal ones, where row 0 holds times.
    Sim_num : int
        Total number of events requested so far across all
        :meth:`simulate` calls.
    rng : numpy.random.Generator
        The stream every draw is taken from.
    """

    Events: np.ndarray

    def __init__(self, *, rng: SeedLike = None) -> None:
        self.rng = np.random.default_rng(rng)
        self.Sim_num = 0

    @abstractmethod
    def _propagate(self, k: int) -> None:
        """Simulate exactly `k` further events, appending them to ``Events``."""

    def simulate(self, k: int) -> None:
        """Simulate `k` further events and append them to :attr:`Events`.

        Calling this repeatedly continues the same realisation; it does not
        restart it.

        Parameters
        ----------
        k : int
            Number of events to generate. ``0`` is a no-op.

        Raises
        ------
        ValueError
            If `k` is negative.
        """
        k = int(k)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if k == 0:
            return
        self._propagate(k)

    # -- Deprecated aliases -------------------------------------------------
    # Declared once here and inherited by every process, rather than repeated
    # (and spelled inconsistently) on each class as they were before 0.2.0.

    propagate_by_amount = DeprecatedAlias("simulate")
    propagate_by_k_events = DeprecatedAlias("simulate")
    propogate_by_amount = DeprecatedAlias("simulate")


class TemporalHawkesProcess(HawkesProcess):
    """A Hawkes process on the time line, simulated by Ogata's thinning.

    Subclasses implement :meth:`_conditional_intensity` and :meth:`_upper_bound`.
    """

    def __init__(self, *, rng: SeedLike = None) -> None:
        super().__init__(rng=rng)
        # A fictitious event at t=0 bootstraps the first thinning step; it is
        # removed once the first simulate() call completes.
        self.Events = np.array([0.0])

    @abstractmethod
    def _conditional_intensity(self, t: float) -> float:
        """Conditional intensity lambda(t | H_t), strictly excluding events at `t`."""

    @abstractmethod
    def _upper_bound(self, t: float) -> float:
        """A value M >= lambda(s | H_s) for all s >= `t` up to the next event.

        This is the ``M`` of Ogata's algorithm. It must dominate the intensity
        over the whole interval the next candidate can land in, or the thinning
        silently degenerates towards a Poisson process.
        """

    def _propagate(self, k: int) -> None:
        t = float(self.Events[-1])
        accepted = 0

        while accepted < k:
            bound = self._upper_bound(t)
            if not bound > 0:
                raise RuntimeError(
                    f"Non-positive thinning bound M={bound!r} at t={t!r}; the "
                    "kernel or nonlinearity must keep the intensity positive."
                )
            t += self.rng.exponential() / bound
            if self.rng.uniform() * bound <= self._conditional_intensity(t):
                self.Events = np.append(self.Events, t)
                accepted += 1

        if self.Sim_num == 0:
            self.Events = self.Events[1:]  # drop the t=0 bootstrap event
        self.Sim_num += k

    def intensity_over_interval(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the conditional intensity on `x` merged with the event times.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate. The realised event times are merged in
            and the result is sorted and de-duplicated, so the returned grid is
            generally longer than `x`.

        Returns
        -------
        times : numpy.ndarray
            The sorted evaluation grid.
        intensity : numpy.ndarray
            ``lambda(t | H_t)`` at each entry of `times`, including the
            background rate.

        Notes
        -----
        Because the intensity excludes events at exactly `t`, the value
        returned at an event time is the left limit — the pre-jump value.

        .. versionchanged:: 0.2.0
           The returned values now include the background intensity. Before
           0.2.0 :class:`~hawkes_package.exponential.ExponentialHawkes` omitted
           it, so its curves sat a constant ``mu`` below the intensity the
           simulator actually thinned against.
        """
        times = np.unique(np.append(np.asarray(x, dtype=float).ravel(), self.Events))
        intensity = np.array([self._conditional_intensity(float(t)) for t in times])
        return times, intensity
