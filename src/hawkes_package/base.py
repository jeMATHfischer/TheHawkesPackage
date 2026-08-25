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
from typing import Any

import numpy as np

from ._deprecation import DeprecatedAlias

__all__ = ["HawkesProcess", "TemporalHawkesProcess"]

#: Anything :func:`numpy.random.default_rng` accepts.
SeedLike = int | np.random.Generator | np.random.SeedSequence | None


def _stalled_message(t: float, bound: float, accepted: int, requested: int) -> str:
    """Explain an exploding process instead of spinning forever.

    When the intensity diverges, the inter-arrival time underflows to zero and
    the thinning loop stops advancing: events pile up at one instant and the
    simulation never returns. Detecting that and raising turns a hang into an
    actionable error.
    """
    return (
        f"Simulation stalled at t={t!r} after {accepted} of {requested} requested "
        f"events: the thinning bound has reached {bound!r} and the inter-arrival "
        "time has underflowed to zero, so time can no longer advance. The process "
        "is exploding -- the expected number of offspring per event is at or above "
        "one. Reduce the kernel's mass or use a more slowly growing nonlinearity."
    )


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
        k_int = int(k)
        if k_int != k:
            raise ValueError(f"k must be a whole number of events, got {k!r}")
        if k_int < 0:
            raise ValueError(f"k must be non-negative, got {k_int}")
        if k_int == 0:
            return
        self._propagate(k_int)

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
        # Empty: `Events` holds only real events, at every moment of the object's
        # life. Before 0.2.0 a fictitious event sat at t=0 until the first
        # simulate() call finished, and it contributed to every intensity sum in
        # the meantime -- so the first events were drawn from a process with one
        # phantom excitation, and the event it was conditioned on was then
        # deleted from the recorded data.
        self.Events = np.empty(0, dtype=float)

    @abstractmethod
    def _conditional_intensity(self, t: float) -> float:
        """Conditional intensity lambda(t | H_t), strictly excluding events at `t`."""

    @abstractmethod
    def _upper_bound(self, t: float) -> float:
        """Return a value M >= lambda(s | H_s) for all s >= `t` until the next event.

        This is the ``M`` of Ogata's algorithm. It must dominate the intensity
        over the whole interval the next candidate can land in, or the thinning
        silently degenerates towards a Poisson process.
        """

    def _propagate(self, k: int) -> None:
        # The bootstrap time is a local, not an element of `Events`.
        t = float(self.Events[-1]) if self.Events.size else 0.0
        accepted = 0

        while accepted < k:
            bound = self._upper_bound(t)
            if not bound > 0:
                raise RuntimeError(
                    f"Non-positive thinning bound M={bound!r} at t={t!r}; the "
                    "kernel or nonlinearity must keep the intensity positive."
                )
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(_stalled_message(t, bound, accepted, k))
            t = advanced

            if self.rng.uniform() * bound <= self._conditional_intensity(t):
                self.Events = np.append(self.Events, t)
                accepted += 1
                # Counted per event, so `Sim_num == len(Events)` still holds if
                # the loop raises partway through.
                self.Sim_num += 1

    def intensity_over_interval(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
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
