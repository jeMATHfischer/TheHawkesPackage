"""Shared machinery for every Hawkes process in this package.

:class:`HawkesProcess` owns the bookkeeping that is identical everywhere — the
random stream, the event record, the simulated-event counter and the canonical
:meth:`simulate` entry point.

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

from ._deprecation import DeprecatedAttribute

__all__ = ["HawkesProcess", "TemporalHawkesProcess"]

#: Anything :func:`numpy.random.default_rng` accepts.
SeedLike = int | np.random.Generator | np.random.SeedSequence | None

#: Slots a fresh buffer reserves. Small enough to be free, large enough that a
#: short run never reallocates.
_MIN_CAPACITY = 8


class _EventBuffer:
    """Amortised-growth storage for the event record.

    Through 0.3.x each accepted event did ``Events = np.append(Events, event)``,
    which allocates a new array and copies the whole record every time: to write
    down ``n`` events it moved ``O(n^2)`` bytes. Doubling makes that ``O(n)``
    amortised.

    The public face is a **view** onto the filled prefix, never a copy, and that
    is safe for a reason worth stating. A view of length ``n`` is a stable
    snapshot of the first ``n`` events: growth allocates a new buffer and leaves
    the old one — and every view onto it — holding exactly what it held before,
    while an in-place write at index ``n`` cannot reach a view that stops at
    ``n``. So ``a = p.events; p.simulate(1)`` leaves ``a`` reading the same
    events it read before, which is what ``np.append`` did too.

    One buffer serves both layouts because the event count is the **last** axis
    in each: ``(n,)`` for a temporal record, ``(ndim + 1, n)`` for a
    spatio-temporal one, whose first row holds the times. Everything here is
    therefore indexed ``[..., :count]``.

    Parameters
    ----------
    rows : int or None
        Number of rows in the record, or ``None`` for a flat one.

    .. versionadded:: 0.4.0
    """

    __slots__ = ("_count", "_data", "_rows")

    def __init__(self, rows: int | None = None) -> None:
        self._rows = rows
        self._count = 0
        self._data = np.empty(self._shaped(0), dtype=float)

    def _shaped(self, capacity: int) -> tuple[int, ...]:
        """Return the buffer's shape at a given capacity."""
        return (capacity,) if self._rows is None else (self._rows, capacity)

    @property
    def view(self) -> np.ndarray:
        """The filled prefix, as a view.

        Empty until the first event is accepted, and holding only real events at
        every moment of the object's life. Before 0.2.0 a fictitious event sat
        at ``t = 0`` until the first :meth:`~HawkesProcess.simulate` call
        finished, contributing to every intensity sum in the meantime -- so the
        first events were drawn from a process with one phantom excitation, and
        the event they were conditioned on was then deleted from the record.
        """
        return self._data[..., : self._count]

    def append(self, event: Any) -> None:
        """Record one event: a scalar time, or a column of time and coordinates."""
        if self._count == self._data.shape[-1]:
            self._reserve(max(_MIN_CAPACITY, 2 * self._count))
        self._data[..., self._count] = event
        self._count += 1

    def _reserve(self, capacity: int) -> None:
        """Move the record into a buffer of the given capacity."""
        bigger = np.empty(self._shaped(capacity), dtype=float)
        bigger[..., : self._count] = self._data[..., : self._count]
        self._data = bigger

    def replace(self, events: Any) -> None:
        """Adopt `events` as the whole record.

        This is what stands behind ``process.events = history``, which is how a
        caller conditions a realisation on events it did not simulate.
        """
        record = np.array(events, dtype=float, copy=True)
        wanted = 1 if self._rows is None else 2
        if record.ndim != wanted:
            raise ValueError(
                f"the event record must be {wanted}-dimensional, got shape {record.shape}"
            )
        if self._rows is not None and record.shape[0] != self._rows:
            raise ValueError(
                f"the event record must have {self._rows} rows -- one of times above "
                f"{self._rows - 1} of coordinates -- got shape {record.shape}"
            )
        self._data = record
        self._count = record.shape[-1]


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

    rows : int or None
        Rows in the event record: ``None`` for a bare sequence of times,
        ``ndim + 1`` when each event also carries a location.

        .. versionadded:: 0.4.0

    Attributes
    ----------
    events : numpy.ndarray
        Simulated events. Shape ``(n,)`` for temporal processes and
        ``(ndim + 1, n)`` for spatio-temporal ones, where row 0 holds times.
        Assignable: setting it conditions the realisation on a history the
        process did not simulate.

        .. versionchanged:: 0.4.0
           Renamed from ``Events``, which still works and warns until 0.5.0. It
           is now a view onto a growing buffer rather than a fresh array per
           event — see :class:`_EventBuffer` for why that is the same thing to
           a reader.
    n_simulated : int
        Number of events accepted so far across all :meth:`simulate` calls.

        .. versionchanged:: 0.4.0
           Renamed from ``Sim_num``, which still works and warns until 0.5.0.
    rng : numpy.random.Generator
        The stream every draw is taken from.
    """

    def __init__(self, *, rng: SeedLike = None, rows: int | None = None) -> None:
        self.rng = np.random.default_rng(rng)
        self.n_simulated = 0
        self._events = _EventBuffer(rows)

    @property
    def events(self) -> np.ndarray:
        """The realised events, as a view onto the record."""
        return self._events.view

    @events.setter
    def events(self, value: Any) -> None:
        self._events.replace(value)

    # Renamed in 0.4.0, and aliased rather than dropped because every notebook
    # cell ever written against this package touches them.
    Events = DeprecatedAttribute("events")
    Sim_num = DeprecatedAttribute("n_simulated")

    @abstractmethod
    def _propagate(self, k: int) -> None:
        """Simulate exactly `k` further events, appending them to :attr:`events`."""

    def simulate(self, k: int) -> None:
        """Simulate `k` further events and append them to :attr:`events`.

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


class TemporalHawkesProcess(HawkesProcess):
    """A Hawkes process on the time line, simulated by Ogata's thinning.

    Subclasses implement :meth:`_conditional_intensity` and :meth:`_upper_bound`.
    """

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
        # The bootstrap time is a local, not an element of `events`.
        record = self.events
        t = float(record[-1]) if record.size else 0.0
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
                self._events.append(t)
                accepted += 1
                # Counted per event, so `n_simulated == len(events)` still holds
                # if the loop raises partway through.
                self.n_simulated += 1

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
        times = np.unique(np.append(np.asarray(x, dtype=float).ravel(), self.events))
        intensity = np.array([self._conditional_intensity(float(t)) for t in times])
        return times, intensity
