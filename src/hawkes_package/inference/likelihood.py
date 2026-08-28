r"""The Hawkes log-likelihood, computed from the simulator's own intensity hooks.

For a **fully observed** Hawkes process there is no latent state: the intensity
is a deterministic function of the events that have already happened. So the
log-likelihood on ``[start, T]`` is available in closed form up to one integral,

.. math::

    \ell(\theta) = \sum_{t_i \le T} \log \lambda_\theta(t_i^- \mid H)
                   - \int_{start}^{T} \lambda_\theta(s)\,\mathrm{d}s,

and in the spatio-temporal case the same with
:math:`\log\lambda_\theta(t_i^-, x_i)` in the sum and
:math:`\int\!\!\int_D \lambda_\theta` in the integral.

Three things about that formula decide the design of this module.

**The left limit is not an aside.** The sum uses :math:`\lambda(t_i^-)`, the
intensity *just before* the event, excluding the event itself. Every intensity
hook in this package already filters ``t_j < t`` strictly, which is why
:func:`_bind_history` can assign the whole observed array at once and get both
terms right in one shot -- and also why tied event times are refused: two events
at the same instant would each be excluded from the other's intensity, and one
:math:`\log\lambda` contribution would quietly go missing.

**The observation window is data.** ``History.end`` has no default. Defaulting
it to the last event time silently switches the model between "observed on
``[0, T]``" and "stopped at the n-th event", and the two differ by
:math:`-\int_{t_n}^{T}\lambda` -- which is the information that *nothing
happened* after the last event. Dropping it biases ``mu`` and the excitation
upward, and nothing raises.

**The integral is where the bias lives.** A compensator computed too small
subtracts too little from the penalty on a high intensity, so the fit comes back
with more background and more excitation than the data supports and looks
converged doing it. Everything in :mod:`~hawkes_package.inference._compensator`
exists for that one failure mode.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..base import HawkesProcess, TemporalHawkesProcess
from ..spatio_temporal.process import SpatioTemporalHawkesProcess
from . import _compensator
from ._geometry import DEFAULT_MAX_BYTES, GeometryCache, build_geometry, extend_geometry
from .models import ProcessModel, SpatialComponents

__all__ = [
    "ExponentialLogLikelihood",
    "History",
    "LikelihoodState",
    "LogLikelihood",
    "SpatioTemporalLogLikelihood",
    "TemporalLogLikelihood",
]

#: Nodes evaluated per chunk in the vectorized spatio-temporal compensator. The
#: intermediate is ``nodes x events``, so an unchunked pass at n = 2000 would
#: allocate 256 MiB to produce a 128 KiB answer.
_NODE_CHUNK = 4096


# ---------------------------------------------------------------------------
# The observed data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class History:
    """An observed realisation, together with the window it was observed on.

    Parameters
    ----------
    times : array_like of shape (n,)
        Event times, strictly increasing and strictly inside ``(start, end]``.
    points : array_like of shape (ndim, n), optional
        Event locations, in the same layout as
        :attr:`~hawkes_package.SpatioTemporalHawkesProcess.events` minus its
        time row. ``None`` for a temporal history.
    start : float
        Beginning of the observation window.
    end : float
        End of the observation window. **Keyword-required and undefaulted**;
        see the module docstring for what a default would cost.

    Raises
    ------
    ValueError
        If the times are not strictly increasing, are not finite, or leave the
        window. Ties in particular: two events at one instant each vanish from
        the other's conditional intensity, so the log-likelihood loses a term
        with nothing said.

    Examples
    --------
    >>> history = History(np.array([0.4, 1.1, 2.9]), None, 0.0, end=4.0)
    >>> history.upto(1.5).times
    array([0.4, 1.1])

    .. versionadded:: 0.5.0
    """

    times: np.ndarray
    points: np.ndarray | None
    start: float
    end: float

    def __post_init__(self) -> None:
        """Normalise the arrays and refuse a history the likelihood cannot use."""
        times = np.asarray(self.times, dtype=float).ravel()
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "start", float(self.start))
        object.__setattr__(self, "end", float(self.end))

        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError(f"start and end must be finite, got {self.start}, {self.end}")
        if self.end < self.start:
            raise ValueError(f"end={self.end} is before start={self.start}")
        if times.size and not np.all(np.isfinite(times)):
            raise ValueError("event times must all be finite")
        if times.size and not np.all(np.diff(times) > 0):
            bad = int(np.argmin(np.diff(times))) if times.size > 1 else 0
            raise ValueError(
                f"event times must be strictly increasing; times[{bad}]={times[bad]} is "
                f"not below times[{bad + 1}]={times[bad + 1]}. Tied times are the case "
                "worth naming: each event is excluded from the other's conditional "
                "intensity, so a log-intensity term vanishes from the likelihood "
                "without anything raising."
            )
        if times.size and (times[0] <= self.start or times[-1] > self.end):
            raise ValueError(
                f"events must lie in (start, end] = ({self.start}, {self.end}], but they "
                f"span [{times[0]}, {times[-1]}]"
            )

        if self.points is not None:
            points = np.asarray(self.points, dtype=float)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            if points.shape[1] != times.size:
                raise ValueError(
                    f"points must have one column per event: got shape {points.shape} "
                    f"for {times.size} event(s)"
                )
            if not np.all(np.isfinite(points)):
                raise ValueError("event locations must all be finite")
            object.__setattr__(self, "points", points)

    @property
    def n_events(self) -> int:
        """Number of observed events."""
        return int(self.times.size)

    @property
    def ndim(self) -> int:
        """Spatial dimension, ``0`` for a temporal history."""
        return 0 if self.points is None else int(self.points.shape[0])

    @property
    def duration(self) -> float:
        """Length of the observation window."""
        return self.end - self.start

    @classmethod
    def from_events(
        cls,
        events: Any,
        *,
        start: float = 0.0,
        end: float,
    ) -> History:
        """Build a history from a process's event record.

        Parameters
        ----------
        events : array_like
            Shape ``(n,)`` for a temporal record, or ``(ndim + 1, n)`` with
            times in row 0 for a spatio-temporal one -- the two layouts
            :attr:`~hawkes_package.base.HawkesProcess.events` uses.
        start : float
            Beginning of the observation window.
        end : float
            End of it. Required.
        """
        record = np.asarray(events, dtype=float)
        if record.ndim == 1:
            return cls(record, None, start, end)
        if record.ndim == 2:
            return cls(record[0], record[1:], start, end)
        raise ValueError(f"events must be 1- or 2-dimensional, got shape {record.shape}")

    @classmethod
    def from_simulation(cls, process: HawkesProcess, *, start: float = 0.0) -> History:
        """Build a history from a process simulated by event count.

        Sets ``end`` to the last event time, which is the *correct* window for
        the output of :meth:`~hawkes_package.base.HawkesProcess.simulate`: that
        call stops at the k-th event, so the observation really did end there
        and there is no empty tail to account for. It would be the wrong window
        for the output of
        :meth:`~hawkes_package.base.HawkesProcess.simulate_until`, where the
        horizon is the window and the tail after the last event is data --
        use :meth:`from_events` with the horizon there.
        """
        record = np.asarray(process.events, dtype=float)
        times = record if record.ndim == 1 else record[0]
        end = float(times[-1]) if times.size else float(start)
        return cls.from_events(record, start=start, end=end)

    def upto(self, t: float) -> History:
        """Return the same history as observed at time `t`.

        Raises
        ------
        ValueError
            If `t` lies outside ``[start, end]``. Extending the window past what
            was observed would assert an empty interval nobody looked at.
        """
        cut = float(t)
        if cut < self.start or cut > self.end:
            raise ValueError(
                f"upto({cut}) leaves the observation window [{self.start}, {self.end}]"
            )
        keep = self.times <= cut
        return History(
            self.times[keep],
            None if self.points is None else self.points[:, keep],
            self.start,
            cut,
        )

    def as_process_events(self) -> np.ndarray:
        """Return the history in the layout a process's event record uses."""
        if self.points is None:
            return self.times
        return np.vstack([self.times[None, :], self.points])


def _located(history: History) -> np.ndarray:
    """Return the event locations, refusing a history that carries none.

    Reached from every spatio-temporal path, so the check is written once rather
    than as an assertion per call site -- and it is a `ValueError` because a
    temporal history handed to a spatial likelihood is a caller's mistake, not
    an internal invariant.
    """
    if history.points is None:
        raise ValueError(
            "a spatio-temporal likelihood needs a history with event locations; this "
            "one carries times only. Build it from a spatio-temporal process's record, "
            "whose rows are one of times above ndim of coordinates."
        )
    return history.points


def _bind_history(process: HawkesProcess, history: History) -> HawkesProcess:
    """Condition `process` on `history`, in place.

    The **only** function in this subpackage that touches a process's event
    record. Every likelihood goes through it, so there is one place where the
    conditioning convention lives and one place to change when the record's
    layout does.

    ``n_simulated`` is set alongside the record. Leaving it at zero would put
    the object in a state it can never reach on its own -- a record of events
    it claims not to have simulated -- and the first thing to notice would be a
    diagnostic reporting a count of zero, long after the fit.
    """
    process.events = history.as_process_events()
    process.n_simulated = history.n_events
    return process


# ---------------------------------------------------------------------------
# Incremental state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LikelihoodState:
    """How far a log-likelihood has been evaluated, and what it carries forward.

    Attributes
    ----------
    upto : float
        The time the evaluation has reached. The next
        :meth:`LogLikelihood.extend` resumes here.
    n_events : int
        Events accounted for so far.
    log_lik : float
        The log-likelihood on ``(start, upto]``.
    carry : tuple of float
        Whatever the implementation needs to resume in O(new events) rather than
        O(all events). Empty where there is no such summary -- a general kernel
        has no finite-dimensional sufficient statistic, which is precisely why
        it costs O(n) per evaluation and the exponential one does not.
    """

    upto: float
    n_events: int
    log_lik: float
    carry: tuple[float, ...] = field(default=())


@runtime_checkable
class LogLikelihood(Protocol):
    """A log-likelihood that can be evaluated in blocks as data arrives."""

    model: ProcessModel

    def initial_state(self, start: float) -> LikelihoodState:
        """Return the state before any data has been accounted for."""
        ...

    def extend(
        self,
        state: LikelihoodState,
        theta: Any,
        history: History,
        upto: float,
    ) -> tuple[LikelihoodState, float]:
        """Account for ``(state.upto, upto]``, returning the new state and increment."""
        ...

    def total(self, theta: Any, history: History, upto: float | None = None) -> float:
        """Return the whole log-likelihood on ``(start, upto]``."""
        ...

    def compensator(self, theta: Any, history: History, times: Any) -> np.ndarray:
        """Return the compensator at each of `times`, measured from ``history.start``."""
        ...


def _check_extension(state: LikelihoodState, history: History, upto: float) -> float:
    """Validate a block boundary, returning it as a float."""
    end = float(upto)
    if end < state.upto:
        raise ValueError(
            f"cannot extend a likelihood backwards: the state reaches {state.upto} and "
            f"upto={end} is before it"
        )
    if end > history.end:
        raise ValueError(f"upto={end} is past the observation window, which ends at {history.end}")
    return end


# ---------------------------------------------------------------------------
# Temporal, through the hooks
# ---------------------------------------------------------------------------


class TemporalLogLikelihood:
    r"""The log-likelihood of any temporal model, through its intensity hook.

    The normative implementation for the temporal classes: it evaluates
    ``_conditional_intensity`` -- the same function the Ogata loop thins against
    -- and integrates it by panelled Gauss-Legendre. Nothing about the intensity
    is re-derived here, which is the point. A second, faster expression for
    :math:`\lambda` is how this package once came to plot a curve a constant
    ``mu`` below the one it simulated, and a second expression used by the
    *likelihood* would be the same mistake somewhere it cannot be seen.

    Costs :math:`O(n^2 P)` for a full evaluation, ``P`` being the panel order:
    each of the ``n`` panels carries ``P`` nodes and each node sums over up to
    ``n`` past events. At ``n = 2000`` that is a few hundred milliseconds per
    particle, so a rejuvenation move over 512 particles is minutes rather than
    seconds -- size a general-kernel fit at a few hundred events, or use
    :class:`ExponentialLogLikelihood`, which is :math:`O(n)`.

    Parameters
    ----------
    model : ProcessModel
        Must be a temporal model.
    order : int
        Gauss-Legendre nodes per panel.
    extra_lags : sequence of float
        Lags at which the kernel has an interior kink, so a panel edge is placed
        there. A triangular kernel peaking at lag 0.5 needs ``(0.5,)``; without
        it the rule integrates across the kink and the compensator carries an
        error with a consistent sign.
    check : bool
        Compare order ``P`` against ``2P`` on the first full evaluation and warn
        if they disagree. Once per object, not once per particle.

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        model: ProcessModel,
        *,
        order: int = _compensator.DEFAULT_ORDER,
        extra_lags: Sequence[float] = (),
        check: bool = True,
    ) -> None:
        if model.ndim != 0:
            raise ValueError(
                f"{type(self).__name__} is for temporal models; this one is "
                f"{model.ndim}-dimensional in space. Use SpatioTemporalLogLikelihood."
            )
        self.model = model
        self.order = int(order)
        self.extra_lags = tuple(float(lag) for lag in extra_lags)
        self.check = bool(check)
        self._checked = False

    def initial_state(self, start: float) -> LikelihoodState:
        """Return the empty state at `start`."""
        return LikelihoodState(upto=float(start), n_events=0, log_lik=0.0, carry=())

    def _process(self, theta: Any, history: History, upto: float) -> TemporalHawkesProcess:
        """Build the process at `theta` and condition it on the history so far."""
        process = self.model(theta)
        if not isinstance(process, TemporalHawkesProcess):  # pragma: no cover - guarded above
            raise TypeError(
                f"{self.model.family!r} built a {type(process).__name__}, which has no "
                "temporal intensity hook"
            )
        _bind_history(process, history.upto(upto))
        return process

    def extend(
        self,
        state: LikelihoodState,
        theta: Any,
        history: History,
        upto: float,
    ) -> tuple[LikelihoodState, float]:
        """Account for the block ``(state.upto, upto]``."""
        end = _check_extension(state, history, upto)
        if end == state.upto:
            return state, 0.0

        process = self._process(theta, history, end)
        # The hook itself, not a second expression for the same intensity.
        intensity = process._conditional_intensity

        fresh = history.times[(history.times > state.upto) & (history.times <= end)]
        log_sum = 0.0
        for t in fresh:
            value = intensity(float(t))
            if not value > 0.0:
                # A zero intensity where an event happened is a likelihood of
                # zero, not an error: the parameter simply cannot have produced
                # this data. -inf propagates to a weight of zero.
                return (
                    LikelihoodState(end, state.n_events + fresh.size, -math.inf, ()),
                    -math.inf,
                )
            log_sum += math.log(value)

        edges = _compensator.breakpoints(state.upto, end, history.times, self.extra_lags)
        if self.check and not self._checked:
            self._checked = True
            _compensator.check_resolution(intensity, edges, order=self.order)
        integral = _compensator.integrate(intensity, *_compensator.panels(edges, self.order))

        increment = log_sum - integral
        return (
            LikelihoodState(
                upto=end,
                n_events=state.n_events + int(fresh.size),
                log_lik=state.log_lik + increment,
                carry=(),
            ),
            increment,
        )

    def total(self, theta: Any, history: History, upto: float | None = None) -> float:
        """Return the whole log-likelihood on ``(start, upto]``, defaulting to the window."""
        end = history.end if upto is None else float(upto)
        return self.extend(self.initial_state(history.start), theta, history, end)[0].log_lik

    def compensator(self, theta: Any, history: History, times: Any) -> np.ndarray:
        r"""Evaluate :math:`\Lambda(t) - \Lambda(start)` at each of `times`.

        Non-decreasing by construction: the panels tile ``[start, t]`` with
        strictly positive weights and the intensity is non-negative, so the
        cumulative sum can only grow.
        """
        query = np.asarray(times, dtype=float).ravel()
        if query.size == 0:
            return np.empty(0, dtype=float)
        if np.any(np.diff(query) < 0):
            raise ValueError("times must be sorted for the compensator")

        process = self._process(theta, history, float(query[-1]))
        # The hook itself, not a second expression for the same intensity.
        intensity = process._conditional_intensity

        out = np.empty(query.size, dtype=float)
        running = 0.0
        left = history.start
        for k, right in enumerate(query):
            edges = _compensator.breakpoints(left, float(right), history.times, self.extra_lags)
            running += _compensator.integrate(intensity, *_compensator.panels(edges, self.order))
            out[k] = running
            left = float(right)
        return out


# ---------------------------------------------------------------------------
# Temporal, exponential kernel, in closed form
# ---------------------------------------------------------------------------


class ExponentialLogLikelihood:
    r"""The exact log-likelihood of the linear exponential-kernel Hawkes process.

    The one model whose likelihood has no integral left in it:

    .. math::

        \Lambda(T) = \mu T + \frac{\alpha}{\beta}
                     \sum_{t_i \le T}\left(1 - e^{-\beta(T - t_i)}\right),

    and Ozaki's recursion turns the log-sum into one pass as well. Writing
    :math:`B(a) = \sum_{t_i \le a} e^{-\beta(a - t_i)}`, an event at :math:`t`
    has :math:`\lambda(t^-) = \mu + \alpha B(t^-)`, and :math:`B` advances by a
    single multiplication per step. So a full evaluation is :math:`O(n)` and an
    incremental one is :math:`O(\text{new events})` -- against :math:`O(n^2 P)`
    and :math:`O(nP)` for the general path.

    Scalar Python arithmetic rather than NumPy, deliberately: the recursion is
    sequential, so a vectorized version would be a loop over ``n`` one-element
    array operations, and array dispatch costs more per step than the
    multiplication it wraps.

    Parameters
    ----------
    model : ProcessModel
        Must come from
        :func:`~hawkes_package.inference.models.exponential_model`. The closed
        form is specific to :math:`\alpha e^{-\beta s}` with an additive
        background, and applying it to any other kernel would return a
        plausible number for the wrong model.

    .. versionadded:: 0.5.0
    """

    def __init__(self, model: ProcessModel) -> None:
        if model.family != "exponential" or model.spec.names != ("mu", "alpha", "beta"):
            raise ValueError(
                f"{type(self).__name__} implements the closed form for "
                f"exponential_model()'s (mu, alpha, beta); this model is "
                f"{model.family!r} with parameters {list(model.spec.names)}. Use "
                "TemporalLogLikelihood, which works through the intensity hook."
            )
        self.model = model

    def initial_state(self, start: float) -> LikelihoodState:
        """Return the empty state at `start`, carrying ``B(start) = 0``."""
        return LikelihoodState(upto=float(start), n_events=0, log_lik=0.0, carry=(0.0,))

    def extend(
        self,
        state: LikelihoodState,
        theta: Any,
        history: History,
        upto: float,
    ) -> tuple[LikelihoodState, float]:
        """Account for the block ``(state.upto, upto]`` in one pass over its events."""
        end = _check_extension(state, history, upto)
        if end == state.upto:
            return state, 0.0

        values = np.asarray(theta, dtype=float).reshape(-1)
        mu, alpha, beta = (float(v) for v in values)
        ratio = alpha / beta

        carry = state.carry[0] if state.carry else 0.0
        left = state.upto
        log_sum = 0.0
        integral = 0.0
        fresh = history.times[(history.times > state.upto) & (history.times <= end)]

        for raw in fresh:
            t = float(raw)
            decay = math.exp(-beta * (t - left))
            # The compensator over (left, t]: no event lies strictly inside, so
            # the only term is the decay of what was already excited.
            integral += mu * (t - left) + ratio * carry * (1.0 - decay)
            carry *= decay
            # B has decayed to t but does not yet count the event at t, which is
            # exactly the left limit the log-sum needs.
            log_sum += math.log(mu + alpha * carry)
            carry += 1.0
            left = t

        decay = math.exp(-beta * (end - left))
        integral += mu * (end - left) + ratio * carry * (1.0 - decay)
        carry *= decay

        increment = log_sum - integral
        return (
            LikelihoodState(
                upto=end,
                n_events=state.n_events + int(fresh.size),
                log_lik=state.log_lik + increment,
                carry=(carry,),
            ),
            increment,
        )

    def total(self, theta: Any, history: History, upto: float | None = None) -> float:
        """Return the whole log-likelihood on ``(start, upto]``, defaulting to the window."""
        end = history.end if upto is None else float(upto)
        return self.extend(self.initial_state(history.start), theta, history, end)[0].log_lik

    def compensator(self, theta: Any, history: History, times: Any) -> np.ndarray:
        r"""Evaluate :math:`\Lambda(t) - \Lambda(start)` at each of `times`.

        One merged pass over the events and the query times, so this is
        :math:`O(n + k)` rather than the :math:`O(nk)` the closed form written
        out per query time would cost -- which matters because the caller with
        the most query times is :func:`~hawkes_package.inference.diagnostics.residuals`,
        and it asks for one per event.
        """
        query = np.asarray(times, dtype=float).ravel()
        if query.size == 0:
            return np.empty(0, dtype=float)
        if np.any(np.diff(query) < 0):
            raise ValueError("times must be sorted for the compensator")

        values = np.asarray(theta, dtype=float).reshape(-1)
        mu, alpha, beta = (float(v) for v in values)
        ratio = alpha / beta

        events = history.times
        out = np.empty(query.size, dtype=float)
        carry = 0.0
        left = history.start
        total = 0.0
        index = 0

        for k, raw in enumerate(query):
            target = float(raw)
            while index < events.size and events[index] <= target:
                t = float(events[index])
                decay = math.exp(-beta * (t - left))
                total += mu * (t - left) + ratio * carry * (1.0 - decay)
                carry = carry * decay + 1.0
                left = t
                index += 1
            decay = math.exp(-beta * (target - left))
            total += mu * (target - left) + ratio * carry * (1.0 - decay)
            carry *= decay
            left = target
            out[k] = total
        return out


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


class SpatioTemporalLogLikelihood:
    r"""The log-likelihood of a spatio-temporal model, by hooks or by cached geometry.

    Two backends compute the same number, and which one ran is always recorded.

    ``backend="hooks"`` is the **normative** definition: it builds the process,
    conditions it on the history and calls ``_full_intensity`` and
    ``_integrated_intensity`` -- what the simulator thins against, floor and
    all. It is also unusably slow for a fit. One space integral costs 114 ms on
    a ``Circle`` at 256 nodes and 50 events, so a single log-likelihood at
    n = 200 is about twelve minutes, and a fit is a thousand of those.

    ``backend="cached"`` is the same quantity rearranged around the separability
    of the intensity. Because :math:`\lambda(t, x) = \mu(x) + \sum_i
    \kappa_t(t - t_i)\kappa_s(d(x, x_i))` is a sum, the space integral
    distributes:

    .. math::

        \int_D \lambda(t, x)\,\mathrm{d}x = \int_D \mu
            + \sum_{t_i < t} \kappa_t(t - t_i)\, S_i, \qquad
        S_i = \int_D \kappa_s(d(x, x_i))\,\mathrm{d}x,

    and every :math:`S_i` is a fixed quadrature sum over distances that do not
    depend on ``theta``. Same answer, three milliseconds instead of twelve
    minutes.

    **The rearrangement has a precondition, and it is checked rather than
    assumed.** The process floors the intensity *after* summing --
    ``max(0, mu(x) + sum)`` -- so the identity above holds only where the
    pre-floor integrand is non-negative at every node. Where it is not, the
    cached form over-counts the negative region and the compensator comes out
    too small, which biases the excitation upward: the one direction of error
    this whole module is arranged to prevent. So the cached backend verifies
    that the background and the spatial kernel are non-negative wherever it
    evaluates them, and **raises** rather than degrading. ``backend="auto"``
    falls back to the hooks once, with a warning, and then stays there.

    Parameters
    ----------
    model : ProcessModel
        Must be a spatio-temporal model.
    order : int
        Gauss-Legendre nodes per panel for the time integral.
    backend : {"auto", "cached", "hooks"}
        Which path to use.
    homogeneous : {"auto", True, False}
        Whether to treat every :math:`S_i` as equal. On a homogeneous domain it
        is true to rounding -- the relative spread is 1.4e-16 on a ``Circle`` --
        and the saving is real, because the spatial kernel then has to be
        evaluated at a handful of events' distances rather than all of them. On
        a domain that is a proper subset of its bounding box the spread is not
        zero but the quadrature error, 1.25e-4 on a hexagon at 32 nodes per
        axis, which makes :attr:`spatial_spread` a free accuracy diagnostic
        rather than only a switch.
    geometry : GeometryCache, optional
        A cache to reuse. Built on first use otherwise, and extended in place as
        the history grows.
    rtol : float
        Relative spread above which ``homogeneous="auto"`` declines to use a
        single :math:`\bar{S}`, and warns.
    extra_lags : sequence of float
        As for :class:`TemporalLogLikelihood`.
    max_cache_bytes : int
        Ceiling on the distance tensors.

    Attributes
    ----------
    backend_used : str
        ``"cached"`` or ``"hooks"``, set on the first evaluation.
    homogeneous_used : bool
        Whether a single :math:`\bar{S}` was used.
    spatial_spread : float
        The measured relative spread of :math:`S_i`, or ``nan`` before the first
        evaluation.

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        model: ProcessModel,
        *,
        order: int = _compensator.DEFAULT_ORDER,
        backend: str = "auto",
        homogeneous: bool | str = "auto",
        geometry: GeometryCache | None = None,
        rtol: float = 1e-2,
        extra_lags: Sequence[float] = (),
        max_cache_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if model.ndim == 0 or not isinstance(model.components, SpatialComponents):
            raise ValueError(
                f"{type(self).__name__} is for spatio-temporal models; this one is "
                "temporal. Use TemporalLogLikelihood."
            )
        if backend not in {"auto", "cached", "hooks"}:
            raise ValueError(f"backend must be auto, cached or hooks, got {backend!r}")
        if homogeneous not in {"auto", True, False}:
            raise ValueError(f"homogeneous must be auto, True or False, got {homogeneous!r}")

        self.model = model
        self.components: SpatialComponents = model.components
        self.order = int(order)
        self.backend = backend
        self.homogeneous = homogeneous
        self.rtol = float(rtol)
        self.extra_lags = tuple(float(lag) for lag in extra_lags)
        self.max_cache_bytes = int(max_cache_bytes)

        self._geometry = geometry
        self.backend_used = backend if backend != "auto" else ""
        self.homogeneous_used = False
        self.spatial_spread = math.nan

        n_base = len(self.components.base.spec)
        n_temporal = len(self.components.temporal.spec)
        self._base_slice = slice(0, n_base)
        self._temporal_slice = slice(n_base, n_base + n_temporal)
        self._spatial_slice = slice(n_base + n_temporal, len(model.spec))

    # -- geometry ----------------------------------------------------------

    def geometry_for(self, history: History) -> GeometryCache:
        """Return the distance cache covering `history`, built or extended as needed."""
        points = _located(history)
        if self._geometry is None:
            self._geometry = build_geometry(
                self.components, history.times, points, max_bytes=self.max_cache_bytes
            )
        else:
            # Unconditionally, not only when the count changed. The prefix check
            # lives inside extend_geometry, so gating the call on a length
            # difference let a second history of the *same* length reuse distances
            # built for different events -- wrong tensors, nothing raised. The call
            # costs nothing when there is nothing new: extend_geometry returns the
            # cache unchanged once `matches` has passed.
            self._geometry = extend_geometry(
                self._geometry,
                self.components,
                history.times,
                points,
                max_bytes=self.max_cache_bytes,
            )
        return self._geometry

    # -- the cached path ---------------------------------------------------

    def _spatial_masses(self, cache: GeometryCache, theta: np.ndarray) -> np.ndarray:
        r"""Return the per-event spatial masses :math:`S_i`, recording their spread.

        Under ``homogeneous`` the kernel is evaluated at a deterministic slice of
        events rather than a random one. A Monte Carlo estimate of a quantity
        this is multiplied by would be unbiased and not dominating, and this
        package has been bitten by that distinction before.
        """
        kernel = self.components.spatial.build(theta[self._spatial_slice])
        n = cache.n_events
        if n == 0:
            self.spatial_spread = 0.0
            self.homogeneous_used = bool(self.homogeneous is True)
            return np.empty(0, dtype=float)

        wants_single = self.homogeneous is not False
        probe = np.unique(np.linspace(0, n - 1, min(n, 8)).astype(int)) if wants_single else None

        if probe is not None:
            sampled = self._masses_at(kernel, cache, probe)
            scale = max(abs(float(np.mean(sampled))), np.finfo(float).tiny)
            self.spatial_spread = float(np.ptp(sampled)) / scale
            if self.homogeneous is True or self.spatial_spread <= self.rtol:
                self.homogeneous_used = True
                return np.full(n, float(np.mean(sampled)), dtype=float)
            warnings.warn(
                f"the spatial mass varies by {100 * self.spatial_spread:.3g}% across "
                f"events, above rtol={self.rtol}, so a single value cannot stand for all "
                "of them. Falling back to a mass per event, which is exact and still far "
                "cheaper than the hooks. On a domain that fills its bounding box this "
                "spread is the quadrature error, so raising n_quad is the other fix.",
                UserWarning,
                stacklevel=4,
            )

        self.homogeneous_used = False
        masses = self._masses_at(kernel, cache, np.arange(n))
        if probe is None:
            scale = max(abs(float(np.mean(masses))), np.finfo(float).tiny)
            self.spatial_spread = float(np.ptp(masses)) / scale
        return masses

    def _masses_at(self, kernel: Any, cache: GeometryCache, which: np.ndarray) -> np.ndarray:
        """Integrate the spatial kernel around the events indexed by `which`."""
        block = np.asarray(kernel(cache.node_event[:, which, :]), dtype=float)
        self._require_non_negative(block, "the spatial kernel")
        return np.asarray(np.einsum("j,jig->i", cache.weights, block), dtype=float)

    def _check_kernel_sign(self, kernel: Any, cache: GeometryCache) -> None:
        """Check the spatial kernel over the whole range of distances in play.

        Not only over the distances the masses happen to be evaluated at: under
        ``homogeneous`` those come from a handful of sampled events, and a kernel
        that dips negative near an event the sample missed would pass. The
        precondition is about the kernel on ``[0, max distance]``, so that is
        what is checked -- on a grid dense enough that a dip narrow enough to
        hide between its points is also too narrow for the quadrature rule that
        would have to integrate it.
        """
        if cache.node_event.size == 0:
            return
        span = float(np.max(cache.node_event))
        probe = np.linspace(0.0, span, 4096)
        self._require_non_negative(np.asarray(kernel(probe), dtype=float), "the spatial kernel")

    def _require_non_negative(self, values: np.ndarray, what: str) -> None:
        """Refuse the cached rearrangement where its precondition fails."""
        worst = float(np.min(values)) if values.size else 0.0
        if worst < 0.0:
            raise ValueError(
                f"{what} reaches {worst:.6g} on this history, and the cached backend's "
                "separability identity holds only where the pre-floor integrand is "
                "non-negative at every quadrature node -- the process floors the "
                "intensity after summing, so below zero the cached form over-counts and "
                "the compensator comes out too small, biasing the excitation upward. "
                'Use backend="hooks", which is the definition, or a non-negative kernel.'
            )

    def _cached_terms(
        self, theta: np.ndarray, history: History
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Per-event intensities, per-event spatial masses, and the background integral."""
        cache = self.geometry_for(history)
        base = self.components.base
        temporal = self.components.temporal
        spatial = self.components.spatial

        background = np.asarray(base.at(theta[self._base_slice], cache.nodes), dtype=float)
        self._require_non_negative(background, "the background intensity")
        background_integral = float(np.dot(cache.weights, background))

        self._check_kernel_sign(spatial.build(theta[self._spatial_slice]), cache)
        masses = self._spatial_masses(cache, theta)

        kappa_t = temporal.build(theta[self._temporal_slice])
        kappa_s = spatial.build(theta[self._spatial_slice])

        times = history.times
        n = times.size
        if n == 0:
            return np.empty(0, dtype=float), masses, background_integral

        # log-sum term: lambda(t_i^-, x_i), strictly over earlier events.
        pair = np.asarray(kappa_s(cache.event_event), dtype=float).sum(axis=2)
        lags = times[:, None] - times[None, :]
        earlier = lags > 0.0
        factors = np.where(earlier, np.asarray(kappa_t(np.where(earlier, lags, 0.0))), 0.0)
        at_events = np.asarray(
            base.at(theta[self._base_slice], _located(history).T), dtype=float
        ) + np.sum(factors * pair, axis=1)
        return at_events, masses, background_integral

    def _integrated_intensity_at(
        self,
        nodes: np.ndarray,
        times: np.ndarray,
        masses: np.ndarray,
        kappa_t: Any,
        background_integral: float,
    ) -> np.ndarray:
        """Evaluate the space-integrated intensity at every time in `nodes`, in chunks."""
        out = np.empty(nodes.size, dtype=float)
        for start in range(0, nodes.size, _NODE_CHUNK):
            stop = min(start + _NODE_CHUNK, nodes.size)
            lags = nodes[start:stop, None] - times[None, :]
            earlier = lags > 0.0
            factors = np.where(earlier, np.asarray(kappa_t(np.where(earlier, lags, 0.0))), 0.0)
            out[start:stop] = background_integral + factors @ masses
        return out

    # -- the hooks path ----------------------------------------------------

    def _process(self, theta: Any, history: History, upto: float) -> SpatioTemporalHawkesProcess:
        """Build the process at `theta` and condition it on the history so far."""
        process = self.model(theta, rng=0)
        if not isinstance(process, SpatioTemporalHawkesProcess):  # pragma: no cover
            raise TypeError(
                f"{self.model.family!r} built a {type(process).__name__}, which has no "
                "space-integrated intensity hook"
            )
        _bind_history(process, history.upto(upto))
        return process

    # -- the protocol ------------------------------------------------------

    def initial_state(self, start: float) -> LikelihoodState:
        """Return the empty state at `start`."""
        return LikelihoodState(upto=float(start), n_events=0, log_lik=0.0, carry=())

    def extend(
        self,
        state: LikelihoodState,
        theta: Any,
        history: History,
        upto: float,
    ) -> tuple[LikelihoodState, float]:
        """Account for the block ``(state.upto, upto]``."""
        end = _check_extension(state, history, upto)
        if end == state.upto:
            return state, 0.0

        values = np.asarray(theta, dtype=float).reshape(-1)
        window = (history.times > state.upto) & (history.times <= end)

        if self._resolve_backend(values, history) == "cached":
            log_sum, integral = self._cached_block(values, history, state.upto, end, window)
        else:
            log_sum, integral = self._hooks_block(values, history, state.upto, end, window)

        increment = log_sum - integral
        return (
            LikelihoodState(
                upto=end,
                n_events=state.n_events + int(np.count_nonzero(window)),
                log_lik=state.log_lik + increment,
                carry=(),
            ),
            increment,
        )

    def _resolve_backend(self, theta: np.ndarray, history: History) -> str:
        """Decide, once, which path this object uses, and say so if it fell back."""
        if self.backend_used:
            return self.backend_used
        try:
            self._cached_terms(theta, history)
        except ValueError as failure:
            warnings.warn(
                f'backend="auto" fell back to the hooks and will stay there: {failure} '
                "The hooks are the definition, so the answer is right -- but the fit is "
                "now several orders of magnitude slower, which is worth knowing before "
                "it runs overnight.",
                UserWarning,
                stacklevel=4,
            )
            self.backend_used = "hooks"
            return "hooks"
        self.backend_used = "cached"
        return "cached"

    def _cached_block(
        self,
        theta: np.ndarray,
        history: History,
        left: float,
        right: float,
        window: np.ndarray,
    ) -> tuple[float, float]:
        """Return the two terms of the block increment, from cached geometry."""
        at_events, masses, background_integral = self._cached_terms(theta, history)
        selected = at_events[window]
        if selected.size and not np.all(selected > 0.0):
            return -math.inf, 0.0
        log_sum = float(np.sum(np.log(selected))) if selected.size else 0.0

        kappa_t = self.components.temporal.build(theta[self._temporal_slice])
        edges = _compensator.breakpoints(left, right, history.times, self.extra_lags)
        nodes, weights = _compensator.panels(edges, self.order)
        if nodes.size == 0:
            return log_sum, 0.0
        rate = self._integrated_intensity_at(
            nodes, history.times, masses, kappa_t, background_integral
        )
        return log_sum, float(np.dot(weights, rate))

    def _hooks_block(
        self,
        theta: np.ndarray,
        history: History,
        left: float,
        right: float,
        window: np.ndarray,
    ) -> tuple[float, float]:
        """Return the two terms of the block increment, through the process's own hooks."""
        process = self._process(theta, history, right)
        log_sum = 0.0
        for t, x in zip(history.times[window], _located(history).T[window], strict=True):
            value = process._full_intensity(x, float(t))
            if not value > 0.0:
                return -math.inf, 0.0
            log_sum += math.log(value)

        edges = _compensator.breakpoints(left, right, history.times, self.extra_lags)
        integral = _compensator.integrate(
            process._integrated_intensity,
            *_compensator.panels(edges, self.order),
        )
        return log_sum, integral

    def total(self, theta: Any, history: History, upto: float | None = None) -> float:
        """Return the whole log-likelihood on ``(start, upto]``, defaulting to the window."""
        end = history.end if upto is None else float(upto)
        return self.extend(self.initial_state(history.start), theta, history, end)[0].log_lik

    def compensator(self, theta: Any, history: History, times: Any) -> np.ndarray:
        r"""Evaluate :math:`\Lambda(t) - \Lambda(start)` at each of `times`."""
        query = np.asarray(times, dtype=float).ravel()
        if query.size == 0:
            return np.empty(0, dtype=float)
        if np.any(np.diff(query) < 0):
            raise ValueError("times must be sorted for the compensator")

        values = np.asarray(theta, dtype=float).reshape(-1)
        out = np.empty(query.size, dtype=float)
        running = 0.0
        left = history.start

        if self._resolve_backend(values, history) == "cached":
            _, masses, background_integral = self._cached_terms(values, history)
            kappa_t = self.components.temporal.build(values[self._temporal_slice])
            for k, right in enumerate(query):
                edges = _compensator.breakpoints(left, float(right), history.times, self.extra_lags)
                nodes, weights = _compensator.panels(edges, self.order)
                if nodes.size:
                    rate = self._integrated_intensity_at(
                        nodes, history.times, masses, kappa_t, background_integral
                    )
                    running += float(np.dot(weights, rate))
                out[k] = running
                left = float(right)
            return out

        process = self._process(values, history, float(query[-1]))
        for k, right in enumerate(query):
            edges = _compensator.breakpoints(left, float(right), history.times, self.extra_lags)
            running += _compensator.integrate(
                process._integrated_intensity,
                *_compensator.panels(edges, self.order),
            )
            out[k] = running
            left = float(right)
        return out
