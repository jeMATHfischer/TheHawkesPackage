r"""Maximum likelihood, beside the sequential machinery rather than instead of it.

The package's position is that the sequential Monte Carlo path is the recommended
one: it reports a posterior, it absorbs data in blocks, and its diagnostics say
when it has failed. This module exists for two things that position cannot
supply. A reviewer comparing point-process libraries runs a maximum-likelihood
fit, and "we do SMC instead" is not an answer they can check. And a mode is a
good place to centre an initial cloud, which
:func:`~hawkes_package.inference.mle.warm_start_prior` turns into a real saving.

**SciPy is used here, and that widens a stated rule.** Runtime dependencies are
still numpy and scipy only, but until 0.10.0 scipy was held to a single call
site -- ``minimize_scalar`` in :mod:`hawkes_package._numerics` -- which is why
:func:`~hawkes_package.inference.diagnostics.ks_exponential` hand-rolls the
Kolmogorov series and :mod:`hawkes_package.inference.priors` hand-writes its
marginals. The rule is now that scipy is reached for where a hand-written version
would be *worse*, not wherever it is convenient. An optimiser is squarely on that
side; a twenty-line series is not, and those stay hand-written.

Three things here are not obvious, and each is a way an MLE goes quietly wrong.

**The Jacobian is deliberately absent.** The optimisation runs on the
unconstrained scale, through
:meth:`~hawkes_package.inference.parameters.ParameterSpec.to_constrained`, and
the log-determinant of that transform is *not* added. Adding it would be natural
-- the sampler does it -- and would silently return the mode of a density under a
flat prior on the unconstrained scale, which is a different estimator wearing the
same name.

**An unconstrained maximiser has no prior to protect it from a bad compensator.**
Every unit of :math:`\int\lambda` that goes missing is a penalty on a high
intensity that never gets applied, so the optimiser takes it: the background and
the excitation both come back large and the fit looks excellent. So the
compensator's resolution check is re-run **at the optimum**, where the integrand
is sharpest, and not only at the starting value.

**An inverse Hessian is not an interval.** At an optimum near the stationarity
boundary a differenced Hessian can come back indefinite, and the standard error
it reports is a plausible number with no content -- the same failure as a
collapsed particle cloud reading as confidence. :func:`profile_interval` walks
the likelihood instead.

.. versionadded:: 0.10.0
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .likelihood import History, LogLikelihood
from .models import ProcessModel
from .parameters import ParameterSpec
from .priors import IndependentPrior, Normal, Prior

__all__ = [
    "MaximumLikelihoodFit",
    "fit_mle",
    "profile_interval",
    "warm_start_prior",
]

#: Half of the chi-squared(1) quantile at a few standard levels, which is the
#: drop in log-likelihood that bounds a profile interval. Tabulated rather than
#: taken from `scipy.stats`: three numbers do not justify the import, and the
#: widened rule above is about optimisers, not about everything.
_CHI2_HALF = {0.90: 1.352772, 0.95: 1.920729, 0.99: 3.317448}

#: What the objective returns outside the support. **Finite on purpose**, though
#: `inf` is the honest value: Nelder-Mead's own convergence test computes
#: ``max|f[0] - f[1:]|``, which is ``inf - inf`` and therefore `nan` as soon as
#: two simplex vertices sit outside, and numpy warns. A suite running
#: ``filterwarnings = ["error"]`` -- this one, and any careful downstream one --
#: then fails inside SciPy, on a fit that was going perfectly well.
#:
#: A minimiser treats a large value as a wall rather than a direction, and 1e12
#: is roughly eight orders above the largest log-likelihood any realistic history
#: produces, so nothing outside the support can ever win. The returned optimum is
#: checked against the support anyway, because "can never win" is an argument and
#: the check is a fact.
_OUTSIDE = 1e12


@dataclass(frozen=True)
class MaximumLikelihoodFit:
    """The result of :func:`fit_mle`.

    Attributes
    ----------
    theta : numpy.ndarray
        The maximiser, on the model's own (constrained) scale.
    log_likelihood : float
        Its log-likelihood, in the same units
        :meth:`~hawkes_package.inference.likelihood.LogLikelihood.total` returns.
    n_evaluations : int
        Objective evaluations the optimiser used. Worth reading: a fit that took
        thousands is one whose objective is rough, usually a compensator that is
        not resolved.
    converged : bool
        What the optimiser reported. ``False`` is accompanied by a warning at fit
        time -- see :func:`fit_mle`.
    message : str
        The optimiser's own account of why it stopped.
    spec : ParameterSpec
        The coordinates, so a result can name its own numbers.
    """

    theta: np.ndarray
    log_likelihood: float
    n_evaluations: int
    converged: bool
    message: str
    spec: ParameterSpec
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def named(self) -> dict[str, float]:
        """Return the estimate as ``{name: value}``."""
        return {name: float(v) for name, v in zip(self.spec.names, self.theta, strict=True)}

    def __repr__(self) -> str:
        """Show the estimate by name, which is what a reader wants."""
        body = ", ".join(f"{k}={v:.4g}" for k, v in self.named().items())
        state = "converged" if self.converged else "NOT CONVERGED"
        return f"MaximumLikelihoodFit({body}, log_lik={self.log_likelihood:.4f}, {state})"


def _model_of(likelihood: LogLikelihood) -> ProcessModel:
    """Return the model a likelihood was built for."""
    model = getattr(likelihood, "model", None)
    if model is None:
        raise ValueError(
            f"{type(likelihood).__name__} does not carry a `model`, so its "
            "parameters have no names, bounds or support to optimise inside"
        )
    return model


def _objective(likelihood: LogLikelihood, history: History, model: ProcessModel) -> Any:
    """Return the negative log-likelihood as a function of the unconstrained vector."""
    spec = model.spec

    def negative(z: np.ndarray) -> float:
        theta = spec.to_constrained(np.asarray(z, dtype=float))
        if not bool(np.asarray(model.support(theta)).ravel()[0]):
            return _OUTSIDE
        value = float(likelihood.total(theta, history))
        # A `-inf` log-likelihood -- a mark below its threshold, an intensity
        # that reached zero at an event -- is the same situation as being outside
        # the support, and gets the same wall for the same reason.
        return -value if math.isfinite(value) else _OUTSIDE

    return negative


def fit_mle(
    likelihood: LogLikelihood,
    history: History,
    start: Any,
    *,
    method: str = "Nelder-Mead",
    max_iterations: int | None = None,
    tol: float | None = None,
    recheck_resolution: bool = True,
) -> MaximumLikelihoodFit:
    r"""Maximise `likelihood` over the model's parameters.

    The objective is the same
    :meth:`~hawkes_package.inference.likelihood.LogLikelihood.total` the
    sequential sampler uses -- never a second expression of the intensity, which
    is the documented way this package has gone wrong before.

    Parameters
    ----------
    likelihood : LogLikelihood
        Any of the likelihoods, carrying the model whose parameters are fitted.
    history : History
        The data.
    start : array_like
        Starting parameters, on the model's own scale. Must lie inside the
        support: a start the process cannot be built at is refused here rather
        than becoming an ``inf`` the optimiser cannot escape from.
    method : str
        Passed to :func:`scipy.optimize.minimize`. The default is
        ``"Nelder-Mead"`` on purpose. The objective is ``inf`` outside the
        support, and a gradient-based method differencing across that boundary
        takes a step nobody can interpret, where a simplex reflects away from it.
    max_iterations : int, optional
        Iteration cap. ``None`` leaves the method's own default.
    tol : float, optional
        Passed through as the method's tolerance.
    recheck_resolution : bool
        Re-run the compensator's order-``P``-versus-``2P`` check at the optimum.
        Defaults to ``True``, and the reason is in the module docstring: an
        unconstrained maximiser has no prior to protect it from an
        under-integrated compensator, which rewards exactly the parameters it
        should penalise.

    Returns
    -------
    MaximumLikelihoodFit

    Raises
    ------
    ValueError
        If `start` lies outside the model's support.

    Warns
    -----
    UserWarning
        If the optimiser reports that it did not converge. Returning the last
        iterate silently is how a failed fit gets published, and the suite runs
        with ``filterwarnings = ["error"]`` so that a downstream test sees it.

    Examples
    --------
    >>> import numpy as np
    >>> import hawkes_package as hp
    >>> from hawkes_package.inference import (
    ...     ExponentialLogLikelihood,
    ...     History,
    ...     exponential_model,
    ...     fit_mle,
    ... )
    >>> process = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=0)
    >>> process.simulate(400)
    >>> history = History.from_simulation(process)
    >>> model = exponential_model()
    >>> fit = fit_mle(ExponentialLogLikelihood(model), history, [1.0, 0.3, 1.0])
    >>> fit.converged
    True

    .. versionadded:: 0.10.0
    """
    model = _model_of(likelihood)
    spec = model.spec
    theta0 = np.asarray(start, dtype=float).reshape(-1)
    if theta0.size != len(spec):
        raise ValueError(
            f"start has {theta0.size} entries but this model has {len(spec)}: {list(spec.names)}"
        )
    if not bool(np.asarray(model.support(theta0)).ravel()[0]):
        raise ValueError(
            f"start={theta0.tolist()} is outside the model's support, so the "
            "objective is infinite there and the optimiser has no direction to "
            "move in. Start from a parameter the process can be built at."
        )

    negative = _objective(likelihood, history, model)
    options: dict[str, Any] = {}
    if max_iterations is not None:
        options["maxiter"] = int(max_iterations)

    result = minimize(
        negative,
        spec.to_unconstrained(theta0),
        method=method,
        tol=tol,
        options=options or None,
    )

    theta = spec.to_constrained(np.asarray(result.x, dtype=float))
    inside = bool(np.asarray(model.support(theta)).ravel()[0])
    if not inside:
        # Unreachable by the argument at `_OUTSIDE`, and checked anyway: the
        # alternative is returning parameters the process cannot be built at,
        # which every downstream call would then fail on with a message about
        # something else.
        warnings.warn(
            f"the optimiser finished outside the model's support at "
            f"{np.round(theta, 4).tolist()}. This is a fit to discard, not a "
            "parameter to use.",
            UserWarning,
            stacklevel=2,
        )
    if not result.success:
        warnings.warn(
            f"the optimiser did not converge: {result.message} It returned "
            f"{np.round(theta, 4).tolist()} after {int(result.nfev)} evaluations, "
            "which is the last iterate and not a maximum. Raise max_iterations, "
            "or start closer.",
            UserWarning,
            stacklevel=2,
        )

    if recheck_resolution:
        _recheck(likelihood, theta, history)

    return MaximumLikelihoodFit(
        theta=theta,
        log_likelihood=-float(result.fun),
        n_evaluations=int(result.nfev),
        converged=bool(result.success) and inside,
        message=str(result.message),
        spec=spec,
    )


def _recheck(likelihood: LogLikelihood, theta: np.ndarray, history: History) -> None:
    """Re-run the compensator's resolution check at the fitted parameters.

    The check is a once-per-likelihood affair by design -- it doubles the
    quadrature work -- so it normally fires at whatever parameters happened to be
    evaluated first. Those are the *starting* parameters, and the integrand there
    is not the one the answer was read off. Resetting the flag and evaluating
    once more puts the check where the optimiser left the fit.
    """
    checked = getattr(likelihood, "_checked", None)
    if checked is None or not getattr(likelihood, "check", False):
        return
    # The flag belongs to this check: it exists so the doubling runs once, and
    # running it once *at the optimum* is more useful than once at the start.
    likelihood._checked = False
    likelihood.total(theta, history)


def profile_interval(
    likelihood: LogLikelihood,
    history: History,
    fit: MaximumLikelihoodFit,
    name: str,
    *,
    level: float = 0.95,
    span: float = 4.0,
    n_grid: int = 24,
) -> tuple[float, float]:
    r"""Return a profile-likelihood interval for one coordinate.

    Walks `name` away from the maximum in both directions, re-maximising every
    other coordinate at each step, and reports where the log-likelihood has
    dropped by :math:`\chi^2_1(\text{level}) / 2`.

    **Not an inverse Hessian**, and that is the point. A numerically differenced
    Hessian at an optimum near the stationarity boundary can come back
    indefinite, and the standard error it produces is a plausible-looking number
    with no content. A profile interval is asymmetric where the likelihood is
    asymmetric, which near a boundary it always is.

    Each step **warm-starts the inner maximisation from the previous step's
    solution**, which is not only faster. Restarting it from the maximum every
    time leaves the profile carrying the inner optimiser's own noise -- measured
    at 0.35 log-likelihood units on a 400-event exponential fit, against a
    threshold of 1.92 -- and a walk that stops at the first crossing would then
    stop wherever that noise dipped.

    Parameters
    ----------
    likelihood : LogLikelihood
        The same objective the fit used.
    history : History
        The same data.
    fit : MaximumLikelihoodFit
        The maximum to walk out from.
    name : str
        Which coordinate to profile.
    level : float
        One of 0.90, 0.95 or 0.99.
    span : float
        How far to search, as a multiple of the estimate. **A returned endpoint
        at the search boundary means the data did not bound this coordinate**,
        which is information rather than a failure -- for an exponential kernel
        it is the ordinary state of ``alpha``, whose lower profile at 400 events
        drops only 0.85 by the time ``alpha`` has fallen a hundredfold, because
        ``beta`` follows it down and the branching ratio is what the data
        actually pins.
    n_grid : int
        Steps per side.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` on the model's own scale.

    .. versionadded:: 0.10.0
    """
    if level not in _CHI2_HALF:
        raise ValueError(f"level must be one of {sorted(_CHI2_HALF)}, got {level!r}")
    model = _model_of(likelihood)
    spec = model.spec
    index = spec.index(name)
    target = fit.log_likelihood - _CHI2_HALF[level]
    centre = float(fit.theta[index])
    if not centre > 0.0 and spec.lower[index] == 0.0:
        raise ValueError(f"the fitted {name} is {centre}, which has no interval to walk")

    others = [i for i in range(len(spec)) if i != index]
    at_optimum = spec.to_unconstrained(fit.theta)

    def maximise_others(value: float, warm: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the profile at `value`, and the free coordinates that gave it."""
        if not others:
            theta = np.array([value], dtype=float)
            inside = bool(np.asarray(model.support(theta)).ravel()[0])
            return (float(likelihood.total(theta, history)) if inside else -math.inf), warm

        def negative(free: np.ndarray) -> float:
            filled = at_optimum.copy()
            filled[others] = free
            theta = spec.to_constrained(filled)
            theta[index] = value
            if not bool(np.asarray(model.support(theta)).ravel()[0]):
                return _OUTSIDE
            got = float(likelihood.total(theta, history))
            return -got if math.isfinite(got) else _OUTSIDE

        result = minimize(negative, warm, method="Nelder-Mead")
        return -float(result.fun), np.asarray(result.x, dtype=float)

    return (
        _walk(maximise_others, centre, target, spec, index, -1, span, n_grid, at_optimum[others]),
        _walk(maximise_others, centre, target, spec, index, +1, span, n_grid, at_optimum[others]),
    )


def _walk(
    maximise_others: Any,
    centre: float,
    target: float,
    spec: ParameterSpec,
    index: int,
    direction: int,
    span: float,
    n_grid: int,
    free: np.ndarray,
) -> float:
    """Step out from `centre` until the profile drops below `target`.

    Linear interpolation on the bracket rather than a root solve. The profile is
    itself the output of an optimiser, so it carries that optimiser's tolerance
    as noise, and bisecting would chase it to a precision it does not have.
    """
    bound = spec.lower[index] if direction < 0 else spec.upper[index]
    if direction > 0:
        reach = centre * (1.0 + span)
        if math.isfinite(bound):
            reach = min(reach, centre + 0.999 * (bound - centre))
    else:
        reach = centre * (1.0 - min(span, 0.99))
        if math.isfinite(bound):
            reach = max(reach, bound + 0.001 * (centre - bound))

    previous_value = centre
    previous_profile, free = maximise_others(centre, free)
    for step in range(1, n_grid + 1):
        value = centre + (reach - centre) * (step / n_grid)
        current, free = maximise_others(value, free)
        if current < target:
            gap = previous_profile - current
            weight = 0.0 if gap <= 0 else (previous_profile - target) / gap
            return float(previous_value + weight * (value - previous_value))
        previous_value, previous_profile = value, current
    return float(reach)


def warm_start_prior(
    fit: MaximumLikelihoodFit,
    *,
    width: float = 0.5,
) -> Prior:
    """Return a prior centred on the maximum, on the unconstrained scale.

    The strongest practical reason to have an MLE at all: an SMC run initialised
    from the mode spends none of its early blocks travelling there. The prior is
    normal on the **unconstrained** scale, which is where the sampler's own
    proposals live, so `width` is in the same units the rejuvenation step's
    scale is.

    This is a *prior*, so it is a modelling statement and not a free lunch: it
    says the truth is near the maximum, and it is exactly as wrong as that is.
    Use it to save time on data you have already looked at, not to report a
    posterior you would defend as uninformative.

    Parameters
    ----------
    fit : MaximumLikelihoodFit
        The maximum to centre on.
    width : float
        Standard deviation per coordinate, on the unconstrained scale.

    .. versionadded:: 0.10.0
    """
    if not fit.converged:
        warnings.warn(
            "warm-starting from a fit that did not converge centres the cloud on "
            "the optimiser's last iterate, which is not a maximum.",
            UserWarning,
            stacklevel=2,
        )
    centre = fit.spec.to_unconstrained(fit.theta)
    return _UnconstrainedNormalPrior(fit.spec, centre, float(width))


@dataclass(frozen=True)
class _UnconstrainedNormalPrior:
    """A normal cloud on the unconstrained scale, mapped back through the spec."""

    spec: ParameterSpec
    centre: np.ndarray
    width: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` parameter vectors, on the model's own scale."""
        generator = np.random.default_rng(rng)
        z = self.centre[None, :] + self.width * generator.standard_normal(
            (int(n), self.centre.size)
        )
        return np.asarray(self.spec.to_constrained(z), dtype=float)

    def log_pdf(self, theta: Any) -> np.ndarray:
        """Log density on the model's own scale, **with** the Jacobian.

        Included here and deliberately absent from the optimiser's objective,
        and the difference is the whole distinction this module rests on: a
        prior is a density over parameters and has to transform like one, while
        a likelihood is not and must not, or the maximum becomes a MAP estimate
        under a flat prior on the unconstrained scale.
        """
        values = np.atleast_2d(np.asarray(theta, dtype=float))
        z = self.spec.to_unconstrained(values)
        gaussian = IndependentPrior(tuple(Normal(float(m), self.width) for m in self.centre))
        density = np.asarray(gaussian.log_pdf(z), dtype=float)
        return np.asarray(density - self.spec.log_abs_det_jacobian(z), dtype=float)
