"""Parameterised kernels, backgrounds and nonlinearities.

The simulation side takes plain callables: a kernel is whatever function the
caller hands it. Inference needs more than a callable, because it has to build
one *per particle, per move* and then say things about it in closed form. A
family is that extra structure -- a :class:`ParameterSpec`, a builder, and two
summaries the sampler cannot afford to compute numerically:

* :meth:`~KernelFamily.peak`, because
  :class:`~hawkes_package.bell_shape.BellShapeHawkes` and the non-monotone
  spatio-temporal path locate the kernel's maximum at construction with
  :func:`~hawkes_package._numerics.locate_peak`, a 513-point scan with a bounded
  refinement. That is the right thing to do for a kernel nobody can
  differentiate, and the wrong thing to do 512 times per rejuvenation move for a
  kernel whose maximum is one line of algebra. Every family here returns the
  algebra, and the tests check it against the scan.
* :meth:`~KernelFamily.mass`, the integral over all lags, because the branching
  ratio decides whether a parameter is one the process can be *simulated* at.
  Guessing it wrong does not raise: the particle survives, the likelihood on a
  finite window is finite, and only the excitation estimate is wrong.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._numerics import PeakLocation
from .parameters import Parameter, ParameterSpec

__all__ = [
    "BaseFamily",
    "ConstantBase",
    "ExponentialKernel",
    "GammaKernel",
    "GaussianSpatial",
    "KernelFamily",
    "LinearNonlinearity",
    "NonlinearityFamily",
    "SoftPlusNonlinearity",
    "SpatialKernelFamily",
]


def _batch(theta: Any, n_dim: int) -> tuple[np.ndarray, bool]:
    """Coerce to ``(n, n_dim)``, reporting whether the input was a single vector."""
    array = np.asarray(theta, dtype=float)
    flat = array.ndim == 1
    if flat:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != n_dim:
        raise ValueError(f"expected {n_dim} parameter(s), got shape {np.shape(theta)}")
    return array, flat


def _unbatch(values: np.ndarray, flat: bool) -> np.ndarray:
    """Undo :func:`_batch`'s promotion."""
    return values[0] if flat else values


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class KernelFamily(Protocol):
    """A parameterised temporal excitation kernel."""

    @property
    def spec(self) -> ParameterSpec:
        """The kernel's own parameters, in the order it consumes them."""
        ...

    @property
    def monotone(self) -> bool:
        """Whether the kernel decreases from lag zero onwards."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel at one parameter vector, vectorized over lags."""
        ...

    def peak(self, theta: Any) -> PeakLocation:
        """Where the kernel attains its maximum, and the value there."""
        ...

    def mass(self, theta: Any) -> np.ndarray:
        """Integral of the kernel over ``[0, inf)``, batched over `theta`."""
        ...


@runtime_checkable
class SpatialKernelFamily(Protocol):
    """A parameterised isotropic spatial kernel, evaluated at a distance."""

    @property
    def spec(self) -> ParameterSpec:
        """The kernel's own parameters."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel at one parameter vector, vectorized over distances."""
        ...

    def mass(self, theta: Any) -> np.ndarray:
        """Integral over the whole model space, batched over `theta`."""
        ...

    def min_scale(self, theta: Any) -> np.ndarray:
        """Return the smallest length the kernel varies on, batched over `theta`."""
        ...


@runtime_checkable
class BaseFamily(Protocol):
    """A parameterised background intensity."""

    @property
    def spec(self) -> ParameterSpec:
        """The background's own parameters."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the background at one parameter vector, as a function of position."""
        ...

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate the background at `points`, shape ``(m, ndim)`` -> ``(m,)``."""
        ...


@runtime_checkable
class NonlinearityFamily(Protocol):
    """A parameterised monotone-increasing map applied to the excitation sum."""

    @property
    def spec(self) -> ParameterSpec:
        """The nonlinearity's own parameters."""
        ...

    @property
    def lipschitz(self) -> float:
        """A Lipschitz constant, which bounds the stability condition."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        ...


# ---------------------------------------------------------------------------
# Temporal kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExponentialKernel:
    r"""The kernel :math:`\kappa(s) = \alpha e^{-\beta s}`.

    Written in the same convention
    :class:`~hawkes_package.exponential.ExponentialHawkes` uses, so ``alpha`` is
    the jump in intensity an event causes and the branching ratio is
    ``alpha / beta`` rather than ``alpha``. Keeping the two conventions apart
    matters: reading ``alpha`` as a branching ratio understates the excitation
    by a factor of ``beta``, and nothing about the fitted numbers says so.

    Monotone decreasing, so its maximum is at lag zero.
    """

    monotone: bool = True

    @property
    def spec(self) -> ParameterSpec:
        """``(alpha, beta)``, both positive."""
        return ParameterSpec((Parameter("alpha"), Parameter("beta")))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 2)
        alpha, beta = float(values[0, 0]), float(values[0, 1])
        return lambda s: alpha * np.exp(-beta * np.asarray(s, dtype=float))

    def peak(self, theta: Any) -> PeakLocation:
        """Return the maximum, at lag ``0`` with value ``alpha``."""
        values, _ = _batch(theta, 2)
        return PeakLocation(lag=0.0, value=float(values[0, 0]))

    def mass(self, theta: Any) -> np.ndarray:
        """``alpha / beta``, the branching ratio of the linear process."""
        values, flat = _batch(theta, 2)
        return _unbatch(values[:, 0] / values[:, 1], flat)


@dataclass(frozen=True)
class GammaKernel:
    r"""The kernel :math:`\kappa(s) = \alpha\, b^k s^{k-1} e^{-b s} / \Gamma(k)`.

    A gamma density scaled by ``alpha``, so ``alpha`` **is** the branching ratio
    -- the mass is ``alpha`` whatever the shape. With ``shape > 1`` it rises from
    zero, peaks at ``(shape - 1) / rate`` and decays: the bell shape
    :class:`~hawkes_package.bell_shape.BellShapeHawkes` exists for, and the one
    whose thinning bound is invalid if the peak is mislocated.

    The shape is bounded below by 1 rather than by 0. Below 1 the kernel
    diverges at lag zero, and an unbounded kernel has no thinning bound at all.
    """

    monotone: bool = False

    @property
    def spec(self) -> ParameterSpec:
        """``(alpha, shape, rate)``; ``shape`` above 1, the others above 0."""
        return ParameterSpec((Parameter("alpha"), Parameter("shape", lower=1.0), Parameter("rate")))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 3)
        alpha, shape, rate = (float(v) for v in values[0])
        log_norm = math.log(alpha) + shape * math.log(rate) - math.lgamma(shape)

        def kernel(s: Any) -> np.ndarray:
            lags = np.asarray(s, dtype=float)
            positive = lags > 0.0
            # `log` is evaluated at 1 rather than at 0 off the support: masking
            # the *result* still evaluates the expression everywhere, so a bare
            # log(0) would emit a divide-by-zero warning -- which this suite
            # turns into a failure -- before `where` discarded it.
            safe = np.where(positive, lags, 1.0)
            return np.asarray(
                np.where(
                    positive,
                    np.exp(log_norm + (shape - 1.0) * np.log(safe) - rate * safe),
                    0.0,
                ),
                dtype=float,
            )

        return kernel

    def peak(self, theta: Any) -> PeakLocation:
        """Return the interior maximum, at ``(shape - 1) / rate``.

        Evaluated in logs. Written directly, ``rate ** shape`` overflows for a
        rate past about 300 at shape 4 while the kernel value itself is
        perfectly ordinary -- the intermediate blows up, not the answer.
        """
        values, _ = _batch(theta, 3)
        alpha, shape, rate = (float(v) for v in values[0])
        lag = (shape - 1.0) / rate
        if lag <= 0.0:
            # shape is bounded below by 1, so this is the boundary case only.
            return PeakLocation(lag=0.0, value=alpha * rate)
        log_value = (
            math.log(alpha)
            + shape * math.log(rate)
            + (shape - 1.0) * math.log(lag)
            - (shape - 1.0)
            - math.lgamma(shape)
        )
        return PeakLocation(lag=lag, value=math.exp(log_value))

    def mass(self, theta: Any) -> np.ndarray:
        """``alpha``: the kernel is a probability density times ``alpha``."""
        values, flat = _batch(theta, 3)
        return _unbatch(values[:, 0].copy(), flat)


# ---------------------------------------------------------------------------
# Spatial kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianSpatial:
    r"""Isotropic Gaussian :math:`(2\pi\sigma^2)^{-n/2} e^{-d^2 / 2\sigma^2}`.

    Normalised to unit mass on :math:`\mathbb{R}^n`, so on a *compact* domain its
    mass is at most one, with equality in the limit of a kernel narrow compared
    to the domain. That inequality is the whole reason for the normalisation:
    it makes the temporal kernel's mass a **sufficient** bound on the branching
    ratio of the separable spatio-temporal process, with no quadrature and no
    dependence on which surface the process lives on.

    Strictly positive everywhere, which is what the cached likelihood backend
    needs: the separability identity it rests on holds only where the pre-floor
    integrand is non-negative at every node.

    Parameters
    ----------
    ndim : int
        Dimension of the domain, which fixes the normalising constant.
    """

    ndim: int

    def __post_init__(self) -> None:
        """Refuse a dimension the normalising constant is not written for."""
        if self.ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {self.ndim}")

    @property
    def spec(self) -> ParameterSpec:
        """``(sigma,)``, positive."""
        return ParameterSpec((Parameter("sigma"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative distances."""
        values, _ = _batch(theta, 1)
        sigma = float(values[0, 0])
        norm = (2.0 * math.pi * sigma**2) ** (-0.5 * self.ndim)

        def kernel(d: Any) -> np.ndarray:
            distances = np.asarray(d, dtype=float)
            return np.asarray(norm * np.exp(-0.5 * (distances / sigma) ** 2), dtype=float)

        return kernel

    def mass(self, theta: Any) -> np.ndarray:
        """``1.0``: unit mass on the model space, and at most that on a domain."""
        values, flat = _batch(theta, 1)
        return _unbatch(np.ones(values.shape[0], dtype=float), flat)

    def min_scale(self, theta: Any) -> np.ndarray:
        """``sigma``, the length the kernel varies on."""
        values, flat = _batch(theta, 1)
        return _unbatch(values[:, 0].copy(), flat)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantBase:
    """A background intensity that does not vary over the domain.

    ``mu`` is the intensity **per unit measure**, so the background event rate
    is ``mu * domain.volume``. Reading it as a rate instead scales every fitted
    background by the area of the surface, which on a unit circle is a factor of
    6.28 and on a genus-2 surface a factor of 25.
    """

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the background as a callable on a domain point."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        return lambda x: mu  # noqa: ARG005 - constant in position, by definition

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate at every row of `points`, shape ``(m, ndim)`` -> ``(m,)``."""
        values, _ = _batch(theta, 1)
        nodes = np.asarray(points, dtype=float)
        return np.full(nodes.shape[0], float(values[0, 0]), dtype=float)


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearNonlinearity:
    r"""The identity plus a background: :math:`\varphi(x) = \mu + x`.

    Makes a :class:`~hawkes_package.monotone.MonotoneKernelHawkes` or
    :class:`~hawkes_package.bell_shape.BellShapeHawkes` a *linear* Hawkes
    process with background ``mu`` and a general kernel, which is what makes it
    comparable with :class:`~hawkes_package.exponential.ExponentialHawkes`.
    """

    lipschitz: float = 1.0

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        return lambda x: mu + np.asarray(x, dtype=float)


@dataclass(frozen=True)
class SoftPlusNonlinearity:
    r"""A saturating nonlinearity, :math:`\varphi(x) = \mu + c\,\log(1 + e^{x/c})`.

    Increasing and 1-Lipschitz for every ``c > 0``, so the same
    ``mass(kernel) < 1`` condition bounds it -- by Bremaud and Massoulie's
    criterion, which asks for the Lipschitz constant times the kernel mass, not
    the branching ratio of a linear process that does not exist here.

    Bounded below by ``mu``, so the intensity stays positive and the thinning
    loop's non-positive-bound guard is unreachable.

    Parameters
    ----------
    scale : float
        ``c``, the excitation at which the response starts to saturate. Not
        fitted: it trades off against the kernel amplitude almost exactly, and
        a pair of parameters that trade off exactly is a ridge the posterior
        never leaves.
    """

    scale: float = 1.0
    lipschitz: float = 1.0

    def __post_init__(self) -> None:
        """Refuse a non-positive saturation scale."""
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        scale = self.scale

        def phi(x: Any) -> np.ndarray:
            # logaddexp, not log1p(exp(.)): the excitation sum reaches a few
            # hundred on a burst, where exp overflows and log1p sees inf.
            return np.asarray(
                mu + scale * np.logaddexp(0.0, np.asarray(x, dtype=float) / scale),
                dtype=float,
            )

        return phi
