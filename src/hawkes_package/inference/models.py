"""Maps from a parameter vector to a process object.

A :class:`ProcessModel` is the bridge between parameter space and the simulator:
given ``theta`` it hands back a fully built
:class:`~hawkes_package.base.HawkesProcess`, and it knows which ``theta`` are
worth handing to it at all.

Two rules hold everywhere here.

**Always build fresh.** :meth:`ProcessModel.__call__` never mutates a process it
was given, because there is no process it was given. A sampler that reused one
object would have to reset :attr:`~hawkes_package.base.HawkesProcess.events`
between particles, and the first time it forgot, the record and
``n_simulated`` would drift apart and every subsequent likelihood would be
conditioned on the wrong history -- with nothing raised.

**Declare the support rather than discovering it.** Every process class in this
package validates its arguments and raises :class:`ValueError` on a bad one --
``ExponentialHawkes`` refuses ``alpha >= beta``, ``locate_peak`` refuses a
negative kernel. Catching those exceptions and reading them as "zero posterior
mass" is how a broadcasting typo in a kernel becomes a plausible-looking
posterior. :meth:`ProcessModel.support` states the conditions up front, is
checked *before* anything is constructed, and a constructor raising anyway is
then a bug rather than a datum.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import HawkesProcess, SeedLike
from ..bell_shape import BellShapeHawkes
from ..exponential import ExponentialHawkes
from ..monotone import MonotoneKernelHawkes
from ..spatio_temporal.domains import Circle, SpatialDomain
from ..spatio_temporal.kernels import make_periodic
from ..spatio_temporal.process import SpatioTemporalHawkesProcess
from .families import (
    BaseFamily,
    ConstantBase,
    ExponentialKernel,
    GammaKernel,
    GaussianSpatial,
    KernelFamily,
    LinearNonlinearity,
    NonlinearityFamily,
    SpatialKernelFamily,
)
from .parameters import ParameterSpec

__all__ = [
    "ProcessModel",
    "SpatialComponents",
    "TemporalComponents",
    "bell_shape_model",
    "exponential_model",
    "monotone_model",
    "spatio_temporal_model",
]


@dataclass(frozen=True)
class TemporalComponents:
    """The families a temporal model is assembled from."""

    kernel: KernelFamily
    nonlinearity: NonlinearityFamily | None


@dataclass(frozen=True)
class SpatialComponents:
    """The families and geometry a spatio-temporal model is assembled from.

    Everything the cached likelihood backend needs in order to reproduce, from
    parameters alone, exactly what the simulator would have computed through its
    hooks -- including whether the spatial kernel is summed over image points,
    which changes the kernel and not merely its accuracy.
    """

    domain: SpatialDomain
    base: BaseFamily
    temporal: KernelFamily
    spatial: SpatialKernelFamily
    n_quad: int | None
    n_images: int | None
    """Images per direction in the periodised sum, or ``None`` when the kernel
    is used unperiodised."""

    @property
    def nodes_per_axis(self) -> int:
        """The node count the process will actually build its rule at."""
        return self.domain.nodes_per_axis if self.n_quad is None else int(self.n_quad)


def _all_true(theta: np.ndarray) -> np.ndarray:
    """Admit every parameter: the support of a model with no extra condition."""
    return np.ones(np.atleast_2d(theta).shape[0], dtype=bool)


@dataclass(frozen=True)
class ProcessModel:
    """A parameterised family of Hawkes processes.

    Attributes
    ----------
    spec : ParameterSpec
        The coordinates, in the order every array in this subpackage uses.
    family : str
        Which process class this builds. The likelihood implementations key on
        it to decide whether their closed form applies.
    ndim : int
        Spatial dimension, ``0`` for a purely temporal model.
    components : TemporalComponents or SpatialComponents
        The families the model was assembled from.

    .. versionadded:: 0.5.0
    """

    spec: ParameterSpec
    family: str
    ndim: int
    components: TemporalComponents | SpatialComponents
    builder: Callable[[np.ndarray, SeedLike], HawkesProcess]
    branching: Callable[[np.ndarray], np.ndarray]
    extra_support: Callable[[np.ndarray], np.ndarray] = field(default=_all_true)

    def __call__(self, theta: Any, *, rng: SeedLike = None) -> HawkesProcess:
        """Build the process at one parameter vector.

        Parameters
        ----------
        theta : array_like of shape (n_dim,)
            A single parameter vector. Batches are not accepted: a process is a
            stateful object, and returning a list of them from a call that looks
            vectorized invites reusing one.
        rng : None, int or numpy.random.Generator
            Passed straight to the process.

        Raises
        ------
        ValueError
            If `theta` lies outside :meth:`support`.
        """
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != len(self.spec):
            raise ValueError(
                f"expected {len(self.spec)} parameter(s) {list(self.spec.names)}, "
                f"got shape {np.shape(theta)}"
            )
        if not bool(self.support(values)):
            raise ValueError(
                f"theta={values.tolist()} is outside the support of this "
                f"{self.family} model: parameters {list(self.spec.names)}, branching "
                f"ratio {float(np.atleast_1d(self.branching_ratio(values))[0]):.4g}"
            )
        return self.builder(values, rng)

    def support(self, theta: Any) -> np.ndarray:
        """Whether each row of `theta` is one the process can be built at.

        The bounds, the branching ratio and whatever else the concrete family
        needs, in one predicate, evaluated without constructing anything.

        Returns
        -------
        numpy.ndarray
            Boolean, shape ``()`` or ``(n_particles,)``.
        """
        array = np.asarray(theta, dtype=float)
        flat = array.ndim == 1
        batch = np.atleast_2d(array)
        ratio = np.asarray(self.branching(batch), dtype=float)
        ok = (
            np.asarray(self.spec.contains(batch), dtype=bool)
            & np.isfinite(ratio)
            & (ratio < 1.0)
            & np.asarray(self.extra_support(batch), dtype=bool)
        )
        return np.asarray(ok[0] if flat else ok, dtype=bool)

    def branching_ratio(self, theta: Any) -> np.ndarray:
        """Return the expected direct offspring per event, batched over `theta`.

        Below one the process is stationary and simulating it terminates; at or
        above one it is not and does not. For a nonlinear model this is the
        Bremaud-Massoulie quantity, the Lipschitz constant of the nonlinearity
        times the kernel's mass, which is a sufficient condition rather than an
        exact rate.
        """
        array = np.asarray(theta, dtype=float)
        flat = array.ndim == 1
        ratio = np.asarray(self.branching(np.atleast_2d(array)), dtype=float)
        return np.asarray(ratio[0] if flat else ratio, dtype=float)

    def split(self, theta: Any, name: str) -> np.ndarray:
        """Return the coordinate called `name` from each row of `theta`."""
        array = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.asarray(array[:, self.spec.index(name)], dtype=float)

    def __repr__(self) -> str:
        """Name the family and its coordinates, and nothing else.

        The generated dataclass repr prints three closures by their `<function
        build at 0x...>` addresses plus every `Parameter` in full -- around 400
        characters, of which the varying part is an address that says nothing.
        Written in the class body, so `dataclass` leaves it alone.
        """
        names = ", ".join(self.spec.names)
        return f"{type(self).__name__}(family={self.family!r}, ndim={self.ndim}, ({names}))"


# ---------------------------------------------------------------------------
# Temporal factories
# ---------------------------------------------------------------------------


def exponential_model() -> ProcessModel:
    """Build the linear Hawkes model with an exponential kernel: ``(mu, alpha, beta)``.

    The one model with a closed-form likelihood -- see
    :class:`~hawkes_package.inference.likelihood.ExponentialLogLikelihood` --
    and so the one to reach for unless the kernel shape is itself in question.

    Its support excludes ``alpha >= beta``, which is exactly what
    :class:`~hawkes_package.exponential.ExponentialHawkes` refuses to be built
    at.

    Examples
    --------
    >>> model = exponential_model()
    >>> model.spec.names
    ('mu', 'alpha', 'beta')
    >>> model.support(np.array([[1.0, 0.5, 2.0], [1.0, 3.0, 2.0]])).tolist()
    [True, False]

    .. versionadded:: 0.5.0
    """
    kernel = ExponentialKernel()
    spec = LinearNonlinearity().spec.concat(kernel.spec)

    def build(theta: np.ndarray, rng: SeedLike) -> HawkesProcess:
        return ExponentialHawkes(theta, rng=rng)

    return ProcessModel(
        spec=spec,
        family="exponential",
        ndim=0,
        components=TemporalComponents(kernel=kernel, nonlinearity=LinearNonlinearity()),
        builder=build,
        branching=lambda theta: kernel.mass(theta[:, 1:3]),
    )


def monotone_model(
    kernel: KernelFamily | None = None,
    nonlinearity: NonlinearityFamily | None = None,
) -> ProcessModel:
    """Build a :class:`~hawkes_package.monotone.MonotoneKernelHawkes` over a kernel family.

    Parameters
    ----------
    kernel : KernelFamily
        Must be monotone decreasing -- the class's thinning bound is only valid
        for one that is, and passing a bell-shaped family here would silently
        produce a Poisson process rather than raising. Defaults to
        :class:`~hawkes_package.inference.families.ExponentialKernel`.
    nonlinearity : NonlinearityFamily
        Defaults to
        :class:`~hawkes_package.inference.families.LinearNonlinearity`, which
        makes this the linear Hawkes process with a general monotone kernel.

    Raises
    ------
    ValueError
        If `kernel` is not marked monotone.

    .. versionadded:: 0.5.0
    """
    excitation: KernelFamily = ExponentialKernel() if kernel is None else kernel
    response: NonlinearityFamily = LinearNonlinearity() if nonlinearity is None else nonlinearity
    if not excitation.monotone:
        raise ValueError(
            f"{type(excitation).__name__} is not monotone decreasing, and "
            "MonotoneKernelHawkes bounds the intensity by its value at the current "
            "time -- which a rising kernel exceeds. Use bell_shape_model instead."
        )

    n_phi = len(response.spec)
    spec = response.spec.concat(excitation.spec)

    def build(theta: np.ndarray, rng: SeedLike) -> HawkesProcess:
        return MonotoneKernelHawkes(
            excitation.build(theta[n_phi:]), response.build(theta[:n_phi]), rng=rng
        )

    return ProcessModel(
        spec=spec,
        family="monotone",
        ndim=0,
        components=TemporalComponents(kernel=excitation, nonlinearity=response),
        builder=build,
        branching=lambda theta: response.lipschitz * excitation.mass(theta[:, n_phi:]),
    )


def bell_shape_model(
    kernel: KernelFamily | None = None,
    nonlinearity: NonlinearityFamily | None = None,
) -> ProcessModel:
    """Build a :class:`~hawkes_package.bell_shape.BellShapeHawkes` over a kernel family.

    The peak is taken from the family's closed form and passed as ``peak_lag``
    and ``peak_value``, so the 513-point scan in
    :func:`~hawkes_package._numerics.locate_peak` never runs. That is not only a
    saving: the scan is the search a rejuvenation move would repeat for every
    particle, and its cost is what would push a fit of a bell-shaped kernel from
    minutes into hours.

    Parameters
    ----------
    kernel : KernelFamily
        Defaults to :class:`~hawkes_package.inference.families.GammaKernel`.
    nonlinearity : NonlinearityFamily
        Defaults to
        :class:`~hawkes_package.inference.families.LinearNonlinearity`.

    Raises
    ------
    ValueError
        If `kernel` does not expose an analytic :meth:`~KernelFamily.peak`.

    .. versionadded:: 0.5.0
    """
    excitation: KernelFamily = GammaKernel() if kernel is None else kernel
    response: NonlinearityFamily = LinearNonlinearity() if nonlinearity is None else nonlinearity
    if not callable(getattr(excitation, "peak", None)):
        raise ValueError(
            f"{type(excitation).__name__} exposes no analytic peak. BellShapeHawkes would "
            "fall back to a numerical search, run once per particle per move; a family "
            "used for inference must state where its kernel peaks."
        )

    n_phi = len(response.spec)
    spec = response.spec.concat(excitation.spec)

    def build(theta: np.ndarray, rng: SeedLike) -> HawkesProcess:
        located = excitation.peak(theta[n_phi:])
        return BellShapeHawkes(
            excitation.build(theta[n_phi:]),
            response.build(theta[:n_phi]),
            rng=rng,
            peak_lag=located.lag,
            peak_value=located.value,
        )

    return ProcessModel(
        spec=spec,
        family="bell_shape",
        ndim=0,
        components=TemporalComponents(kernel=excitation, nonlinearity=response),
        builder=build,
        branching=lambda theta: response.lipschitz * excitation.mass(theta[:, n_phi:]),
    )


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


def spatio_temporal_model(
    domain: SpatialDomain | None = None,
    *,
    base: BaseFamily | None = None,
    temporal: KernelFamily | None = None,
    spatial: SpatialKernelFamily | None = None,
    n_quad: int | None = None,
    n_images: int | None = None,
) -> ProcessModel:
    """Build a :class:`~hawkes_package.SpatioTemporalHawkesProcess` over families.

    Parameters
    ----------
    domain : SpatialDomain
        Defaults to the unit :class:`~hawkes_package.Circle`.
    base : BaseFamily
        Background per unit measure. Defaults to
        :class:`~hawkes_package.inference.families.ConstantBase`.
    temporal, spatial : KernelFamily, SpatialKernelFamily
        Default to :class:`~hawkes_package.inference.families.ExponentialKernel`
        and :class:`~hawkes_package.inference.families.GaussianSpatial` at the
        domain's dimension.
    n_quad : int, optional
        Nodes per axis for the process's spatial rule. Defaults to the domain's
        own recommendation. **Per axis**, and rounded down to a whole number of
        four-point panels -- never derive it from a total node budget.
    n_images : int, optional
        Periodise the spatial kernel over this many images per direction, as
        :func:`~hawkes_package.make_periodic` does. ``None`` leaves the kernel
        unperiodised, which is a different model rather than a coarser one
        wherever the kernel has not decayed by the boundary.

    Notes
    -----
    The support carries a lower bound on the spatial scale of
    ``2 * width / n_quad``, half a panel of the composite Gauss-Legendre rule.
    Below that a kernel can fall entirely between two nodes, so the quadrature
    reports the background integral and the excitation becomes invisible to the
    thinning while the location sampler still sees it -- the process silently
    degenerating towards Poisson in time. The bound is where the kernel stops
    being *visible*, not where the rule stops being *accurate*; accuracy at a
    given ``theta`` is the business of
    :func:`~hawkes_package.spatio_temporal._integration.check_resolution`, which
    the process runs at construction.

    .. versionadded:: 0.5.0
    """
    surface: SpatialDomain = Circle() if domain is None else domain
    ndim = int(np.asarray(surface.bounds, dtype=float).shape[0])
    background: BaseFamily = ConstantBase() if base is None else base
    excitation: KernelFamily = ExponentialKernel() if temporal is None else temporal
    shape: SpatialKernelFamily = GaussianSpatial(ndim) if spatial is None else spatial

    components = SpatialComponents(
        domain=surface,
        base=background,
        temporal=excitation,
        spatial=shape,
        n_quad=n_quad,
        n_images=n_images,
    )

    n_base = len(background.spec)
    n_temporal = len(excitation.spec)
    spec = background.spec.concat(excitation.spec).concat(shape.spec)
    temporal_slice = slice(n_base, n_base + n_temporal)
    spatial_slice = slice(n_base + n_temporal, len(spec))

    bounds = np.asarray(surface.bounds, dtype=float)
    widest = float(np.max(bounds[:, 1] - bounds[:, 0]))
    min_scale = 2.0 * widest / components.nodes_per_axis

    def build(theta: np.ndarray, rng: SeedLike) -> HawkesProcess:
        kernel_fn = shape.build(theta[spatial_slice])
        spatial_fn: Any = kernel_fn
        if n_images is not None:
            spatial_fn = make_periodic(kernel_fn, surface, n_images)
        temporal_fn = excitation.build(theta[temporal_slice])
        # The peak is passed through explicitly rather than analytically located
        # by the constructor: `locate_peak` is a 513-point scan, and this runs
        # once per particle per move.
        located = None if excitation.monotone else excitation.peak(theta[temporal_slice])
        return SpatioTemporalHawkesProcess(
            base=background.build(theta[:n_base]),
            spatial=spatial_fn,
            temporal=temporal_fn,
            domain=surface,
            monotone_temporal_kernel=excitation.monotone,
            rng=rng,
            n_quad=n_quad,
            peak_lag=None if located is None else located.lag,
            peak_value=None if located is None else located.value,
        )

    def branching(theta: np.ndarray) -> np.ndarray:
        # The spatial kernel is normalised to unit mass on the model space, so
        # its mass over a compact domain is at most one and this dominates the
        # true branching ratio. Bounding it, rather than integrating the kernel
        # over the surface per particle, is what keeps `support` free of
        # quadrature -- and `support` is evaluated on every particle of every
        # proposal.
        return np.asarray(
            excitation.mass(theta[:, temporal_slice]) * shape.mass(theta[:, spatial_slice]),
            dtype=float,
        )

    def resolvable(theta: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.asarray(shape.min_scale(theta[:, spatial_slice]), dtype=float) >= min_scale,
            dtype=bool,
        )

    return ProcessModel(
        spec=spec,
        family="spatio_temporal",
        ndim=ndim,
        components=components,
        builder=build,
        branching=branching,
        extra_support=resolvable,
    )
