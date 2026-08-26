#!/usr/bin/env python3
"""
Spatial domain abstractions for spatio-temporal Hawkes processes.

Each domain defines how distances are measured, how coordinates are wrapped to
stay within the domain, and how to sample uniformly from the domain.

:class:`Circle` and :class:`Torus2D` write out one quotient each by hand.
:class:`FundamentalDomain` is the general construction both are instances of --
a region of a model space glued to itself along its boundary -- and is the route
to surfaces neither of them can present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .._numerics import as_point

__all__ = ["Circle", "FundamentalDomain", "SpatialDomain", "Torus2D"]


class SpatialDomain(ABC):
    """Abstract base class for spatial domains.

    The spatial integral is a quadrature rule over :attr:`bounds`, restricted to
    the nodes :meth:`contains` admits and weighted by :meth:`volume_element`.
    Subclasses must therefore satisfy

    ``volume == integral of volume_element over {x in bounds : contains(x)}``.

    The two hooks default to "the domain fills its bounding box, with the flat
    chart measure", which reduces that requirement to the older
    ``volume == prod(bounds widths)`` and leaves an existing subclass — which
    overrides neither — behaving exactly as before.

    Attributes
    ----------
    periodic : bool
        Whether :meth:`wrap` folds coordinates through a translation symmetry,
        as opposed to clipping them. Only a genuinely periodic domain may have
        MCMC proposals folded rather than rejected: folding is a symmetric move
        on the quotient and therefore reversible, whereas folding with a
        clipping map would pile every draw onto the boundary. Defaults to
        ``False``, so a third-party domain is treated conservatively.
    """

    periodic: bool = False

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic distance between points x and y in the domain."""

    @abstractmethod
    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Map x to its canonical representative inside the domain."""

    @abstractmethod
    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw a single point uniformly at random from the domain."""

    @property
    @abstractmethod
    def volume(self) -> float:
        """Measure (length/area/volume) of the domain."""

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """Bounding box as array of shape (ndim, 2) — used for MCMC initialisation."""

    # ------------------------------------------------------------------
    # Optional hooks. Concrete, not abstract: a domain that fills its
    # bounding box with the flat measure and has no deck group -- which is
    # every domain that existed before 0.3.0 -- inherits all three.
    # ------------------------------------------------------------------

    def contains(self, x: np.ndarray) -> bool:  # noqa: ARG002 - the interface, not this default
        """Whether `x` lies in the domain.

        Defaults to ``True``: the domain *is* its bounding box, so every node
        of the quadrature rule is admitted.

        .. versionadded:: 0.3.0
        """
        return True

    def volume_element(self, x: np.ndarray) -> float:  # noqa: ARG002 - as above
        """Return the measure density ``sqrt(det g)`` at `x`.

        Relates the chart measure the quadrature nodes live in to the true
        measure of the domain. Defaults to ``1.0``, the flat chart.

        Must be **strictly positive**: the Ogata thinning bound is valid because
        a pointwise-dominating integrand meets strictly positive weights on a
        shared node set, and a zero or negative weight breaks that.

        .. versionadded:: 0.3.0
        """
        return 1.0

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray] | None:  # noqa: ARG002
        """Return the images of `y` under the deck group, or ``None`` if there is no group.

        :func:`~hawkes_package.spatio_temporal.kernels.make_periodic` sums a
        kernel over this set. ``None`` — the default — means the domain declares
        no translational structure, and periodisation falls back to a single
        evaluation at :meth:`distance`.

        .. versionadded:: 0.3.0
        """
        return None

    @property
    def interior_point(self) -> np.ndarray:
        """A point known to lie inside the domain.

        The centre of :attr:`bounds` by default, which is correct whenever the
        domain fills its box. A domain that is a proper subset must override
        this: the box centre may well lie outside it, and this point is used to
        probe whether the spatial kernel is resolved by the quadrature rule.

        .. versionadded:: 0.3.0
        """
        return np.asarray(self.bounds, dtype=float).mean(axis=1)


class Circle(SpatialDomain):
    """1-D circular domain [0, 2π·radius) with periodic boundary.

    Points are represented as arc lengths in [-π·radius, π·radius);
    :attr:`bounds` reports the closed bounding box used for quadrature.
    """

    periodic = True

    def __init__(self, radius: float = 1.0):
        self.radius = radius
        self._period = 2 * np.pi * radius

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Arc length between x and y, taking the shorter way round."""
        # as_point rather than .flat[0]: the latter silently accepted a
        # two-vector and measured only its first component.
        diff = abs(as_point(x, 1)[0] - as_point(y, 1)[0]) % self._period
        return float(min(diff, self._period - diff))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Fold x into the canonical half-open interval."""
        half = self._period / 2
        return (as_point(x, 1) + half) % self._period - half

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the circle."""
        half = self._period / 2
        return rng.uniform(-half, half, size=(1,))

    @property
    def volume(self) -> float:
        """Circumference of the circle."""
        return self._period

    @property
    def bounds(self) -> np.ndarray:
        """Bounding interval, shape (1, 2)."""
        half = self._period / 2
        return np.array([[-half, half]])

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Images of `y` under translation by whole periods."""
        y = as_point(y, 1)
        return [y + n * self._period for n in range(-n_images, n_images + 1)]


class Torus2D(SpatialDomain):
    """Flat 2-D torus [0, L1) x [0, L2) with periodic boundaries in both dimensions.

    Points are represented in [-L1/2, L1/2) x [-L2/2, L2/2).
    """

    periodic = True

    def __init__(self, L1: float = 2 * np.pi, L2: float = 2 * np.pi):
        self.L1 = L1
        self.L2 = L2

    def _wrap_1d(self, x: float, period: float) -> float:
        """Fold a single coordinate into (-period/2, period/2]."""
        half = period / 2
        return (x + half) % period - half

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean distance on the flat torus, wrapping both axes."""
        x, y = as_point(x, 2), as_point(y, 2)
        dx = abs(float(x[0] - y[0])) % self.L1
        dy = abs(float(x[1] - y[1])) % self.L2
        dx = min(dx, self.L1 - dx)
        dy = min(dy, self.L2 - dy)
        return float(np.sqrt(dx**2 + dy**2))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Fold both coordinates into the canonical rectangle."""
        x = as_point(x, 2)
        return np.array(
            [
                self._wrap_1d(x[0], self.L1),
                self._wrap_1d(x[1], self.L2),
            ]
        )

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the torus."""
        return np.array(
            [
                rng.uniform(-self.L1 / 2, self.L1 / 2),
                rng.uniform(-self.L2 / 2, self.L2 / 2),
            ]
        )

    @property
    def volume(self) -> float:
        """Surface area of the torus."""
        return self.L1 * self.L2

    @property
    def bounds(self) -> np.ndarray:
        """Bounding rectangle, shape (2, 2)."""
        return np.array([[-self.L1 / 2, self.L1 / 2], [-self.L2 / 2, self.L2 / 2]])

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Lattice images of `y`, out to `n_images` periods on each axis."""
        y = as_point(y, 2)
        periods = np.array([self.L1, self.L2])
        return [
            y + np.array([n1, n2]) * periods
            for n1 in range(-n_images, n_images + 1)
            for n2 in range(-n_images, n_images + 1)
        ]


# ---------------------------------------------------------------------------
# Fundamental domains
# ---------------------------------------------------------------------------


def _affine(matrix: Any) -> np.ndarray:
    """Coerce a side pairing to a 3x3 affine matrix and check it is admissible.

    Rejects anything that is not an orientation-preserving isometry. An
    orientation-*reversing* pairing quotients to a non-orientable surface, on
    which the oriented constructions this package is built towards do not exist;
    failing here beats producing a domain whose geometry silently disagrees with
    the one the caller meant.
    """
    a = np.asarray(matrix, dtype=float)
    if a.shape == (2, 3):
        a = np.vstack([a, [0.0, 0.0, 1.0]])
    if a.shape != (3, 3):
        raise ValueError(f"a side pairing must be a 3x3 (or 2x3) affine matrix, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("a side pairing must be finite")
    if not np.allclose(a[2], [0.0, 0.0, 1.0]):
        raise ValueError(f"the last row of an affine matrix must be [0, 0, 1], got {a[2].tolist()}")

    linear = a[:2, :2]
    if not np.allclose(linear @ linear.T, np.eye(2), atol=1e-9):
        raise ValueError("a side pairing must be an isometry; its linear part is not orthogonal")
    determinant = float(np.linalg.det(linear))
    if not np.isclose(determinant, 1.0, atol=1e-9):
        raise ValueError(
            f"a side pairing must be orientation-preserving (det +1), got det {determinant:.3f}. "
            "An orientation-reversing pairing quotients to a non-orientable surface."
        )
    return a


def _group_window(generators: list[np.ndarray], depth: int) -> np.ndarray:
    """Every word of length at most `depth` in `generators` and their inverses.

    Returned as a ``(m, 3, 3)`` stack with the identity first, so a caller can
    treat index 0 as "no move". Deduplicated, which matters: for an abelian
    group the word count grows exponentially while the element count does not.
    """
    identity = np.eye(3)
    alphabet = [*generators, *(np.linalg.inv(g) for g in generators)]

    def key(m: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(m.ravel(), 9) + 0.0)  # +0.0 folds -0.0 onto 0.0

    elements = [identity]
    seen = {key(identity)}
    frontier = [identity]
    for _ in range(max(0, depth)):
        next_frontier = []
        for word in frontier:
            for letter in alphabet:
                candidate = letter @ word
                candidate_key = key(candidate)
                if candidate_key not in seen:
                    seen.add(candidate_key)
                    elements.append(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return np.array(elements)


class FundamentalDomain(SpatialDomain):
    """A convex Euclidean polygon with side pairings, presenting a flat surface.

    The domain is the polygon `vertices`; `pairings` are the side-pairing
    isometries that generate the deck group, and points differing by a group
    element are the same point of the quotient surface. :class:`Torus2D` is the
    rectangle case, available here as :meth:`rectangle`; :meth:`hexagon` gives
    the hexagonal torus, which no rectangular domain expresses.

    Parameters
    ----------
    vertices : array_like, shape (n, 2)
        Polygon corners in order, either winding. Must be convex.
    pairings : sequence of array_like
        Side-pairing isometries as ``3x3`` (or ``2x3``) affine matrices. Each
        must be orientation-preserving, so that the quotient is orientable.
    n_images : int
        Word length the deck group is truncated at, for :meth:`distance` and
        :meth:`orbit`. Both points are reduced into the polygon first, so the
        minimising element is short and the default is ample.

    Notes
    -----
    :attr:`periodic` is ``False``, so MCMC proposals leaving the polygon are
    *rejected* rather than folded back through :meth:`wrap`. Folding is
    reversible only when the group acts by translations, which leave a Gaussian
    proposal invariant; it happens to hold for both domains constructed here,
    but not for a general pairing, and this package has a history of bounds and
    samplers that were wrong silently rather than loudly.

    .. versionadded:: 0.3.0

    Examples
    --------
    >>> hexagon = FundamentalDomain.hexagon(1.0)
    >>> round(hexagon.volume, 6)
    2.598076
    >>> bool(hexagon.contains(hexagon.wrap([5.0, -4.0])))
    True
    """

    periodic = False

    def __init__(self, vertices: Any, pairings: Any, n_images: int = 3) -> None:
        corners = np.asarray(vertices, dtype=float)
        if corners.ndim != 2 or corners.shape[1] != 2 or corners.shape[0] < 3:
            raise ValueError(f"vertices must have shape (n >= 3, 2), got {corners.shape}")
        if not np.isfinite(corners).all():
            raise ValueError("vertices must be finite")

        # Orient counter-clockwise so the outward normal below is unambiguous.
        signed_area = 0.5 * float(
            np.sum(
                corners[:, 0] * np.roll(corners[:, 1], -1)
                - np.roll(corners[:, 0], -1) * corners[:, 1]
            )
        )
        if signed_area == 0.0:
            raise ValueError("vertices are degenerate: the polygon has zero area")
        if signed_area < 0:
            corners = corners[::-1]
            signed_area = -signed_area

        self.vertices = corners
        self._area = signed_area
        self._centre = corners.mean(axis=0)

        edges = np.roll(corners, -1, axis=0) - corners
        # Scalar 2-D cross product, spelled out: NumPy 2.0 removed `np.cross`
        # for 2-vectors.
        turn = np.roll(edges, -1, axis=0)
        if np.any(edges[:, 0] * turn[:, 1] - edges[:, 1] * turn[:, 0] < -1e-12):
            raise ValueError(
                "vertices must describe a convex polygon; `contains` is a half-plane test "
                "against every edge, which only characterises a convex region."
            )
        # Outward normal of a counter-clockwise edge (dx, dy) is (dy, -dx).
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        self._normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self._offsets = np.einsum("ij,ij->i", self._normals, corners)
        # Half-open on paired sides, so a boundary point has exactly one
        # representative: close the edge whose outward normal is
        # lexicographically negative, open the other. On a centrally symmetric
        # polygon -- both of the ones built here -- opposite edges carry
        # opposite normals, so exactly one of each pair is closed. This is the
        # generalisation of Circle's [-pi, pi) convention.
        self._closed = (self._normals[:, 0] < 0) | (
            (self._normals[:, 0] == 0) & (self._normals[:, 1] < 0)
        )
        self._scale = float(np.max(np.linalg.norm(corners - self._centre, axis=1)))
        self._eps = 1e-12 * self._scale  # the width of "on the boundary"
        self._tol = 1e-9 * self._scale  # when the reduction has stopped making progress

        pairing_list = [_affine(p) for p in pairings]
        if not pairing_list:
            raise ValueError("at least one side pairing is required")
        self.pairings = pairing_list
        self.n_images = int(n_images)
        self._window_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # -- construction helpers ------------------------------------------

    @classmethod
    def rectangle(cls, L1: float = 2 * np.pi, L2: float = 2 * np.pi) -> FundamentalDomain:
        """Build the rectangle ``[-L1/2, L1/2] x [-L2/2, L2/2]``, opposite sides paired.

        Presents the same flat torus as :class:`Torus2D`, and exists chiefly so
        that the general machinery can be checked against that hand-written
        implementation.
        """
        half_1, half_2 = L1 / 2, L2 / 2
        corners = [
            [-half_1, -half_2],
            [half_1, -half_2],
            [half_1, half_2],
            [-half_1, half_2],
        ]
        return cls(corners, [_translation(L1, 0.0), _translation(0.0, L2)])

    @classmethod
    def hexagon(cls, side: float = 1.0) -> FundamentalDomain:
        """Build the regular hexagon of circumradius `side`, opposite sides paired.

        This is the Dirichlet domain of the triangular lattice of spacing
        ``side * sqrt(3)``, and presents the hexagonal torus.
        """
        if not side > 0:
            raise ValueError(f"side must be positive, got {side}")
        angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
        corners = side * np.column_stack([np.cos(angles), np.sin(angles)])
        spacing = side * np.sqrt(3.0)
        return cls(
            corners,
            [
                _translation(spacing, 0.0),
                _translation(spacing / 2, spacing * np.sqrt(3.0) / 2),
            ],
        )

    # -- the deck group ------------------------------------------------

    def _window(self, depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Linear parts and translations of the truncated group, identity first."""
        cached = self._window_cache.get(depth)
        if cached is None:
            elements = _group_window(self.pairings, depth)
            cached = (elements[:, :2, :2], elements[:, :2, 2])
            self._window_cache[depth] = cached
        return cached

    def _images(self, y: np.ndarray, depth: int) -> np.ndarray:
        """Every image of `y` under the truncated group, shape ``(m, 2)``."""
        linear, translation = self._window(depth)
        images: np.ndarray = np.einsum("kij,j->ki", linear, y) + translation
        return images

    def orbit(self, y: np.ndarray, n_images: int = 3) -> list[np.ndarray]:
        """Images of `y` under the deck group, out to word length `n_images`."""
        return list(self._images(as_point(y, 2), n_images))

    # -- the SpatialDomain contract ------------------------------------

    def _inside(self, x: np.ndarray, slack: float = 0.0) -> np.bool_:
        """Strictly inside every edge, or on a *closed* one.

        The comparison runs against a band rather than exact zero, and it has to.
        An edge midpoint and its partner under the side pairing both evaluate to
        a signed distance of *about* zero on their respective edges, with a sign
        set by rounding; testing ``signed == 0`` therefore admits both -- and the
        quotient double-counts the edge.
        """
        signed = self._normals @ x - self._offsets
        edge = self._eps + slack
        return np.all((signed < -edge) | ((signed <= edge) & self._closed))

    def contains(self, x: np.ndarray) -> bool:
        """Whether `x` lies in the polygon, half-open on paired sides."""
        return bool(self._inside(as_point(x, 2), 0.0))

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance on the quotient: the shortest hop to any image of `y`.

        Both points are reduced into the polygon before the search, for the same
        reason :func:`~hawkes_package.spatio_temporal.kernels.make_periodic`
        reduces to the canonical offset first: measured from an unreduced lift,
        the nearest image can lie outside the truncated window.
        """
        reduced_x = self.wrap(x)
        images = self._images(self.wrap(y), self.n_images)
        return float(np.min(np.linalg.norm(images - reduced_x, axis=1)))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Map `x` to its representative inside the polygon."""
        point = self._reduce(as_point(x, 2))
        if self._inside(point, 0.0):
            return point

        # `_reduce` lands in the Dirichlet cell about the centre, which is the
        # polygon itself for both domains built here but need not be for a
        # caller's own. Sweeping the window covers that, and also re-homes a
        # point that landed on an open edge.
        images = self._images(point, self.n_images)
        order = np.argsort(np.linalg.norm(images - self._centre, axis=1))
        for tol in (0.0, self._tol):
            for index in order:
                candidate: np.ndarray = images[index]
                if self._inside(candidate, tol):
                    return candidate
        raise ValueError(
            f"no image of {np.asarray(x).tolist()} under the deck group lies in the polygon; "
            "the side pairings do not tile the plane by it."
        )

    def _reduce(self, x: np.ndarray, max_steps: int = 1000) -> np.ndarray:
        """Walk `x` towards the centre one generator at a time.

        Terminates because the distance to the centre strictly decreases at
        every step; the cap only guards a caller's non-discrete "group".
        """
        current = float(np.linalg.norm(x - self._centre))
        for _ in range(max_steps):
            images = self._images(x, 1)
            distances = np.linalg.norm(images - self._centre, axis=1)
            best = int(np.argmin(distances))
            if best == 0 or distances[best] >= current - self._tol:
                return x
            x, current = np.asarray(images[best]), float(distances[best])
        raise ValueError(
            "reducing a point into the polygon did not converge; the side pairings "
            "do not generate a discrete group."
        )

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one point uniformly on the polygon, by rejection in its box."""
        box = self.bounds
        for _ in range(10_000):
            candidate: np.ndarray = rng.uniform(box[:, 0], box[:, 1])
            if self._inside(candidate, 0.0):
                return candidate
        raise RuntimeError(  # pragma: no cover - needs a pathological polygon
            "10000 uniform draws over the bounding box all fell outside the polygon"
        )

    @property
    def volume(self) -> float:
        """Area of the polygon."""
        return self._area

    @property
    def bounds(self) -> np.ndarray:
        """Bounding rectangle of the polygon, shape (2, 2)."""
        return np.column_stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def interior_point(self) -> np.ndarray:
        """The mean of the vertices, which a convex polygon always contains.

        Also the point :meth:`wrap` reduces towards, and for a domain that is
        the Dirichlet cell of its lattice -- both of the ones built here -- it is
        the lattice site that cell belongs to.
        """
        return self._centre


def _translation(dx: float, dy: float) -> np.ndarray:
    """Return the affine matrix translating by ``(dx, dy)``."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
