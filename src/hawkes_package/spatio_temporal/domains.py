#!/usr/bin/env python3
"""
Spatial domain abstractions for spatio-temporal Hawkes processes.

Each domain defines how distances are measured, how coordinates are wrapped to
stay within the domain, and how to sample uniformly from the domain.
"""

from abc import ABC, abstractmethod

import numpy as np

from .._numerics import as_point


class SpatialDomain(ABC):
    """Abstract base class for spatial domains.

    Subclasses must satisfy ``volume == prod(bounds widths)``: the spatial
    integral is a quadrature rule over :attr:`bounds`, so a domain that is a
    proper subset of its bounding box is not supported.

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
