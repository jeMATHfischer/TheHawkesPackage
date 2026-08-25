#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial domain abstractions for spatio-temporal Hawkes processes.

Each domain defines how distances are measured, how coordinates are wrapped to
stay within the domain, and how to sample uniformly from the domain.
"""

from abc import ABC, abstractmethod
import numpy as np


class SpatialDomain(ABC):
    """Abstract base class for spatial domains."""

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

    Points are represented as angles in (-π·radius, π·radius].
    """

    def __init__(self, radius: float = 1.0):
        self.radius = radius
        self._period = 2 * np.pi * radius

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = abs(np.asarray(x).flat[0] - np.asarray(y).flat[0]) % self._period
        return min(diff, self._period - diff)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        half = self._period / 2
        return (np.asarray(x) + half) % self._period - half

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        half = self._period / 2
        return rng.uniform(-half, half, size=(1,))

    @property
    def volume(self) -> float:
        return self._period

    @property
    def bounds(self) -> np.ndarray:
        half = self._period / 2
        return np.array([[-half, half]])


class Torus2D(SpatialDomain):
    """Flat 2-D torus [0, L1) × [0, L2) with periodic boundaries in both dimensions.

    Points are represented in (-L1/2, L1/2] × (-L2/2, L2/2].
    """

    def __init__(self, L1: float = 2 * np.pi, L2: float = 2 * np.pi):
        self.L1 = L1
        self.L2 = L2

    def _wrap_1d(self, x: float, period: float) -> float:
        half = period / 2
        return (x + half) % period - half

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        x, y = np.asarray(x), np.asarray(y)
        dx = abs(float(x[0] - y[0])) % self.L1
        dy = abs(float(x[1] - y[1])) % self.L2
        dx = min(dx, self.L1 - dx)
        dy = min(dy, self.L2 - dy)
        return float(np.sqrt(dx**2 + dy**2))

    def wrap(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.array([
            self._wrap_1d(x[0], self.L1),
            self._wrap_1d(x[1], self.L2),
        ])

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([
            rng.uniform(-self.L1 / 2, self.L1 / 2),
            rng.uniform(-self.L2 / 2, self.L2 / 2),
        ])

    @property
    def volume(self) -> float:
        return self.L1 * self.L2

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[-self.L1 / 2, self.L1 / 2],
                         [-self.L2 / 2, self.L2 / 2]])
