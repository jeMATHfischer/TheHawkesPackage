"""Tests for the shared numerical helpers."""

import numpy as np
import pytest

from hawkes_package._numerics import as_float, as_point, locate_peak

# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [1.5, np.float64(1.5), np.array(1.5), np.array([1.5]), np.array([[1.5]])]
)
def test_as_float_accepts_every_single_element_shape(value):
    """`float()` rejects a shape-(1,) array under NumPy 2; user kernels return them."""
    assert as_float(value) == 1.5


def test_as_float_rejects_multiple_elements():
    with pytest.raises(ValueError, match="single-element"):
        as_float(np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    ("value", "ndim"),
    [(0.3, 1), ([0.3], 1), (np.array([0.3]), 1), (np.array([[0.3]]), 1), (np.array(0.3), 1)],
)
def test_as_point_normalises_one_dimensional_input(value, ndim):
    point = as_point(value, ndim)
    assert point.shape == (ndim,)
    assert point[0] == pytest.approx(0.3)


def test_as_point_accepts_a_column_vector():
    np.testing.assert_allclose(as_point(np.array([[1.0], [2.0]]), 2), [1.0, 2.0])


def test_as_point_rejects_a_size_mismatch():
    with pytest.raises(ValueError, match="1 coordinate"):
        as_point([1.0, 2.0], 1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_as_point_rejects_non_finite_coordinates(bad):
    with pytest.raises(ValueError, match="finite"):
        as_point(bad, 1)


# ---------------------------------------------------------------------------
# Peak location
# ---------------------------------------------------------------------------


def test_finds_a_delayed_peak(delayed_bump_kernel):
    """Regression: a local search from lag 0 sees a flat kernel and returns 0.

    The peak value then collapses to 0, the bell-shaped bound degrades to the
    monotone one, and 46% of thinning steps violated M >= lambda.
    """
    peak = locate_peak(delayed_bump_kernel)
    assert peak.lag == pytest.approx(2.5, abs=1e-2)
    assert peak.value == pytest.approx(1.0, abs=1e-6)


def test_finds_a_peak_far_outside_the_initial_window():
    peak = locate_peak(lambda s: np.exp(-((s - 40.0) ** 2)))
    assert peak.lag == pytest.approx(40.0, abs=1e-2)
    assert peak.value == pytest.approx(1.0, abs=1e-6)


def test_monotone_kernel_legitimately_peaks_at_zero():
    """A maximum at lag 0 is the right answer, not a failed search."""
    peak = locate_peak(lambda s: np.exp(-10.0 * s))
    assert peak.lag == 0.0
    assert peak.value == pytest.approx(1.0)


def test_accepts_a_kernel_that_only_handles_scalars():
    """The old search passed the kernel an array, so this spelling crashed."""
    peak = locate_peak(lambda dt: 2.0 * np.exp(-2.0 * float(dt)))
    assert peak.value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(lambda s: np.exp(-10.0 * s), id="monotone"),
        pytest.param(lambda s: np.exp(-((s - 3.0) ** 2)), id="bell"),
        pytest.param(lambda s: np.maximum(0.0, 1.0 - abs(s - 2.5) * 2.0), id="triangular"),
        pytest.param(lambda s: s * np.exp(-2.0 * s), id="gamma-like"),
    ],
)
def test_result_dominates_an_independent_dense_grid(kernel):
    """The bound depends on the returned value dominating the kernel."""
    peak = locate_peak(kernel)
    grid = np.linspace(0.0, max(60.0, 4 * peak.lag + 1.0), 4001)
    assert peak.value >= max(as_float(kernel(float(s))) for s in grid) - 1e-12


def test_refinement_never_degrades_the_scan():
    """A kernel whose maximum sits exactly on a scan node must not get worse."""
    kernel = lambda s: np.maximum(0.0, 1.0 - abs(s - 0.5) * 2.0)
    scanned = locate_peak(kernel, refine=False)
    refined = locate_peak(kernel, refine=True)
    assert refined.value >= scanned.value


def test_negative_kernel_raises():
    """The thinning bound assumes a non-negative temporal kernel."""
    with pytest.raises(ValueError, match="non-negative"):
        locate_peak(lambda s: -1.0 + 0.0 * s, name="temporal kernel")


def test_non_decaying_kernel_warns():
    with pytest.warns(UserWarning, match="has not decayed"):
        locate_peak(lambda s: 1.0 + 0.0 * s, max_expansions=3)
