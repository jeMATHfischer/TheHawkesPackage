"""Panelled quadrature for the compensator.

Checked against integrals with closed forms, to a tolerance that would catch a
misplaced panel edge rather than merely a rough rule: with the panels cut at the
jumps and the integrand analytic between them, an order-8 Gauss-Legendre rule is
at machine precision, and anything looser means a break is in the wrong place.
"""

import numpy as np
import pytest

from hawkes_package.inference import _compensator


def test_breakpoints_span_the_interval_and_cut_at_the_events():
    times = np.array([0.5, 1.5, 2.5, 12.0])
    edges = _compensator.breakpoints(0.0, 3.0, times)
    np.testing.assert_allclose(edges, [0.0, 0.5, 1.5, 2.5, 3.0])


def test_events_outside_the_interval_add_no_panels():
    times = np.array([-1.0, 0.0, 5.0, 9.0])
    edges = _compensator.breakpoints(1.0, 4.0, times)
    np.testing.assert_allclose(edges, [1.0, 4.0])


def test_extra_lags_break_the_panel_at_every_event_plus_the_lag():
    edges = _compensator.breakpoints(0.0, 3.0, np.array([1.0, 2.0]), extra_lags=(0.5,))
    np.testing.assert_allclose(edges, [0.0, 1.0, 1.5, 2.0, 2.5, 3.0])


def test_a_degenerate_interval_gives_an_empty_rule():
    nodes, weights = _compensator.panels(_compensator.breakpoints(2.0, 2.0, np.array([])))
    assert nodes.size == 0
    assert weights.size == 0
    assert _compensator.integrate(lambda t: 1.0, nodes, weights) == 0.0


def test_weights_are_positive_and_sum_to_the_length():
    edges = _compensator.breakpoints(0.0, 7.0, np.array([1.0, 3.5, 6.0]))
    nodes, weights = _compensator.panels(edges)
    assert np.all(weights > 0)
    assert float(np.sum(weights)) == pytest.approx(7.0, rel=1e-14)
    assert np.all((nodes > 0.0) & (nodes < 7.0))


@pytest.mark.parametrize("order", [4, 8, 16])
def test_a_polynomial_is_integrated_exactly(order):
    """Gauss-Legendre with `order` nodes is exact to degree ``2*order - 1``."""
    degree = 2 * order - 1
    nodes, weights = _compensator.panels([0.0, 2.0], order)
    value = _compensator.integrate(lambda t: t**degree, nodes, weights)
    assert value == pytest.approx(2.0 ** (degree + 1) / (degree + 1), rel=1e-12)


def test_the_exponential_compensator_matches_its_closed_form():
    r"""``int mu + alpha sum exp(-beta (s - t_i))`` against its analytic value.

    The identity the whole exponential fast path rests on, checked here against
    the quadrature that the general path would use instead.
    """
    mu, alpha, beta = 1.7, 0.6, 1.3
    times = np.array([0.4, 1.1, 2.9, 3.0, 6.2])
    end = 8.0

    def intensity(t):
        past = times[times < t]
        return mu + alpha * np.sum(np.exp(-beta * (t - past)))

    edges = _compensator.breakpoints(0.0, end, times)
    numerical = _compensator.integrate(intensity, *_compensator.panels(edges))
    analytic = mu * end + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (end - times)))
    assert numerical == pytest.approx(analytic, rel=1e-12)


def test_a_kink_inside_a_panel_is_detected():
    """Guard the guard: the resolution check must fire on an undeclared kink.

    The triangular kernel this project uses throughout kinks at ``t_i + 0.5``.
    Integrating across that with a smooth rule converges at second order rather
    than spectrally, and the error has a sign -- so a check that never fired
    would let the compensator be quietly wrong.
    """
    times = np.array([1.0, 2.0, 4.0])

    def triangular_intensity(t):
        lags = t - times[times < t]
        rise = (lags > 0) & (lags < 0.5)
        fall = (lags >= 0.5) & (lags < 1.0)
        return 1.0 + float(np.sum(2 * lags * rise + (-2 * lags + 2) * fall))

    edges = _compensator.breakpoints(0.0, 6.0, times)
    with pytest.warns(UserWarning, match="not resolved"):
        _compensator.check_resolution(triangular_intensity, edges)

    # Declaring the kink puts a panel edge on it, and the warning goes away.
    declared = _compensator.breakpoints(0.0, 6.0, times, extra_lags=(0.5, 1.0))
    _compensator.check_resolution(triangular_intensity, declared)


def test_a_smooth_integrand_passes_the_resolution_check():
    times = np.array([0.4, 1.1, 2.9])

    def intensity(t):
        return 1.0 + float(np.sum(np.exp(-2.0 * (t - times[times < t]))))

    edges = _compensator.breakpoints(0.0, 5.0, times)
    fine = _compensator.check_resolution(intensity, edges)
    coarse = _compensator.integrate(intensity, *_compensator.panels(edges))
    assert fine == pytest.approx(coarse, rel=1e-12)
