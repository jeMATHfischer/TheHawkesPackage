"""Parameterised kernels, backgrounds and nonlinearities.

Two properties matter and both are closed forms the sampler relies on rather
than recomputing. The **peak** is checked against
:func:`~hawkes_package._numerics.locate_peak`, the 513-point scan the process
classes would otherwise run once per particle per move; agreeing with it is what
makes skipping it safe. The **mass** is checked against numerical integration,
because it decides the branching ratio, and a branching ratio guessed wrong does
not raise -- the particle survives, the likelihood on a finite window is finite,
and only the excitation estimate is wrong.
"""

import numpy as np
import pytest
from scipy import integrate

import hawkes_package as hp
from hawkes_package._numerics import locate_peak
from hawkes_package.inference import (
    ConstantBase,
    ExponentialKernel,
    GammaKernel,
    GaussianSpatial,
    History,
    LinearNonlinearity,
    SoftPlusNonlinearity,
    TemporalLogLikelihood,
    bell_shape_model,
    exponential_model,
    monotone_model,
)

TEMPORAL_KERNELS = {
    "exponential": (ExponentialKernel(), np.array([0.6, 2.0])),
    "gamma-1.5": (GammaKernel(), np.array([0.5, 1.5, 2.0])),
    "gamma-2.5": (GammaKernel(), np.array([0.4, 2.5, 2.0])),
    "gamma-4": (GammaKernel(), np.array([0.7, 4.0, 3.0])),
}


@pytest.fixture(params=sorted(TEMPORAL_KERNELS), ids=sorted(TEMPORAL_KERNELS))
def temporal_kernel(request):
    return TEMPORAL_KERNELS[request.param]


def test_the_analytic_peak_matches_the_numerical_scan(temporal_kernel):
    """To 1e-6, which is what makes skipping `locate_peak` a saving and not a risk."""
    family, theta = temporal_kernel
    analytic = family.peak(theta)
    scanned = locate_peak(family.build(theta), name="kernel")
    assert analytic.lag == pytest.approx(scanned.lag, abs=1e-6)
    assert analytic.value == pytest.approx(scanned.value, rel=1e-6)


def test_the_peak_really_is_the_maximum(temporal_kernel):
    """Not merely equal to the scan: no lag anywhere may exceed it."""
    family, theta = temporal_kernel
    kernel = family.build(theta)
    lags = np.linspace(0.0, 40.0, 20_001)
    assert np.max(kernel(lags)) <= family.peak(theta).value * (1 + 1e-12)


def test_the_mass_matches_numerical_integration(temporal_kernel):
    family, theta = temporal_kernel
    kernel = family.build(theta)
    integral, _ = integrate.quad(lambda s: float(kernel(s)), 0.0, np.inf, limit=200)
    assert float(family.mass(theta)) == pytest.approx(integral, rel=1e-6)


def test_kernels_are_non_negative_and_vanish_at_infinity(temporal_kernel):
    family, theta = temporal_kernel
    kernel = family.build(theta)
    lags = np.linspace(0.0, 60.0, 5001)
    values = np.asarray(kernel(lags))
    assert np.all(values >= 0.0)
    assert values[-1] < 1e-12


def test_the_mass_is_batched(temporal_kernel):
    family, theta = temporal_kernel
    batch = np.vstack([theta, 2.0 * theta])
    masses = family.mass(batch)
    assert masses.shape == (2,)
    assert masses[0] == pytest.approx(float(family.mass(theta)))


def test_the_gamma_kernel_is_zero_at_and_below_zero_lag():
    """It rises from zero; a kernel that did not would have no thinning bound."""
    kernel = GammaKernel().build(np.array([0.5, 2.5, 2.0]))
    assert float(kernel(0.0)) == 0.0
    np.testing.assert_array_equal(kernel(np.array([-1.0, 0.0])), [0.0, 0.0])


def test_the_gamma_peak_survives_a_rate_that_would_overflow():
    """``rate ** shape`` overflows where the kernel's value is perfectly ordinary."""
    peak = GammaKernel().peak(np.array([1.0, 4.0, 400.0]))
    assert np.isfinite(peak.value)
    assert peak.lag == pytest.approx(3.0 / 400.0)


def test_the_exponential_kernel_peaks_at_lag_zero():
    family = ExponentialKernel()
    assert family.monotone is True
    assert family.peak(np.array([0.6, 2.0])).lag == 0.0
    assert family.peak(np.array([0.6, 2.0])).value == 0.6


def test_the_gamma_kernel_declares_itself_non_monotone():
    assert GammaKernel().monotone is False


def test_a_kernel_family_refuses_the_wrong_number_of_parameters():
    with pytest.raises(ValueError, match="expected 2 parameter"):
        ExponentialKernel().build(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2])
def test_the_gaussian_spatial_kernel_has_unit_mass_on_the_model_space(ndim):
    """Which is what makes the temporal mass a *sufficient* branching bound."""
    kernel = GaussianSpatial(ndim).build(np.array([0.7]))
    if ndim == 1:
        integral, _ = integrate.quad(lambda d: float(kernel(abs(d))), -30, 30)
    else:
        integral, _ = integrate.quad(lambda r: 2 * np.pi * r * float(kernel(r)), 0, 30)
    assert integral == pytest.approx(1.0, rel=1e-6)
    assert float(GaussianSpatial(ndim).mass(np.array([0.7]))) == 1.0


def test_the_gaussian_spatial_kernel_is_strictly_positive():
    """The precondition of the cached likelihood backend."""
    kernel = GaussianSpatial(1).build(np.array([0.5]))
    assert np.all(np.asarray(kernel(np.linspace(0.0, 4.0, 501))) > 0.0)


def test_the_spatial_scale_is_reported():
    assert float(GaussianSpatial(1).min_scale(np.array([0.3]))) == 0.3


def test_a_zero_dimensional_spatial_kernel_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        GaussianSpatial(0)


def test_the_constant_background_is_per_unit_measure():
    """Reading it as a rate scales the fit by the area of the surface."""
    background = ConstantBase()
    nodes = np.linspace(-np.pi, np.pi, 17).reshape(-1, 1)
    np.testing.assert_allclose(background.at(np.array([0.4]), nodes), 0.4)
    assert background.build(np.array([0.4]))(np.array([1.0])) == 0.4


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------


def test_the_linear_nonlinearity_is_the_identity_plus_the_background():
    phi = LinearNonlinearity().build(np.array([1.5]))
    np.testing.assert_allclose(phi(np.array([0.0, 2.0, -0.5])), [1.5, 3.5, 1.0])
    assert LinearNonlinearity().lipschitz == 1.0


def test_the_softplus_nonlinearity_saturates_and_stays_above_the_background():
    phi = SoftPlusNonlinearity(scale=2.0).build(np.array([0.5]))
    values = phi(np.array([-50.0, 0.0, 50.0]))
    assert values[0] == pytest.approx(0.5, abs=1e-9), "bounded below by mu"
    assert values[1] == pytest.approx(0.5 + 2.0 * np.log(2.0))
    # Increasing, and 1-Lipschitz -- so `mass(kernel) < 1` still bounds it.
    assert np.all(np.diff(values) > 0)
    assert (values[2] - values[1]) < 50.0
    assert SoftPlusNonlinearity().lipschitz == 1.0


def test_the_softplus_survives_an_excitation_that_would_overflow_exp():
    """A burst reaches a few hundred, where `log1p(exp(x))` sees `inf`."""
    phi = SoftPlusNonlinearity(scale=1.0).build(np.array([1.0]))
    assert phi(np.array([800.0]))[0] == pytest.approx(801.0, rel=1e-9)


def test_a_non_positive_saturation_scale_is_refused():
    with pytest.raises(ValueError, match="scale must be positive"):
        SoftPlusNonlinearity(scale=0.0)


# ---------------------------------------------------------------------------
# Models built from them
# ---------------------------------------------------------------------------


MODELS = {
    "exponential": (exponential_model(), np.array([2.0, 0.5, 1.0])),
    "monotone": (monotone_model(), np.array([2.0, 0.5, 1.0])),
    "monotone-softplus": (
        monotone_model(nonlinearity=SoftPlusNonlinearity(2.0)),
        np.array([1.0, 0.4, 1.0]),
    ),
    "bell": (bell_shape_model(), np.array([1.0, 0.5, 2.5, 2.0])),
}


@pytest.fixture(params=sorted(MODELS), ids=sorted(MODELS))
def model_and_theta(request):
    return MODELS[request.param]


def test_every_model_builds_a_process_that_simulates(model_and_theta):
    model, theta = model_and_theta
    process = model(theta, rng=0)
    process.simulate(20)
    assert len(process.events) == 20
    assert np.all(np.diff(process.events) > 0)


def test_every_model_has_a_finite_log_likelihood(model_and_theta):
    model, theta = model_and_theta
    process = model(theta, rng=1)
    process.simulate(30)
    history = History.from_simulation(process)
    assert np.isfinite(TemporalLogLikelihood(model).total(theta, history))


def test_a_model_refuses_a_parameter_outside_its_support(model_and_theta):
    model, theta = model_and_theta
    supercritical = theta.copy()
    supercritical[model.spec.index("alpha")] = 50.0
    assert not bool(model.support(supercritical))
    with pytest.raises(ValueError, match="outside the support"):
        model(supercritical)


def test_a_model_refuses_the_wrong_number_of_parameters(model_and_theta):
    model, theta = model_and_theta
    with pytest.raises(ValueError, match="expected"):
        model(np.append(theta, 1.0))


def test_the_bell_shaped_model_passes_the_analytic_peak_through():
    """Not the numerical search: it would run once per particle per move."""
    model = bell_shape_model()
    theta = np.array([1.0, 0.5, 2.5, 2.0])
    process = model(theta, rng=0)
    analytic = GammaKernel().peak(theta[1:])
    assert process.ext == pytest.approx(analytic.lag)
    assert process.peak == pytest.approx(analytic.value)


def test_a_bell_shaped_kernel_is_refused_by_the_monotone_model():
    """`MonotoneKernelHawkes` bounds by the current value; a rising kernel exceeds it."""
    with pytest.raises(ValueError, match="not monotone decreasing"):
        monotone_model(kernel=GammaKernel())


def test_a_family_without_an_analytic_peak_is_refused():
    class NoPeak:
        monotone = False

        @property
        def spec(self):
            return ExponentialKernel().spec

        def build(self, theta):
            return ExponentialKernel().build(theta)

        def mass(self, theta):
            return ExponentialKernel().mass(theta)

    with pytest.raises(ValueError, match="no analytic peak"):
        bell_shape_model(kernel=NoPeak())


def test_the_branching_ratio_carries_the_lipschitz_constant():
    """Bremaud-Massoulie, not the branching ratio of a linear process that is absent."""
    model = monotone_model(nonlinearity=SoftPlusNonlinearity(2.0))
    theta = np.array([1.0, 0.6, 2.0])
    assert float(model.branching_ratio(theta)) == pytest.approx(0.3)  # 1.0 * alpha/beta


def test_the_spatial_scale_is_bounded_below_by_the_quadrature():
    """A kernel narrower than half a panel falls between the nodes and vanishes.

    The quadrature then reports the background integral, the excitation becomes
    invisible to the thinning while the location sampler still sees it, and the
    process degenerates towards Poisson in time without anything raised.
    """
    from hawkes_package.inference import spatio_temporal_model

    model = spatio_temporal_model(hp.Circle(), n_quad=32)
    panel = 2.0 * (2 * np.pi) / 32
    assert bool(model.support(np.array([0.5, 0.5, 2.0, panel * 1.1])))
    assert not bool(model.support(np.array([0.5, 0.5, 2.0, panel * 0.9])))
