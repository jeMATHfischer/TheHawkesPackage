"""The multivariate model, and what it deliberately did *not* have to change.

Most of the sampler is generic over `ProcessModel.spec` and
`ProcessModel.support`, and the spectral radius arrives through the same
injected `branching` callable a scalar ratio did. So the interesting assertions
here are as much about what stayed still as about what moved.

The exception is the one hole this model opens and must close in the same
breath: a multivariate model is `ndim == 0`, so `TemporalLogLikelihood` would
accept it and then read the total intensity where the accepted component's is
wanted.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    MultivariateComponents,
    SoftPlusNonlinearity,
    TemporalLogLikelihood,
    multivariate_model,
    stationarity,
)
from hawkes_package.inference.families import GammaKernel

STABLE = np.array([0.4, 0.2, 0.3, 0.1, 0.5, 0.2, 2.0])


def test_the_coordinates_are_background_then_matrix_then_kernel():
    model = multivariate_model(2)
    assert model.spec.names == ("mu_0", "mu_1", "a_0_0", "a_0_1", "a_1_0", "a_1_1", "beta")
    assert model.ndim == 0
    assert model.family == "multivariate"
    assert isinstance(model.components, MultivariateComponents)
    assert model.components.n_types == 2


def test_the_nonlinearity_does_not_bring_a_second_background():
    """`LinearNonlinearity.spec` is ``(mu,)``, and that must not appear here.

    The background is the vector `MultivariateBase` supplies. Leaving the
    nonlinearity's own scalar in would fit a per-type background plus a shared
    one, a sum no single value of either can identify -- a ridge the posterior
    would wander along forever while every diagnostic looked healthy.
    """
    names = multivariate_model(2).spec.names
    assert names.count("mu") == 0
    assert len(multivariate_model(3).spec) == 3 + 9 + 1


def test_the_support_gates_on_the_spectral_radius():
    model = multivariate_model(2)
    stable = STABLE.copy()
    unstable = STABLE.copy()
    unstable[2] = 3.0  # a_0_0, pushing the radius past one

    assert model.support(stable).tolist() is True or bool(model.support(stable))
    assert not bool(model.support(unstable))
    assert model.branching_ratio(stable) < 1.0
    assert model.branching_ratio(unstable) >= 1.0


def test_the_support_admits_and_refuses_a_whole_batch():
    model = multivariate_model(2)
    rng = np.random.default_rng(0)
    batch = np.tile(STABLE, (256, 1))
    batch[:, 2:6] = rng.uniform(0.0, 4.0, size=(256, 4))

    admitted = model.support(batch)
    ratios = model.branching_ratio(batch)
    np.testing.assert_array_equal(admitted, ratios < 1.0)


def test_every_admitted_row_can_actually_be_simulated():
    """`support` promises the process can be built and run at these values."""
    model = multivariate_model(2)
    rng = np.random.default_rng(1)
    batch = np.tile(STABLE, (60, 1))
    batch[:, 2:6] = rng.uniform(0.0, 1.5, size=(60, 4))
    admitted = batch[model.support(batch)]
    assert len(admitted) > 10, "the sweep should admit a useful fraction"

    for theta in admitted[:20]:
        process = model(theta, rng=0)
        process.simulate(20)
        assert process.events.shape == (2, 20)


def test_the_one_type_branching_ratio_is_the_scalar_one():
    """A one-type multivariate model must agree with `exponential_model`.

    Its coordinates are ``(mu_0, a_0_0, beta)`` against that model's
    ``(mu, alpha, beta)`` -- the same three numbers, which is the point.
    """
    model = multivariate_model(1)
    assert model.spec.names == ("mu_0", "a_0_0", "beta")
    alpha, beta = 0.5, 2.0
    theta = np.array([1.0, alpha, beta])
    assert model.branching_ratio(theta) == pytest.approx(alpha / beta, rel=1e-15)


def test_the_default_kernel_carries_no_amplitude_of_its_own():
    """The matrix is the only amplitude, or nothing is identifiable.

    `ExponentialKernel` carries ``alpha``, and ``A[i, j] * alpha`` would make
    only the product identifiable: scaling every matrix entry by c and dividing
    alpha by c is the same process. The posterior would wander that ridge with
    the effective sample size, the acceptance rate and the move size all
    reporting health, and the matrix it eventually reported would be arbitrary.
    """
    assert "alpha" not in multivariate_model(2).spec.names


def test_scaling_the_matrix_against_the_kernel_changes_the_process():
    """The negative control for the test above: no ridge remains.

    With a unit-amplitude kernel, doubling the matrix and halving the mass is
    *not* the same process -- the branching ratio is unchanged but the intensity
    is not, so the two are distinguishable. Were the amplitude still in the
    kernel, both the ratio and the intensity would be identical and the fit
    would have nothing to prefer.
    """
    model = multivariate_model(2)
    base = STABLE.copy()
    scaled = STABLE.copy()
    scaled[2:6] *= 2.0
    scaled[6] *= 2.0  # beta, so the branching ratio is preserved

    assert model.branching_ratio(base) == pytest.approx(model.branching_ratio(scaled))
    first, second = model(base, rng=4), model(scaled, rng=4)
    first.simulate(30)
    second.simulate(30)
    assert not np.allclose(first.events[0], second.events[0])


def test_the_stationarity_prior_needs_no_change():
    """It takes the ratio through the injected callable and never inspects it."""
    model = multivariate_model(2)
    predicate = stationarity(model.branching_ratio, limit=0.9)
    stable = STABLE.copy()
    loud = STABLE.copy()
    loud[2] = 1.9

    assert bool(predicate(stable))
    assert not bool(predicate(loud))


def test_a_non_monotone_kernel_is_refused():
    """The default bound is the monotone one, as in `monotone_model`."""
    with pytest.raises(ValueError, match="not monotone decreasing"):
        multivariate_model(2, kernel=GammaKernel())


def test_a_saturating_nonlinearity_is_accepted_and_shapes_the_intensity():
    model = multivariate_model(2, nonlinearity=SoftPlusNonlinearity())
    assert model.spec.names[:2] == ("mu_0", "mu_1")
    process = model(STABLE, rng=0)
    process.simulate(20)
    assert process.events.shape == (2, 20)


def test_the_temporal_likelihood_refuses_a_multivariate_model():
    """The silent hole, closed in the commit that opens it.

    A multivariate model is `ndim == 0`, so the existing dimensionality check
    lets it through. `_conditional_intensity` then returns the total across
    types, where the log-sum needs the intensity of the type each event carries.
    The compensator would be right and the log-sum wrong: a converged, plausible
    posterior for a model nobody specified.
    """
    with pytest.raises(ValueError, match="MultivariateLogLikelihood"):
        TemporalLogLikelihood(multivariate_model(2))


def test_the_temporal_likelihood_still_accepts_a_univariate_model():
    from hawkes_package.inference import exponential_model, monotone_model

    assert TemporalLogLikelihood(monotone_model()) is not None
    assert TemporalLogLikelihood(exponential_model()) is not None


def test_the_built_process_matches_the_parameters():
    model = multivariate_model(2)
    process = model(STABLE, rng=0)
    assert isinstance(process, hp.MultivariateHawkes)
    np.testing.assert_array_equal(process.mu, STABLE[:2])
    np.testing.assert_array_equal(process.excitation, STABLE[2:6].reshape(2, 2))
