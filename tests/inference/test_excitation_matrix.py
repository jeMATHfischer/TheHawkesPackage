"""The excitation matrix as a parameter family, and the eigensolver hazard.

`ProcessModel.support` evaluates `branching` on **every** row before
`spec.contains` filters any, so rows carrying `inf` or `nan` reach whatever the
branching callable does. `numpy.linalg.eigvals` raises `LinAlgError` on those
rather than returning `nan` -- which turns a particle the sampler was about to
reject anyway into a crashed fit, on a proposal drawn hundreds of times per
rejuvenation move. Nothing in the happy path exercises it, so it is pinned here.
"""

import numpy as np
import pytest

from hawkes_package.inference import ExcitationMatrix, MultivariateBase
from hawkes_package.inference.families import spectral_radii


def test_the_parameters_are_named_row_major():
    """``theta.reshape(d, d)`` must be the matrix, so the order has to match."""
    matrix = ExcitationMatrix(2)
    assert matrix.spec.names == ("a_0_0", "a_0_1", "a_1_0", "a_1_1")
    assert len(set(matrix.spec.names)) == 4
    assert matrix.size == 4

    theta = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(matrix.matrix(theta), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_the_matrix_round_trips_through_the_parameter_vector():
    matrix = ExcitationMatrix(3)
    rng = np.random.default_rng(0)
    original = rng.uniform(0.01, 2.0, size=(5, 3, 3))
    np.testing.assert_array_equal(matrix.matrices(original.reshape(5, 9)), original)


def test_every_entry_is_a_positive_parameter():
    """The bound's per-term supremum needs a non-negative scale, so the
    coordinate itself is constrained rather than checked after the fact."""
    for parameter in ExcitationMatrix(2).spec.parameters:
        assert parameter.lower == 0.0
        assert parameter.kind == "positive"


def test_the_spectral_radius_reduces_to_alpha_over_beta():
    """At one type the branching matrix is ``[[alpha * mass]]``."""
    alpha, beta = 0.5, 2.0
    radius = ExcitationMatrix(1).spectral_radius(np.array([[alpha]]), np.array([1.0 / beta]))
    assert radius[0] == pytest.approx(alpha / beta, rel=1e-15)


def test_the_spectral_radius_is_below_the_row_sum_bound():
    """``rho(G) <= max_i sum_j G[i, j]`` for a non-negative matrix.

    The cheap majorant, asserted rather than shipped: an eigensolve is what the
    guard uses, because the row sum would refuse stationary processes near the
    boundary. But if the eigensolve ever disagreed with the bound, one of them
    would be wrong.
    """
    rng = np.random.default_rng(1)
    batch = rng.uniform(0.0, 1.0, size=(1000, 5, 5))
    radii = spectral_radii(batch)
    assert np.all(radii <= batch.sum(axis=2).max(axis=1) + 1e-9)


def test_a_non_finite_row_returns_inf_without_raising_or_warning():
    """The hazard this file exists for.

    `filterwarnings = ["error"]` makes the no-warning half of this real: a
    RuntimeWarning from the eigensolver would fail the suite just as loudly as
    an exception, and either would surface as an unexplained failure inside a
    fit rather than as a rejected particle.
    """
    batch = np.zeros((64, 3, 3))
    batch[:] = np.eye(3) * 0.5
    batch[7, 0, 0] = np.inf
    batch[31, 2, 1] = np.nan

    radii = spectral_radii(batch)

    assert np.isinf(radii[7])
    assert np.isinf(radii[31])
    finite = np.setdiff1d(np.arange(64), [7, 31])
    assert np.all(np.isfinite(radii[finite]))
    np.testing.assert_allclose(radii[finite], 0.5)


def test_the_spectral_radius_accepts_a_batch_of_parameter_rows():
    matrix = ExcitationMatrix(2)
    theta = np.array([[0.5, 0.0, 0.0, 0.4], [0.5, 3.0, 0.0, 0.4]])
    # Upper triangular in the second row, so both radii are the larger diagonal.
    radii = matrix.spectral_radius(theta, np.array([0.5, 0.5]))
    np.testing.assert_allclose(radii, [0.25, 0.25])


def test_a_flat_parameter_vector_gives_a_scalar_radius():
    matrix = ExcitationMatrix(2)
    radius = matrix.spectral_radius(np.array([0.5, 0.0, 0.0, 0.4]), 0.5)
    assert np.ndim(radius) == 0


def test_spectral_radii_refuses_a_non_square_batch():
    with pytest.raises(ValueError, match="batch of square matrices"):
        spectral_radii(np.zeros((4, 2, 3)))


def test_the_background_is_one_rate_per_type():
    background = MultivariateBase(3)
    assert background.spec.names == ("mu_0", "mu_1", "mu_2")
    np.testing.assert_array_equal(
        background.rates(np.array([0.4, 0.2, 0.1])), np.array([0.4, 0.2, 0.1])
    )


@pytest.mark.parametrize("family", [ExcitationMatrix, MultivariateBase])
@pytest.mark.parametrize("n_types", [0, -1, 2.5])
def test_the_type_count_is_validated(family, n_types):
    with pytest.raises(ValueError, match="positive whole number"):
        family(n_types)
