"""The unconstraining transforms and their Jacobians.

The Jacobian is the part worth testing hardest. It is invisible in the output
-- a posterior computed without it is still a posterior, just of the wrong
distribution -- so it is checked against central differences rather than against
a second copy of the same algebra.
"""

import math

import numpy as np
import pytest

from hawkes_package.inference import Parameter, ParameterSpec

SPECS = {
    "positive": ParameterSpec((Parameter("a"), Parameter("b"))),
    "shifted": ParameterSpec((Parameter("a", lower=1.0), Parameter("b", lower=-2.0))),
    "real": ParameterSpec((Parameter("a", lower=-np.inf), Parameter("b", lower=-np.inf))),
    "bounded_above": ParameterSpec(
        (Parameter("a", lower=-np.inf, upper=3.0), Parameter("b", lower=-np.inf, upper=0.0))
    ),
    "interval": ParameterSpec(
        (Parameter("a", lower=0.0, upper=1.0), Parameter("b", lower=-1.0, upper=4.0))
    ),
}


@pytest.fixture(params=sorted(SPECS), ids=sorted(SPECS))
def spec(request):
    return SPECS[request.param]


@pytest.fixture
def unconstrained(rng):
    """A batch of unconstrained points, including some far out in both tails."""
    ordinary = rng.normal(0.0, 1.5, size=(24, 2))
    extreme = np.array([[-40.0, 40.0], [40.0, -40.0], [0.0, 0.0], [-700.0, 700.0]])
    return np.vstack([ordinary, extreme])


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_transforms_round_trip(spec, unconstrained):
    theta = spec.to_constrained(unconstrained)
    finite = np.all(np.isfinite(theta), axis=1) & spec.contains(theta)
    assert np.any(finite)
    back = spec.to_unconstrained(theta[finite])
    np.testing.assert_allclose(back, unconstrained[finite], rtol=1e-9, atol=1e-9)


def test_constrained_values_respect_the_bounds(spec, unconstrained):
    theta = spec.to_constrained(unconstrained)
    inside = spec.contains(theta)
    assert np.all(theta[inside] > spec.lower)
    assert np.all(theta[inside] < spec.upper)


def test_a_single_vector_keeps_its_rank(spec):
    z = np.array([0.3, -0.7])
    theta = spec.to_constrained(z)
    assert theta.shape == (2,)
    assert spec.to_unconstrained(theta).shape == (2,)
    assert np.ndim(spec.log_abs_det_jacobian(z)) == 0
    assert np.ndim(spec.contains(theta)) == 0


# ---------------------------------------------------------------------------
# The Jacobian
# ---------------------------------------------------------------------------


def test_log_jacobian_matches_central_differences(spec, rng):
    """Against finite differences of the transform, not against more algebra.

    The Jacobian is diagonal, so its log determinant is the sum of
    ``log |d theta_j / d z_j|``. Differentiating the transform numerically is the
    only check that does not reuse the derivation being checked.
    """
    z = rng.normal(0.0, 1.0, size=(12, 2))
    step = 1e-6
    expected = np.zeros(z.shape[0])
    for j in range(z.shape[1]):
        forward, backward = z.copy(), z.copy()
        forward[:, j] += step
        backward[:, j] -= step
        derivative = (spec.to_constrained(forward)[:, j] - spec.to_constrained(backward)[:, j]) / (
            2 * step
        )
        expected += np.log(np.abs(derivative))
    np.testing.assert_allclose(spec.log_abs_det_jacobian(z), expected, rtol=1e-5, atol=1e-6)


def test_the_interval_jacobian_survives_a_deep_tail():
    """``log1p(exp(-z))`` overflows where ``logaddexp`` does not.

    A rejuvenation move on a cloud collapsed against a boundary proposes exactly
    this, and the naive spelling returns ``inf`` there -- which propagates to a
    weight of ``nan``, not to a rejection.
    """
    spec = ParameterSpec((Parameter("p", lower=0.0, upper=1.0),))
    z = np.array([[-800.0], [800.0], [0.0]])
    values = spec.log_abs_det_jacobian(z)
    assert np.all(np.isfinite(values))
    # log(1) - z - 2*log(1 + exp(-z)) is -|z| in both tails, and -2*log 2 at 0.
    np.testing.assert_allclose(values, [-800.0, -800.0, -2 * math.log(2.0)], rtol=1e-12)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_contains_is_strict_at_the_bounds():
    spec = ParameterSpec((Parameter("a", lower=0.0, upper=1.0),))
    values = np.array([[0.0], [1.0], [0.5], [np.nan], [np.inf]])
    np.testing.assert_array_equal(spec.contains(values), [False, False, True, False, False])


def test_unconstraining_a_value_on_the_boundary_raises():
    spec = ParameterSpec((Parameter("a"),))
    with pytest.raises(ValueError, match="outside the open bounds"):
        spec.to_unconstrained(np.array([[0.0]]))


def test_an_overflowing_coordinate_is_rejected_rather_than_raised():
    """A wild proposal is a rejection, not a failed run."""
    spec = ParameterSpec((Parameter("a"),))
    theta = spec.to_constrained(np.array([[1e5]]))
    assert np.isinf(theta[0, 0])
    assert not spec.contains(theta)[0]


def test_a_wrong_width_is_refused():
    spec = ParameterSpec((Parameter("a"), Parameter("b")))
    with pytest.raises(ValueError, match="n_dim=2"):
        spec.to_constrained(np.zeros((3, 5)))


@pytest.mark.parametrize(
    ("lower", "upper", "match"),
    [(1.0, 1.0, "non-empty"), (2.0, 1.0, "non-empty"), (np.nan, 1.0, "nan")],
)
def test_an_empty_interval_is_refused(lower, upper, match):
    with pytest.raises(ValueError, match=match):
        Parameter("a", lower=lower, upper=upper)


def test_an_unnamed_parameter_is_refused():
    with pytest.raises(ValueError, match="needs a name"):
        Parameter("")


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSpec((Parameter("a"), Parameter("a")))


def test_an_empty_spec_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        ParameterSpec(())


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_prefixed_and_concat_compose_without_collision():
    kernel = ParameterSpec((Parameter("alpha"), Parameter("beta")))
    joined = kernel.prefixed("fast_").concat(kernel.prefixed("slow_"))
    assert joined.names == ("fast_alpha", "fast_beta", "slow_alpha", "slow_beta")
    assert len(joined) == 4
    assert joined.index("slow_beta") == 3


def test_index_names_the_spec_it_could_not_find_in():
    spec = ParameterSpec((Parameter("mu"),))
    with pytest.raises(KeyError, match=r"\['mu'\]"):
        spec.index("alpha")


def test_kind_reports_which_transform_applies():
    assert Parameter("a").kind == "positive"
    assert Parameter("a", lower=-np.inf).kind == "real"
    assert Parameter("a", lower=-np.inf, upper=1.0).kind == "bounded_above"
    assert Parameter("a", lower=0.0, upper=1.0).kind == "interval"
