"""Distributional correctness of the multivariate simulator.

The thinning harness proves ``M >= lambda``, which stops the output being a
Poisson process wearing a Hawkes costume. It does not prove the *type* draw is
right: a bound can dominate perfectly while the uniform is sliced in the wrong
proportions, and the result is a valid Hawkes process with the wrong excitation
matrix. That is what this file is for.

Every threshold below was set by sweeping seeds 0-20 and recording the worst
case, per ``CONTRIBUTING.md``. The measurement sits beside the threshold.
"""

import numpy as np
import pytest
from scipy import stats

import hawkes_package as hp

MU = np.array([0.6, 0.3])
BETA = 2.0

#: No cross-excitation, so the two components are independent univariate Hawkes
#: processes and each may be compared against one simulated on its own.
BLOCK = np.array([[0.5, 0.0], [0.0, 0.4]])

#: Type 1 drives type 0 hard. Upper triangular, so the spectral radius is still
#: ``max(0.5, 0.4) / beta = 0.25`` and the process is comfortably stationary --
#: the contrast is in the coupling, not in being closer to the boundary.
CROSS = np.array([[0.5, 3.0], [0.0, 0.4]])

SIZE = 2000


def stationary_rates(mu, excitation, beta):
    r"""Return :math:`\Lambda = (I - A/\beta)^{-1}\mu`, the stationary rate per type.

    The kernel has mass ``1 / beta``, so ``A / beta`` is the branching matrix and
    each type's long-run rate is its own background plus the offspring every
    type sends it. Independent of the simulator, which is the point: it is the
    answer the type draw has to reproduce rather than a restatement of it.
    """
    branching = np.asarray(excitation, dtype=float) / float(beta)
    return np.linalg.solve(np.eye(len(mu)) - branching, np.asarray(mu, dtype=float))


def component_gaps(process, kind):
    """Inter-arrival times of one type within a joint realisation."""
    return np.diff(process.events[0][process.types == kind])


def reference_gaps(excitation_entry, count, seed):
    """The same, from a univariate process simulated on its own."""
    reference = hp.ExponentialHawkes(np.array([MU[0], excitation_entry, BETA]), rng=seed)
    reference.simulate(count)
    return np.diff(reference.events)


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 7, 13])
def test_block_diagonal_excitation_is_two_independent_processes(seed):
    """With no cross-excitation, type 0's events *are* a univariate Hawkes process.

    The strongest available check on the type draw, because the answer is known
    exactly rather than approximately: if the uniform were sliced in the wrong
    proportions, the type-0 subprocess would carry a different excitation than
    ``A[0, 0]`` and its gap distribution would move with it.
    """
    joint = hp.MultivariateExponentialHawkes(mu=MU, excitation=BLOCK, beta=BETA, rng=seed)
    joint.simulate(SIZE)
    gaps = component_gaps(joint, 0)

    # Seeds 0-20: worst p = 0.054, median 0.43. The threshold is 50x below the
    # worst observed, so this is not a test that passes by luck.
    p = stats.ks_2samp(gaps, reference_gaps(BLOCK[0, 0], gaps.size, seed + 500)).pvalue
    assert p > 1e-3, f"type-0 gaps do not look like their univariate process (p={p:.4g})"


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 7, 13])
def test_the_independence_check_rejects_real_cross_excitation(seed):
    """Guard the guard: the comparison above must have power.

    Without this the test before it could pass because the KS statistic cannot
    tell any two Hawkes processes apart, which would make it evidence of
    nothing.
    """
    joint = hp.MultivariateExponentialHawkes(mu=MU, excitation=CROSS, beta=BETA, rng=seed)
    joint.simulate(SIZE)
    gaps = component_gaps(joint, 0)

    # Seeds 0-20: worst p = 1.9e-22, median 4.5e-37. Ten orders of margin.
    p = stats.ks_2samp(gaps, reference_gaps(CROSS[0, 0], gaps.size, seed + 500)).pvalue
    assert p < 1e-12, f"cross-excitation went undetected (p={p:.4g})"


@pytest.mark.statistical
@pytest.mark.parametrize("seed", [0, 7, 13])
@pytest.mark.parametrize(("name", "excitation"), [("block", BLOCK), ("cross", CROSS)])
def test_the_type_shares_match_the_stationary_rates(seed, name, excitation):
    """How often each type is drawn must match the rate the matrix implies.

    The gap comparison above checks the shape of one component; this checks the
    split between them, which is the quantity `searchsorted` actually decides.
    A bound that dominated but sliced ``(0, M]`` in the wrong proportions would
    pass every invariant check and fail here.
    """
    del name
    process = hp.MultivariateExponentialHawkes(mu=MU, excitation=excitation, beta=BETA, rng=seed)
    process.simulate(SIZE)

    rates = stationary_rates(MU, excitation, BETA)
    expected = rates[0] / rates.sum()
    observed = float(np.count_nonzero(process.types == 0)) / SIZE

    # Seeds 0-20: sd 0.015, worst deviation 0.040 on the block-diagonal matrix.
    # 0.06 is four standard deviations and 1.5x the worst case. The spread is
    # wider than a binomial's because events arrive in clusters, so the count is
    # overdispersed -- do not tighten this to the binomial standard error.
    assert abs(observed - expected) < 0.06, (
        f"type-0 share {observed:.4f} against the stationary {expected:.4f}"
    )
