"""Does fitting a multivariate model find the matrix it was simulated from.

Everything else in the multivariate suite checks that a piece computes what it
says. This checks the only thing a user cares about: simulate from a known
excitation matrix, fit, and see whether the intervals cover.

Two guards on the guard, in the spirit of `test_inference_recovery`. A coverage
count alone can pass while measuring nothing -- so a second test requires the
posterior to *distinguish* a matrix with strong cross-excitation from one with
almost none, and every fit asserts on `min_move_size`, because a frozen
rejuvenation kernel reports a very tight posterior centred wherever resampling
left it, with the effective sample size and the acceptance rate both looking
healthy.

Thresholds swept over seeds 0-20; the measurement sits beside each one.
Each fit is about 90 seconds, so this file is slow and runs in the coverage
job rather than in the ten-job matrix.
"""

import numpy as np
import pytest

from hawkes_package.inference import (
    ConstrainedPrior,
    History,
    IndependentPrior,
    LogNormal,
    MultivariateExponentialLogLikelihood,
    fit_smc,
    multivariate_model,
)

pytestmark = [pytest.mark.statistical, pytest.mark.slow]

#: (mu_0, mu_1, a_00, a_01, a_10, a_11, beta)
TRUTH = np.array([0.6, 0.3, 0.5, 0.2, 0.4, 0.6, 2.0])

#: Type 1 barely drives type 0.
QUIET = np.array([0.6, 0.3, 0.5, 0.02, 0.4, 0.6, 2.0])

#: Type 1 drives type 0 hard. Still stationary -- the spectral radius of an
#: upper-triangular block is set by the diagonal, not by the corner.
LOUD = np.array([0.6, 0.3, 0.5, 1.2, 0.4, 0.6, 2.0])

N_PARTICLES = 384
SIZE = 1000
BLOCKS = 5


def fit(truth, seed):
    """Simulate `SIZE` events at `truth` and fit them back."""
    model = multivariate_model(2)
    prior = ConstrainedPrior(
        IndependentPrior(tuple([LogNormal(-0.7, 1.0)] * 6 + [LogNormal(0.7, 0.7)])),
        model.support,
    )
    process = model(truth, rng=seed)
    process.simulate(SIZE)
    history = History.from_multivariate_events(
        process.events, n_types=2, end=float(process.events[0, -1])
    )
    return fit_smc(
        MultivariateExponentialLogLikelihood(model),
        prior,
        history,
        blocks=BLOCKS,
        n_particles=N_PARTICLES,
        rng=seed,
    )


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_the_posterior_covers_the_excitation_matrix(seed):
    """Seven parameters, 90% marginals, and a floor on how many must cover.

    A count rather than "all seven": at nominal 90% coverage the expected number
    of misses across seven parameters is 0.7, so demanding all seven would be
    demanding a run that got lucky. Measured over seeds 0-20: minimum 4 of 7 (at
    seeds 10 and 19), mean 5.8, and the threshold sits one below the worst.
    `a_0_1` is the coordinate that misses most often -- it is the entry the data
    constrains least, since type 1's influence on type 0 is visible only through
    type 0's own clustering.
    """
    smc = fit(TRUTH, seed)
    cloud = smc.cloud
    lower, upper = cloud.quantile(0.05), cloud.quantile(0.95)
    covered = int(np.sum((lower <= TRUTH) & (upper >= TRUTH)))

    assert covered >= 3, (
        f"only {covered}/7 truths inside their 90% intervals; "
        f"means {np.round(cloud.mean(), 3).tolist()}"
    )
    # A collapsed cloud reports a tight posterior centred on resampling noise,
    # with a perfect ESS and an acceptance rate of 1.000. Only the distance the
    # particles actually travelled tells the two apart.
    # Seeds 0-20: minimum 0.631, so this has an order of magnitude of margin.
    assert smc.diagnostics.min_move_size > 0.05, (
        f"the rejuvenation kernel barely moved (min_move_size="
        f"{smc.diagnostics.min_move_size:.3g}); the posterior is resampling noise"
    )


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_the_fit_distinguishes_cross_excitation_from_none(seed):
    """Guard the guard: coverage means nothing if the posterior ignores the data.

    A prior that merely liked small positive numbers would cover `a_0_1 = 0.2`
    handsomely and would also "cover" it when the truth was 1.2. So the two fits
    must separate: the quiet posterior's 95th percentile must sit below the loud
    posterior's 5th. Measured over seeds 0-7, the two never came close to
    touching -- quiet reached at most 0.446 and loud fell no lower than 0.768.
    """
    index = 3  # a_0_1
    quiet, loud = fit(QUIET, seed).cloud, fit(LOUD, seed).cloud

    quiet_high = float(quiet.quantile(0.95)[index])
    loud_low = float(loud.quantile(0.05)[index])

    assert quiet_high < loud_low, (
        f"the posterior cannot tell a_0_1=0.02 from a_0_1=1.2: quiet q95 "
        f"{quiet_high:.3f} against loud q05 {loud_low:.3f}"
    )
    assert float(quiet.mean()[index]) < float(loud.mean()[index])
