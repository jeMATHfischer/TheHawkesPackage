"""Does fitting a marked model find the mark law it was simulated from.

The rest of the marked suite checks that each piece computes what it says. This
checks the thing a user cares about: simulate at a known ``(mu, alpha, beta,
scale, b_value)``, fit, and see whether the intervals cover.

The second test is the one that earns its runtime. `MarkedLogLikelihood`
includes the mark density by default, and `test_marked_likelihood.py` measures
what dropping it does to the *likelihood* -- three different rates, one identical
value. This measures what it does to the **posterior**, which is what a user
would actually read: the `b_value` marginal goes from a standard deviation of
0.14 to one of 0.71 or worse, which is the prior. Nothing about the fit says so;
the diagnostics are healthy either way.

Thresholds swept over seeds 0-20, with the measurement beside each. Each fit is
12 to 20 seconds, so this file is slow and runs in the coverage job rather than
in the ten-job matrix.
"""

import numpy as np
import pytest

from hawkes_package.inference import (
    ConstrainedPrior,
    History,
    IndependentPrior,
    LogNormal,
    MarkedLogLikelihood,
    fit_smc,
    marked_model,
)

pytestmark = [pytest.mark.statistical, pytest.mark.slow]

#: (mu, alpha, beta, scale, b_value). Branching ratio 0.225: the excitation is
#: real but the catalogue is not one long cascade.
TRUTH = np.array([1.0, 0.3, 2.0, 0.5, 1.5])

N_PARTICLES = 96
SIZE = 120
BLOCKS = 4


def fit(seed, *, include_mark_density=True):
    """Simulate `SIZE` marked events at `TRUTH` and fit them back."""
    model = marked_model()
    process = model(TRUTH, rng=seed)
    process.simulate(SIZE)
    history = History.from_marked_events(process.events, end=float(process.events[0, -1]))
    prior = ConstrainedPrior(
        IndependentPrior(
            (
                LogNormal(0.0, 0.7),  # mu
                LogNormal(-1.0, 0.7),  # alpha
                LogNormal(0.7, 0.7),  # beta
                LogNormal(-0.7, 0.7),  # scale
                LogNormal(0.4, 0.7),  # b_value
            )
        ),
        model.support,
    )
    return fit_smc(
        MarkedLogLikelihood(model, include_mark_density=include_mark_density),
        prior,
        history,
        blocks=BLOCKS,
        n_particles=N_PARTICLES,
        rng=seed,
    )


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_the_posterior_covers_the_marked_parameters(seed):
    """Five parameters and 90% marginals, with a floor rather than "all five".

    At nominal 90% coverage the expected number of misses across five
    coordinates is 0.5, so demanding all five would be demanding a lucky run.
    Measured over seeds 0-20: 5 of 5 on sixteen seeds and 4 of 5 on five, so the
    minimum observed is 4 and the threshold sits one below it. `b_value` is the
    coordinate that misses -- it is estimated from 120 marks alone, where
    ``1/mean(m)`` has a standard deviation of 0.14, and the posterior tracks that
    sample statistic rather than the truth behind it.
    """
    smc = fit(seed)
    cloud = smc.cloud
    lower, upper = cloud.quantile(0.05), cloud.quantile(0.95)
    covered = int(np.sum((lower <= TRUTH) & (upper >= TRUTH)))

    assert covered >= 3, (
        f"only {covered}/5 truths inside their 90% intervals; "
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


@pytest.mark.parametrize("seed", [0, 11])
def test_dropping_the_mark_density_reports_the_prior_for_b_value(seed):
    """What the missing term costs a *posterior*, not just a likelihood value.

    `b_value` appears in the mark density and nowhere else in the model -- its
    only other appearance is the support boundary ``scale < b_value``. Drop the
    term and the fit reports the prior truncated at that line, which looks like
    a wide but honest marginal rather than like a parameter nobody estimated.

    Measured over seeds 0-20: with the term the marginal's standard deviation is
    0.087 to 0.147; without it, 0.714 to 2.032. The ratio is never below 5.29,
    and the threshold below is 4. Both fits pass every diagnostic.
    """
    with_marks = fit(seed).cloud
    without = fit(seed, include_mark_density=False).cloud

    tight = float(with_marks.std()[4])
    loose = float(without.std()[4])
    assert loose > 4.0 * tight, (
        f"dropping the mark density should leave `b_value` at its prior, but the "
        f"marginal only widened from {tight:.3f} to {loose:.3f}"
    )
    # And the term is what puts the estimate in the right place: the informed
    # posterior sits near the truth, the uninformed one near the prior's mass.
    assert abs(float(with_marks.mean()[4]) - float(TRUTH[4])) < 0.5
