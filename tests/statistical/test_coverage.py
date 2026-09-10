"""Do the credible intervals cover at the rate they claim?

The most persuasive artefact a statistical library can ship, and the most
seductive threshold in its own suite. Two mistakes are easy here and this file
is arranged around both.

**A one-sided assertion cannot fail in the interesting direction.** A posterior
twice as wide as it should be covers the truth *more* often than nominal, so a
floor on the coverage passes it with room to spare. The check is therefore run
at two levels: 95%, where a fit that is too tight fails, and **50%, where a fit
that is too wide fails** -- a doubled posterior covers about 80% of the time at
that level, against the 50% it should.

**A coverage count is a statistical threshold like any other.** Over 40
replicates a nominal 95% has a standard error near 2 points, so 98% is not a bug
and 91% is not proof of one. The measured numbers are below, and the thresholds
sit outside them by more than the noise:

=========  ====================  ====================
level      per coordinate        total (120 checks)
=========  ====================  ====================
95%        98% / 95% / 100%      117/120 = 98%
50%        45% / 60% / 65%       68/120 = 57%
=========  ====================  ====================

Both sit slightly above nominal, which is what a proper prior does at 500
events: it adds a little information, and a little information makes an interval
a little conservative.

**Every replicate asserts on `move_size`.** A collapsed cloud reports a very
tight posterior centred wherever resampling noise left it, with a perfect
effective sample size and an acceptance rate of 1.000 -- so a coverage harness
that did not check it could be measuring resampling noise and reporting it as
calibration.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior,
    ExponentialLogLikelihood,
    History,
    IndependentPrior,
    LogNormal,
    exponential_model,
    fit_smc,
)

pytestmark = [pytest.mark.statistical, pytest.mark.slow]

TRUTH = np.array([1.0, 0.5, 2.0])
SIZE = 500
REPLICATES = 40


def coverage(level):
    """Return the per-coordinate coverage counts and the worst `move_size`."""
    model = exponential_model()
    prior = ConstrainedPrior(
        IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )
    likelihood = ExponentialLogLikelihood(model)
    lower_q, upper_q = (1 - level) / 2, 1 - (1 - level) / 2

    counts = np.zeros(3, dtype=int)
    worst_move = np.inf
    for seed in range(REPLICATES):
        process = hp.ExponentialHawkes(TRUTH, rng=seed)
        process.simulate(SIZE)
        history = History.from_simulation(process)
        # The fitting seed is offset from the simulation seed on purpose: reusing
        # one seed for both would correlate the data with the sampler's own
        # randomness, and a coverage number is exactly the quantity that would
        # quietly benefit.
        smc = fit_smc(likelihood, prior, history, blocks=6, n_particles=256, rng=1000 + seed)
        lower, upper = smc.cloud.quantile(lower_q), smc.cloud.quantile(upper_q)
        counts += ((lower <= TRUTH) & (upper >= TRUTH)).astype(int)
        worst_move = min(worst_move, float(smc.diagnostics.min_move_size))
    return counts, worst_move


def test_a_nominal_95_percent_interval_covers_about_95_percent():
    """Measured 117 of 120; the floor is 105, which is four standard errors below.

    The floor is what a genuinely over-tight posterior fails. It is set from the
    measurement rather than from the nominal rate, because a threshold at 95%
    exactly would fail one run in two by construction.
    """
    counts, worst_move = coverage(0.95)
    total = int(counts.sum())

    assert total >= 105, (
        f"only {total}/120 nominal 95% intervals covered the truth "
        f"({counts.tolist()} per coordinate); the posterior is too tight"
    )
    assert worst_move > 0.05, (
        f"a replicate's rejuvenation kernel barely moved (move_size={worst_move:.3g}), "
        "so this coverage number is measuring resampling noise"
    )


def test_a_nominal_50_percent_interval_does_not_cover_far_more():
    """The assertion a too-wide posterior fails, which a floor never would.

    At the 50% level a posterior twice as wide as it should be covers about 80%
    of the time. Measured here: 68 of 120, or 57%, against a ceiling of 84.
    """
    counts, worst_move = coverage(0.5)
    total = int(counts.sum())

    assert total <= 84, (
        f"{total}/120 nominal 50% intervals covered the truth "
        f"({counts.tolist()} per coordinate); the posterior is too wide, and a "
        "coverage floor at 95% would have passed it"
    )
    assert total >= 48, (
        f"only {total}/120 nominal 50% intervals covered the truth; too tight in "
        "the other direction"
    )
    assert worst_move > 0.05


def test_no_single_coordinate_is_badly_calibrated():
    """The aggregate can hide one coordinate that is wrong in both directions.

    `beta` covered 40 of 40 at the 95% level and 26 of 40 at the 50% level, which
    is the widest of the three -- the decay rate is the coordinate 500 events
    constrain least, and the interval is honest about it rather than tight and
    wrong.
    """
    counts, _ = coverage(0.95)
    assert int(counts.min()) >= 30, (
        f"a coordinate covered {int(counts.min())}/40 nominal 95% intervals: {counts.tolist()}"
    )
