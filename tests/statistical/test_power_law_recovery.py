"""Fitting a power-law kernel back, and what the data actually pins.

The evidence a new kernel family owes is recovery: simulate at known
parameters, refit, and see whether the intervals cover. For `OmoriUtsuKernel`
that check comes with a caveat worth more than the check itself.

**The three kernel parameters are only weakly identified on their own.** A
smaller `c` with a larger `alpha` and a larger `p` describes nearly the same
decay over the range a finite catalogue observes, so the posterior wanders a
ridge: over eight seeds the posterior mean of `p` ranged from 1.52 to 2.74
against a truth of 1.8, and `c` from 0.16 to 0.38 against 0.2. Read
individually, those look like a fit that failed.

What the data pins is the **branching ratio**, `alpha c^(1-p) / (p - 1)`, which
is the quantity the three of them exist to produce: 0.643 on average against a
truth of 0.679, over the same eight seeds, range [0.508, 0.776]. That is the
number to report from an Omori fit, and it is why this file asserts on it rather
than on `p`.

Each fit is about 30 seconds, so this runs in the coverage job rather than in
the ten-job matrix.
"""

import numpy as np
import pytest

from hawkes_package.inference import (
    ConstrainedPrior,
    History,
    IndependentPrior,
    LogNormal,
    OmoriUtsuKernel,
    TemporalLogLikelihood,
    Uniform,
    fit_smc,
    monotone_model,
)

pytestmark = [pytest.mark.statistical, pytest.mark.slow]

#: (mu, alpha, c, p). Branching ratio 0.679 with the default phi(x) = x.
TRUTH = np.array([0.8, 0.15, 0.2, 1.8])

SIZE = 250
N_PARTICLES = 96
BLOCKS = 4


def omori_model():
    return monotone_model(kernel=OmoriUtsuKernel())


def fit(seed):
    """Simulate `SIZE` events from the power law and fit them back."""
    model = omori_model()
    process = model(TRUTH, rng=seed)
    process.simulate(SIZE)
    history = History.from_events(process.events, end=float(process.events[-1]))
    prior = ConstrainedPrior(
        IndependentPrior(
            (
                LogNormal(-0.5, 0.8),  # mu
                LogNormal(-2.0, 1.0),  # alpha
                LogNormal(-1.6, 0.8),  # c
                # Uniform rather than log-normal: `p` is bounded below by 1 and
                # the interesting range is narrow, so a heavy-tailed prior on it
                # would spend most of its mass where the kernel is nearly flat.
                Uniform(1.05, 3.5),
            )
        ),
        model.support,
    )
    return fit_smc(
        # `check=False` because the resolution check is exercised directly in
        # `tests/inference/test_power_law_compensator.py`, and here it would fire
        # once per fit from inside a loop that cannot act on it.
        TemporalLogLikelihood(model, check=False),
        prior,
        history,
        blocks=BLOCKS,
        n_particles=N_PARTICLES,
        rng=seed,
    )


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_the_posterior_covers_the_power_law_parameters(seed):
    """Four parameters, 90% marginals, and a floor rather than "all four".

    Measured over seeds 0-7: 4 of 4 covered on five seeds and 3 of 4 on three,
    so the minimum observed is 3 and the threshold sits one below it. The
    coordinate that misses is whichever end of the (alpha, c, p) ridge the run
    settled on.
    """
    smc = fit(seed)
    cloud = smc.cloud
    lower, upper = cloud.quantile(0.05), cloud.quantile(0.95)
    covered = int(np.sum((lower <= TRUTH) & (upper >= TRUTH)))

    assert covered >= 2, (
        f"only {covered}/4 truths inside their 90% intervals; "
        f"means {np.round(cloud.mean(), 3).tolist()}"
    )
    # Seeds 0-7: minimum 0.586, so this has an order of magnitude of margin.
    # A collapsed cloud reports a tight posterior centred on resampling noise
    # with a perfect ESS and an acceptance rate of 1.000; only the distance the
    # particles travelled tells the two apart.
    assert smc.diagnostics.min_move_size > 0.05, (
        f"the rejuvenation kernel barely moved (min_move_size="
        f"{smc.diagnostics.min_move_size:.3g}); the posterior is resampling noise"
    )


@pytest.mark.parametrize("seed", [0, 2])
def test_the_branching_ratio_is_what_the_data_pins(seed):
    """The quantity the three kernel parameters jointly identify.

    `alpha`, `c` and `p` trade off along a ridge -- a sharper core with a
    steeper tail describes nearly the same decay over a finite window -- so any
    one of them can come back well off the truth while the process the posterior
    describes is right. The branching ratio is the invariant along that ridge,
    and over seeds 0-7 it sat in [0.508, 0.776] against a truth of 0.679.
    """
    model = omori_model()
    truth_ratio = float(model.branching_ratio(TRUTH))
    posterior = float(np.mean(model.branching_ratio(fit(seed).cloud.theta)))

    assert truth_ratio == pytest.approx(0.679, abs=1e-3), "the truth itself moved"
    assert abs(posterior - truth_ratio) < 0.25, (
        f"the posterior branching ratio is {posterior:.3f} against a truth of {truth_ratio:.3f}"
    )
