"""Can a fit tell a daily cycle from self-excitation? The package should not ship without asking.

Both produce clustering. A model carrying both can split the clustering anywhere
along a ridge, and a fit that reports a plausible cycle *and* a plausible
excitation is not by itself evidence that either is real. So the check is the one
the scope document asks for: **switch each effect off in turn and see whether the
fit notices.**

Measured at 110 events on a unit `Circle`, period 3, two seeds per arm:

======================  ==========================  ==========================
data                    cycle amplitude, 90%        branching ratio, 90%
======================  ==========================  ==========================
cycle, no excitation    [1.82, 2.79] / [1.40, 2.54]  [0.14, 0.38] / [0.07, 0.25]
excitation, no cycle    [0.10, 0.57] / [0.07, 0.74]  [0.12, 0.54] / [0.38, 0.87]
======================  ==========================  ==========================

The amplitude separates cleanly -- the lowest cycle-driven interval starts at
1.40, above the highest excitation-driven one at 0.74 -- so a fit *can* tell them
apart at this size. Two things are worth reading off the table as well, because
neither is a failure and both would look like one:

* the cycle-driven **branching ratio does not reach zero**, and cannot: `alpha`
  lives on ``(0, inf)``. Its upper end, 0.25 to 0.38, is the residual confusion
  -- the part of a periodic pile-up that self-excitation can also explain;
* the fitted amplitude runs **above** the truth of 0.8. A schedule floored at
  zero has a smaller trough than the raw series, so the likelihood buys some of
  the observed contrast with amplitude it does not have to pay for.

Each arm simulates 110 events on a surface, which is around three minutes before
any fitting, so this is a slow test and runs on the master push.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    ConstantBase,
    ConstrainedPrior,
    ExponentialKernel,
    GaussianSpatial,
    History,
    IndependentPrior,
    LogNormal,
    Normal,
    PeriodicBase,
    SpatioTemporalLogLikelihood,
    fit_smc,
    spatio_temporal_model,
)

pytestmark = [pytest.mark.statistical, pytest.mark.slow]

PERIOD = 3.0
SIZE = 110
DOMAIN = hp.Circle()


def simulate(amplitude, alpha, seed):
    """Simulate with a cycle, with excitation, or with one of them switched off."""
    schedule = hp.PeriodicSchedule([amplitude], [0.0], period=PERIOD)
    process = hp.SpatioTemporalHawkesProcess(
        base=hp.PeriodicBackground(lambda _: 0.6, schedule),
        spatial=GaussianSpatial(1).build(np.array([0.6])),
        temporal=lambda s: alpha * np.exp(-1.5 * np.asarray(s, dtype=float)),
        domain=DOMAIN,
        monotone_temporal_kernel=True,
        rng=seed,
    )
    process.simulate(SIZE)
    return History.from_simulation(process)


def fit(history, seed):
    """Fit the full model -- cycle *and* excitation -- whichever generated the data."""
    model = spatio_temporal_model(
        DOMAIN,
        base=PeriodicBase(ConstantBase(), 1, PERIOD),
        temporal=ExponentialKernel(),
        spatial=GaussianSpatial(1),
        n_quad=64,
    )
    prior = ConstrainedPrior(
        IndependentPrior(
            (
                LogNormal(-0.7, 0.7),  # mu
                Normal(0.0, 0.5),  # cos_1
                Normal(0.0, 0.5),  # sin_1
                LogNormal(-1.5, 1.0),  # alpha
                LogNormal(0.4, 0.7),  # beta
                LogNormal(-0.7, 0.5),  # sigma
            )
        ),
        model.support,
    )
    smc = fit_smc(
        SpatioTemporalLogLikelihood(model),
        prior,
        history,
        blocks=4,
        n_particles=192,
        rng=seed,
    )
    return model, smc


def summarise(model, smc):
    """Return the 90% intervals for the cycle amplitude and the branching ratio."""
    theta = smc.cloud.theta
    amplitude = np.hypot(theta[:, 1], theta[:, 2])
    ratios = np.asarray(model.branching_ratio(theta), dtype=float)
    return np.quantile(amplitude, [0.05, 0.95]), np.quantile(ratios, [0.05, 0.95])


def test_a_cycle_with_no_excitation_is_read_as_a_cycle():
    """The amplitude is found; the excitation stays small but not zero.

    Not zero because it cannot be: `alpha` lives on the open positive line, so
    "no excitation" is a region the posterior approaches rather than a point it
    covers. The number that matters is how small it stays.
    """
    model, smc = fit(simulate(0.8, 1e-6, 0), 0)
    amplitude, ratio = summarise(model, smc)

    assert amplitude[0] > 1.0, f"the cycle was not detected: amplitude {amplitude}"
    assert ratio[1] < 0.6, f"a cycle was read as self-excitation: branching {ratio}"
    assert smc.diagnostics.min_move_size > 0.05


def test_excitation_with_no_cycle_is_not_read_as_a_cycle():
    """The mirror image, and the one that would embarrass the model if it failed.

    Self-excitation makes events arrive in bursts, and a burst folded onto a
    period looks like a cycle if the fit wants one badly enough.
    """
    model, smc = fit(simulate(0.0, 0.9, 0), 0)
    amplitude, ratio = summarise(model, smc)

    assert amplitude[1] < 1.0, f"excitation was read as a cycle: amplitude {amplitude}"
    assert ratio[1] > 0.2, f"the excitation was not detected: branching {ratio}"
    assert smc.diagnostics.min_move_size > 0.05


def test_the_two_regimes_do_not_overlap():
    """The contrast, which is what "can tell them apart" actually means.

    Coverage on its own proves nothing here: a posterior wide enough covers both
    truths and separates nothing. The cycle-driven amplitude interval must sit
    entirely above the excitation-driven one, and over two seeds per arm the gap
    was 1.40 against 0.74.
    """
    cycle_model, cycle_smc = fit(simulate(0.8, 1e-6, 1), 1)
    burst_model, burst_smc = fit(simulate(0.0, 0.9, 1), 1)

    cycle_amplitude, _ = summarise(cycle_model, cycle_smc)
    burst_amplitude, _ = summarise(burst_model, burst_smc)

    assert cycle_amplitude[0] > burst_amplitude[1], (
        f"the two regimes overlap: a real cycle gives {np.round(cycle_amplitude, 3)} "
        f"and pure excitation gives {np.round(burst_amplitude, 3)}, so the model "
        "cannot separate them at this sample size"
    )
