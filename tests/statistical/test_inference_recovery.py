"""Does the posterior actually find the truth, and does it know how sure it is.

Every other test in the inference suite checks that a piece computes what it
says it computes. These check the only thing a user cares about: that fitting
simulated data with known parameters recovers them, with intervals that cover.

Seeds are fixed and thresholds are loose, as ``CONTRIBUTING.md`` requires --
these answer "is this broken", not "is this publication-grade". Each was swept
over seeds 0-20 locally before being pinned.

**Two guards on the guards.** A recovery battery can pass while measuring
nothing: if the credible interval were computed wrongly it might cover for any
parameter at all, and if the sampler never moved the posterior would be the
prior with a posterior's name on it. So one test requires the interval to
*exclude* a truth fitted under a deliberately broken compensator, and another
requires a frozen rejuvenation kernel to leave the posterior indistinguishable
from the prior.
"""

import itertools

import numpy as np
import pytest
from scipy import stats

import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior,
    ExponentialLogLikelihood,
    History,
    IndependentPrior,
    LikelihoodState,
    LogNormal,
    RandomWalkDrift,
    SMCSampler,
    SpatioTemporalLogLikelihood,
    Static,
    batch_posterior,
    exponential_model,
    fit_smc,
    ks_exponential,
    residuals,
    spatio_temporal_model,
)

pytestmark = pytest.mark.statistical

TRUTH = np.array([2.0, 0.5, 1.0])


def exponential_prior(model):
    """Vague but proper, and truncated to the parameters the process exists at."""
    base = IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0)))
    return ConstrainedPrior(base, model.support)


def simulate(n_events, seed, param=TRUTH):
    process = hp.ExponentialHawkes(param, rng=seed)
    process.simulate(n_events)
    return History.from_simulation(process)


def covers(cloud, truth, level=0.9):
    low, high = cloud.credible_interval(level)
    return (np.asarray(truth) > low) & (np.asarray(truth) < high)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [3, 11, 29])
def test_ibis_recovers_the_exponential_parameters(seed):
    """The truth inside every 90% marginal interval, and the mean within 25%."""
    model = exponential_model()
    history = simulate(800, seed)
    smc = fit_smc(
        ExponentialLogLikelihood(model),
        exponential_prior(model),
        history,
        blocks=8,
        n_particles=256,
        rng=seed,
    )
    cloud = smc.cloud
    assert np.all(covers(cloud, TRUTH)), (
        f"seed {seed}: 90% intervals {cloud.credible_interval(0.9).tolist()} do not all "
        f"cover {TRUTH.tolist()}\n{cloud.summary()}"
    )
    relative = np.abs(cloud.mean() - TRUTH) / TRUTH
    assert np.all(relative < 0.25), f"seed {seed}: relative error {relative.tolist()}"


@pytest.mark.slow
def test_ibis_agrees_with_an_independent_metropolis_run():
    """Two samplers, no shared machinery beyond the likelihood, one posterior.

    This is the test the sequential sampler cannot fake: a resample-move bug
    that produced a plausible but wrong cloud would have to reproduce, by
    coincidence, what a single long chain over the whole history finds.
    """
    model = exponential_model()
    prior = exponential_prior(model)
    likelihood = ExponentialLogLikelihood(model)
    # Seed swept over 0-20: the worst mean discrepancy across the sweep is 0.35
    # standard errors and the worst spread discrepancy 0.25, both dominated by
    # the Monte Carlo error of the two estimates rather than by any difference
    # between them. This seed sits comfortably inside the thresholds below.
    history = simulate(600, 0)

    smc = fit_smc(likelihood, prior, history, blocks=6, n_particles=512, rng=0)
    chain = batch_posterior(likelihood, prior, history, n_samples=40_000, burn_in=10_000, rng=100)

    smc_mean, smc_sd = smc.cloud.mean(), smc.cloud.std()
    chain_mean, chain_sd = chain.samples.mean(axis=0), chain.samples.std(axis=0)

    assert np.all(np.abs(smc_mean - chain_mean) < 0.15 * chain_sd), (
        f"means differ: SMC {smc_mean.tolist()} vs chain {chain_mean.tolist()}, "
        f"chain sd {chain_sd.tolist()}"
    )
    assert np.all(np.abs(smc_sd - chain_sd) / chain_sd < 0.35), (
        f"spreads differ: SMC {smc_sd.tolist()} vs chain {chain_sd.tolist()}"
    )
    # The alpha marginal, compared as whole distributions rather than by moments.
    alpha_index = model.spec.index("alpha")
    resampled, _ = smc.cloud.resample(np.random.default_rng(0))
    two_sample = stats.ks_2samp(resampled.theta[:, alpha_index], chain.samples[::40, alpha_index])
    assert two_sample.pvalue > 1e-3, f"alpha marginals differ, p={two_sample.pvalue:.3g}"


@pytest.mark.slow
def test_online_updates_converge_to_the_batch_posterior():
    """Eight blocks of a hundred must reach where one block of eight hundred is.

    The statement that the online entry point is the same estimator as the batch
    one, not merely a cheaper thing that looks similar.
    """
    model = exponential_model()
    prior = exponential_prior(model)
    history = simulate(800, 7)

    online = fit_smc(
        ExponentialLogLikelihood(model), prior, history, blocks=8, n_particles=512, rng=4
    )
    batch = fit_smc(
        ExponentialLogLikelihood(model), prior, history, blocks=1, n_particles=512, rng=4
    )
    reference = batch.cloud.std()
    difference = np.abs(online.cloud.mean() - batch.cloud.mean())
    assert np.all(difference < 0.1 * reference), (
        f"online {online.cloud.mean().tolist()} vs batch {batch.cloud.mean().tolist()}, "
        f"batch sd {reference.tolist()}"
    )


@pytest.mark.slow
def test_the_posterior_contracts_with_more_data():
    """Four times the data must narrow the posterior, by roughly the square root.

    A posterior that does not contract is not learning; one that contracts too
    fast is over-confident, which is what a degenerate cloud looks like from
    outside.
    """
    model = exponential_model()
    prior = exponential_prior(model)
    alpha_index = model.spec.index("alpha")

    spreads = []
    for n_events in (200, 800, 3200):
        history = simulate(n_events, 19)
        smc = fit_smc(
            ExponentialLogLikelihood(model),
            prior,
            history,
            blocks=max(1, n_events // 100),
            n_particles=256,
            rng=6,
        )
        spreads.append(float(smc.cloud.std()[alpha_index]))

    assert spreads[0] > spreads[1] > spreads[2], f"sd of alpha did not fall: {spreads}"
    for coarse, fine in itertools.pairwise(spreads):
        ratio = coarse / fine
        assert 1.4 <= ratio <= 3.0, f"contraction per 4x data was {ratio:.2f}: {spreads}"


@pytest.mark.slow
def test_time_rescaling_at_the_posterior_mean_is_consistent():
    """The fitted model must also *fit*, not merely be confidently located."""
    model = exponential_model()
    history = simulate(1500, 13)
    smc = fit_smc(
        ExponentialLogLikelihood(model),
        exponential_prior(model),
        history,
        blocks=6,
        n_particles=256,
        rng=8,
    )
    gaps = residuals(ExponentialLogLikelihood(model), smc.cloud.mean(), history)
    assert ks_exponential(gaps).pvalue > 1e-3


@pytest.mark.parametrize("seed", [0, 9, 17])
def test_the_cloud_recovers_from_every_concentration(seed):
    """Resample-move is what keeps the cloud alive; this is the measurement of that.

    Not "the effective sample size stays high" -- it does not, and should not.
    Weights concentrate whenever a block is informative, and over eight blocks
    of a 400-event history the pre-resample minimum reaches 0.04 of the cloud
    across seeds 0-20. What has to hold is that a concentrated cloud is always
    rebuilt before the next block, and that the rebuild restores diversity
    rather than merely flattening the weights over duplicates.
    """
    model = exponential_model()
    history = simulate(400, seed + 2)
    smc = fit_smc(
        ExponentialLogLikelihood(model),
        exponential_prior(model),
        history,
        blocks=8,
        n_particles=256,
        rng=seed,
    )
    diagnostics = smc.diagnostics
    assert diagnostics.ess_recovered(), diagnostics.summary()
    assert diagnostics.warnings() == [], diagnostics.summary()

    # Swept over seeds 0-20 at this configuration: the worst distinct-ancestor
    # fraction is 0.16 and the worst rejuvenation displacement 0.49 of the
    # cloud's own width. The two together are the statement that resampling took
    # the diversity and the move put it back.
    assert diagnostics.min_unique_fraction > 0.12, diagnostics.summary()
    assert diagnostics.min_move_size > 0.3, diagnostics.summary()

    accepted = [s.move_acceptance for s in diagnostics.steps if not np.isnan(s.move_acceptance)]
    assert accepted, "no block ever resampled, so the rejuvenation was never exercised"
    assert min(accepted) > 0.0


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [16, 18])
def test_a_drifting_filter_follows_a_regime_change_and_a_static_fit_does_not(seed):
    """A step in the background rate, and the contrast that gives it meaning.

    Tracking on its own proves little: a sampler that merely followed the most
    recent block would also "track". What the contrast shows is where the
    tracking comes from. The static fit is not simply slower -- over seeds 0-20
    its estimate of ``mu`` moves the *wrong way* on every one of them, because a
    stationary Hawkes process explains a burst as a long excitation cascade from
    a low background rather than as a higher background. The drifting filter
    rises on every seed. Worst rise 0.08, worst gap between the two 0.54.

    The regime change is built with
    :meth:`~hawkes_package.base.HawkesProcess.simulate_until` so the two halves
    carry equal *observation time* rather than equal event counts -- which a
    fixed-count simulation could not express, and which matters because the
    second regime produces events five times faster.

    :class:`~hawkes_package.inference.evolution.RandomWalkDrift`, not
    :class:`~hawkes_package.inference.evolution.LiuWest`. Liu-West preserves the
    cloud's variance exactly, which is what makes it well behaved on a parameter
    that is merely uncertain -- and what makes it unable to follow a jump once
    the cloud has contracted, because its jitter is proportional to a variance
    that has already gone small. An absolute jitter can re-expand; a
    variance-preserving one cannot. Measured on this configuration: Liu-West's
    second-half mean moved by -0.44 to +0.07 across seeds, against +0.08 to
    +2.31 for the random walk.
    """
    model = exponential_model()
    prior = ConstrainedPrior(
        IndependentPrior((LogNormal(0.8, 1.0), LogNormal(-2.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )
    likelihood = ExponentialLogLikelihood(model)
    switch, end, n_blocks = 40.0, 80.0, 40
    mu_index = model.spec.index("mu")

    quiet = hp.ExponentialHawkes(np.array([1.0, 0.05, 2.0]), rng=20 + seed)
    quiet.simulate_until(switch)
    busy = hp.ExponentialHawkes(np.array([5.0, 0.05, 2.0]), rng=40 + seed)
    busy.events = quiet.events
    busy.simulate_until(end, start=switch)
    history = History.from_events(busy.events, start=0.0, end=end)

    def halves(evolution, n_move):
        smc = SMCSampler(
            likelihood,
            prior,
            n_particles=256,
            evolution=evolution,
            n_move=n_move,
            rng=3,
        )
        smc.initialise(start=history.start)
        early, late = [], []
        for upto in np.linspace(0.0, end, n_blocks + 1)[1:]:
            smc.update(history, float(upto))
            (early if upto <= switch else late).append(float(smc.cloud.mean()[mu_index]))
        return float(np.mean(early)), float(np.mean(late))

    drift_early, drift_late = halves(RandomWalkDrift(0.15), 0)
    static_early, static_late = halves(Static(), 3)
    drift_rise = drift_late - drift_early
    static_rise = static_late - static_early

    assert drift_rise > 0.05, (
        f"the drifting filter did not follow the change in mu: {drift_early:.3f} -> "
        f"{drift_late:.3f}"
    )
    assert static_rise < 0.0, (
        f"the static fit rose too, so the contrast measures nothing: "
        f"{static_early:.3f} -> {static_late:.3f}"
    )
    assert drift_rise - static_rise > 0.3, (
        f"drift rose {drift_rise:+.3f} and static {static_rise:+.3f}: too close to "
        "distinguish the two"
    )


# ---------------------------------------------------------------------------
# Spatio-temporal
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_spatio_temporal_recovery_on_a_circle():
    """The same battery, one dimension of space added.

    Distributional only: ``CONTRIBUTING.md`` reserves exact reproducibility for
    the temporal classes, because the spatio-temporal simulator branches on
    floating-point comparisons whose last bits move with the BLAS build.
    """
    truth = np.array([0.5, 0.6, 1.5, 0.5])
    model = spatio_temporal_model(hp.Circle(), n_quad=64)
    process = model(truth, rng=31)
    # Sixty events, not the several hundred the temporal tests use, and the
    # limit is the *simulator* rather than the fit: 60 spatio-temporal events
    # cost about 65 s to generate -- a Metropolis-Hastings run per event over a
    # spatial intensity that is itself O(n) -- against 3 s for the whole fit,
    # whose full log-likelihood is 2.8 ms. The plan's 150 events would be five
    # minutes of simulation to buy a two-second fit.
    process.simulate(60)
    history = History.from_simulation(process)

    likelihood = SpatioTemporalLogLikelihood(model)
    prior = ConstrainedPrior(
        IndependentPrior(
            (
                LogNormal(-0.7, 0.8),
                LogNormal(-0.5, 0.8),
                LogNormal(0.4, 0.8),
                LogNormal(-0.7, 0.6),
            )
        ),
        model.support,
    )
    smc = fit_smc(likelihood, prior, history, blocks=6, n_particles=128, rng=5)

    assert likelihood.backend_used == "cached", "the fit fell back to the hooks"
    assert np.all(covers(smc.cloud, truth)), (
        f"90% intervals do not all cover {truth.tolist()}\n{smc.cloud.summary()}"
    )


# ---------------------------------------------------------------------------
# Guard the guards
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_broken_compensator_biases_the_fit_and_the_residuals_say_so():
    """Guard the guard: fit with a 20%-too-small compensator and both checks must fire.

    Without it, an interval computation that always covered and a residual test
    that never rejected would let the whole battery above pass vacuously. The
    defect chosen is the one this subpackage is built around -- a compensator
    that under-counts, subtracting too little from the penalty on a high
    intensity, so the fit comes back with more background and more excitation
    than the data supports and looks converged doing it.

    The residuals are computed with the **honest** compensator at the biased
    posterior mean, which is what a user would see. Computing them with the
    broken one instead cannot detect anything: the fit inflates the intensity by
    exactly the factor the compensator was shrunk by, so the two errors cancel
    and the rescaled gaps come back looking perfect. That is worth knowing --
    a diagnostic that shares a bug with the estimator it checks is not a check.

    Swept over seeds 0-20: the residual test rejects on all 21 (worst
    ``p = 7.4e-4``) and the mean rescaled gap is 1.21 to 1.24 against a true 1.0
    -- the 25% inflation the missing fifth of the compensator buys.
    """
    model = exponential_model()
    history = simulate(800, 7)

    class Undercounting(ExponentialLogLikelihood):
        """Every compensator 20% too small; the log-sum untouched."""

        def extend(self, state, theta, history, upto):
            honest, increment = super().extend(state, theta, history, upto)
            spanned = super().compensator(theta, history, np.array([state.upto, float(upto)]))
            forgiven = 0.2 * float(spanned[1] - spanned[0])
            return (
                LikelihoodState(
                    upto=honest.upto,
                    n_events=honest.n_events,
                    log_lik=honest.log_lik + forgiven,
                    carry=honest.carry,
                ),
                increment + forgiven,
            )

    smc = fit_smc(
        Undercounting(model),
        exponential_prior(model),
        history,
        blocks=8,
        n_particles=256,
        rng=1,
    )
    gaps = residuals(ExponentialLogLikelihood(model), smc.cloud.mean(), history)

    assert ks_exponential(gaps).pvalue < 1e-3, (
        f"the residual test failed to reject a 20%-too-small compensator: "
        f"p={ks_exponential(gaps).pvalue:.3g}"
    )
    assert gaps.mean() > 1.15, (
        f"the rescaled gaps averaged {gaps.mean():.3f}: the bias is supposed to be "
        "upward, and a test that cannot see its direction cannot see its size"
    )
    assert not np.all(covers(smc.cloud, TRUTH)), (
        "a compensator 20% too small still covered the truth on every coordinate, so "
        f"the interval is not measuring anything\n{smc.cloud.summary()}"
    )


@pytest.mark.slow
def test_a_frozen_rejuvenation_kernel_leaves_the_prior_behind():
    """Guard the guard: a move that never moves must be visible.

    With the proposal scaled to nothing, resample-move degenerates to resample:
    the cloud loses diversity and never regains it, and the posterior is a
    resampled prior wearing a posterior's weights.

    Neither of the two obvious diagnostics catches it. The effective sample size
    cannot -- N copies of one particle with equal weights score a perfect ESS --
    and neither can the *acceptance rate*, which reads 1.000, because a proposal
    scaled to 1e-12 proposes the point it starts from and that is always
    accepted. Nor, for the same reason, does an exact distinct-particle count:
    the particles differ, in the twelfth decimal place. What catches it is the
    distance the cloud actually travelled, measured in units of its own width:
    5.7e-12 here against 0.35 for the same fit with a real proposal.
    """
    model = exponential_model()
    history = simulate(600, 3)
    smc = SMCSampler(
        ExponentialLogLikelihood(model),
        exponential_prior(model),
        n_particles=128,
        n_move=3,
        rng=2,
    )
    smc.initialise(start=history.start)
    smc._proposal_factor = lambda cloud: 1e-12 * np.eye(3)
    smc.run(history, blocks=8)

    healthy = fit_smc(
        ExponentialLogLikelihood(model),
        exponential_prior(model),
        history,
        blocks=8,
        n_particles=128,
        n_move=3,
        rng=2,
    )
    assert smc.diagnostics.min_move_size < 1e-6, (
        f"the frozen kernel moved the cloud by {smc.diagnostics.min_move_size:.3g} of its "
        "width, so this configuration is not frozen and the guard tests nothing"
    )
    assert healthy.diagnostics.min_move_size > 0.05, (
        f"the healthy kernel moved by only {healthy.diagnostics.min_move_size:.3g}: the "
        "contrast measures the wrong thing"
    )
    assert any("never moved" in note for note in smc.diagnostics.warnings()), (
        f"a frozen kernel produced no warning: {smc.diagnostics.summary()}"
    )
    assert healthy.diagnostics.warnings() == [], healthy.diagnostics.summary()
