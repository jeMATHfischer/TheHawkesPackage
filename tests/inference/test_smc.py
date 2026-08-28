"""The particle cloud, the sampler's invariants, and its refusals.

Recovery -- does the posterior find the truth -- lives in
``tests/statistical/test_inference_recovery.py``. What is checked here is that
the loop is coherent: that the weights stay a distribution, that blocks compose,
that the diagnostics report what happened, and that the configurations which
produce plausible output aimed at nothing are refused rather than run.
"""

import math

import numpy as np
import pytest

from hawkes_package.inference import (
    ExponentialLogLikelihood,
    History,
    IndependentPrior,
    LiuWest,
    LogNormal,
    Parameter,
    ParameterSpec,
    ParticleCloud,
    RandomWalkDrift,
    SMCSampler,
    Static,
    fit_smc,
    multinomial,
)
from hawkes_package.inference.smc import block_boundaries

SPEC = ParameterSpec((Parameter("a"), Parameter("b")))


def flat_cloud(theta):
    theta = np.asarray(theta, dtype=float)
    return ParticleCloud(theta, np.full(theta.shape[0], -math.log(theta.shape[0])), SPEC)


# ---------------------------------------------------------------------------
# ParticleCloud
# ---------------------------------------------------------------------------


def test_a_cloud_refuses_unnormalised_weights():
    """An unnormalised cloud reports a healthy ESS while carrying no mass."""
    with pytest.raises(ValueError, match="not normalised"):
        ParticleCloud(np.ones((4, 2)), np.zeros(4), SPEC)


def test_a_cloud_refuses_a_width_the_spec_does_not_describe():
    with pytest.raises(ValueError, match="coordinate"):
        ParticleCloud(np.ones((4, 3)), np.full(4, -math.log(4)), SPEC)


def test_a_cloud_refuses_a_weight_per_particle_mismatch():
    with pytest.raises(ValueError, match="weight"):
        ParticleCloud(np.ones((4, 2)), np.full(3, -math.log(3)), SPEC)


def test_moments_match_the_weighted_definitions(rng):
    theta = rng.lognormal(0.0, 0.5, size=(500, 2))
    log_weights = np.log(rng.dirichlet(np.ones(500)))
    cloud = ParticleCloud(theta, log_weights, SPEC)
    weights = np.exp(log_weights)
    np.testing.assert_allclose(cloud.mean(), np.average(theta, axis=0, weights=weights))
    centred = theta - cloud.mean()
    expected = (centred * weights[:, None]).T @ centred
    np.testing.assert_allclose(cloud.covariance(), expected, rtol=1e-12)
    np.testing.assert_allclose(cloud.std(), np.sqrt(np.diag(expected)), rtol=1e-12)


def test_quantiles_of_an_equally_weighted_cloud_track_numpy(rng):
    theta = rng.normal(3.0, 1.0, size=(4000, 2)) + 5.0
    cloud = flat_cloud(theta)
    for q in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(cloud.quantile(q), np.quantile(theta, q, axis=0), rtol=0.02)


def test_a_credible_interval_brackets_and_widens_with_the_level(rng):
    cloud = flat_cloud(rng.lognormal(0.0, 0.4, size=(2000, 2)))
    narrow = cloud.credible_interval(0.5)
    wide = cloud.credible_interval(0.95)
    assert np.all(wide[0] < narrow[0])
    assert np.all(wide[1] > narrow[1])
    assert cloud.credible_interval(0.9).shape == (2, 2)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 2.0])
def test_a_credible_level_outside_the_unit_interval_is_refused(level):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        flat_cloud(np.ones((4, 2))).credible_interval(level)


def test_resampling_flattens_the_weights_and_keeps_the_heavy_particles(rng):
    theta = np.arange(20, dtype=float).reshape(10, 2)
    log_weights = np.log(np.array([0.5] + [0.5 / 9] * 9))
    cloud = ParticleCloud(theta, log_weights, SPEC)
    resampled, ancestors = cloud.resample(rng)
    assert resampled.n_particles == 10
    np.testing.assert_allclose(resampled.log_weights, -math.log(10))
    assert (ancestors == 0).sum() >= 5


def test_the_summary_names_every_parameter():
    text = flat_cloud(np.ones((6, 2))).summary()
    assert "a" in text
    assert "b" in text
    assert "90% interval" in text


# ---------------------------------------------------------------------------
# The sampler's refusals
# ---------------------------------------------------------------------------


@pytest.fixture
def sampler_parts(exp_model, exp_prior):
    return ExponentialLogLikelihood(exp_model), exp_prior


def test_drift_and_rejuvenation_together_are_refused(sampler_parts):
    """An invariant kernel targeting a posterior the model says does not exist.

    Exactly the configuration reached by tuning until the output looks
    plausible, which is why it is refused at construction rather than warned
    about at the end.
    """
    likelihood, prior = sampler_parts
    with pytest.raises(ValueError, match="does not exist"):
        SMCSampler(likelihood, prior, evolution=LiuWest(), n_move=3, rng=0)
    with pytest.raises(ValueError, match="does not exist"):
        SMCSampler(likelihood, prior, evolution=RandomWalkDrift(0.05), n_move=1, rng=0)
    # n_move=0 is the coherent pairing, and is accepted.
    SMCSampler(likelihood, prior, evolution=LiuWest(), n_move=0, rng=0)


def test_a_prior_of_the_wrong_width_is_refused(exp_model):
    """NumPy would broadcast the mismatch into a plausible-looking array."""
    likelihood = ExponentialLogLikelihood(exp_model)
    short = IndependentPrior((LogNormal(0.0, 1.0), LogNormal(0.0, 1.0)))
    with pytest.raises(ValueError, match="coordinate"):
        SMCSampler(likelihood, short, rng=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_particles": 1}, "at least 2"),
        ({"ess_threshold": 0.0}, r"\(0, 1\]"),
        ({"ess_threshold": 1.5}, r"\(0, 1\]"),
        ({"n_move": -1}, "non-negative"),
        ({"scale": 0.0}, "scale must be positive"),
        ({"on_invalid": "shrug"}, "on_invalid"),
    ],
)
def test_bad_configuration_is_refused(sampler_parts, kwargs, match):
    likelihood, prior = sampler_parts
    with pytest.raises(ValueError, match=match):
        SMCSampler(likelihood, prior, rng=0, **kwargs)


def test_a_prior_outside_the_support_is_refused_rather_than_filtered(exp_model):
    """The prior is the caller's statement about where the truth might be."""
    likelihood = ExponentialLogLikelihood(exp_model)
    # alpha centred well above beta: most draws are supercritical.
    prior = IndependentPrior((LogNormal(0.0, 0.1), LogNormal(2.0, 0.1), LogNormal(-2.0, 0.1)))
    smc = SMCSampler(likelihood, prior, n_particles=16, rng=0)
    with pytest.raises(ValueError, match="ConstrainedPrior"):
        smc.initialise()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_the_cloud_stays_a_distribution_at_every_block(history, sampler_parts):
    likelihood, prior = sampler_parts
    smc = SMCSampler(likelihood, prior, n_particles=48, rng=1)
    smc.initialise(start=history.start)
    for upto in (2.0, 5.0, 10.0):
        smc.update(history, upto)
        assert smc.cloud.log_weights.shape == (48,)
        assert math.isclose(np.exp(smc.cloud.log_weights).sum(), 1.0, rel_tol=1e-9)
        assert np.all(smc.model.support(smc.cloud.theta))
    assert len(smc.diagnostics.steps) == 3


def test_updating_twice_equals_running_one_block(history, exp_model, exp_prior):
    """The block increments telescope under a static parameter -- exactly.

    That is what makes IBIS the data-tempered sampler rather than an
    approximation to it, so it is asserted rather than assumed.
    """
    likelihood = ExponentialLogLikelihood(exp_model)
    split = SMCSampler(likelihood, exp_prior, n_particles=32, n_move=0, rng=3)
    split.initialise(start=history.start)
    split.update(history, 4.0)
    split.update(history, history.end)

    single = SMCSampler(
        ExponentialLogLikelihood(exp_model), exp_prior, n_particles=32, n_move=0, rng=3
    )
    single.initialise(start=history.start)
    single.update(history, history.end)

    np.testing.assert_allclose(split.cloud.theta, single.cloud.theta, rtol=0, atol=0)
    np.testing.assert_allclose(split.cloud.log_weights, single.cloud.log_weights, rtol=1e-10)
    assert split.diagnostics.log_evidence == pytest.approx(
        single.diagnostics.log_evidence, rel=1e-10
    )


def test_run_and_a_loop_of_updates_agree(history, exp_model, exp_prior):
    boundaries = block_boundaries(history, 3)
    looped = SMCSampler(ExponentialLogLikelihood(exp_model), exp_prior, n_particles=32, rng=5)
    looped.initialise(start=history.start)
    for upto in boundaries:
        looped.update(history, upto)

    run = SMCSampler(ExponentialLogLikelihood(exp_model), exp_prior, n_particles=32, rng=5)
    cloud = run.run(history, blocks=3)
    np.testing.assert_allclose(cloud.theta, looped.cloud.theta, rtol=0, atol=0)


def test_the_same_seed_reproduces_the_whole_fit(history, exp_model, exp_prior):
    """Assertable on the temporal path: no floating-point-branching simulation runs."""
    clouds = []
    for _ in range(2):
        smc = SMCSampler(ExponentialLogLikelihood(exp_model), exp_prior, n_particles=32, rng=9)
        clouds.append(smc.run(history, blocks=3))
    np.testing.assert_array_equal(clouds[0].theta, clouds[1].theta)
    np.testing.assert_array_equal(clouds[0].log_weights, clouds[1].log_weights)


def test_fit_smc_returns_the_sampler_with_its_diagnostics(history, exp_model, exp_prior):
    smc = fit_smc(
        ExponentialLogLikelihood(exp_model), exp_prior, history, blocks=2, n_particles=32, rng=0
    )
    assert isinstance(smc, SMCSampler)
    assert len(smc.diagnostics.steps) == 2
    assert smc.cloud is not None


def test_an_alternative_resampler_is_accepted(history, exp_model, exp_prior):
    smc = SMCSampler(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        n_particles=32,
        resampler=multinomial,
        rng=0,
    )
    cloud = smc.run(history, blocks=2)
    assert cloud.n_particles == 32


def test_a_drifting_evolution_runs_without_rejuvenation(history, exp_model, exp_prior):
    smc = SMCSampler(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        n_particles=32,
        evolution=LiuWest(0.98),
        n_move=0,
        rng=0,
    )
    cloud = smc.run(history, blocks=3)
    assert np.all(smc.model.support(cloud.theta))


def test_static_is_the_default_evolution(sampler_parts):
    likelihood, prior = sampler_parts
    assert isinstance(SMCSampler(likelihood, prior, rng=0).evolution, Static)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def test_block_boundaries_end_at_the_window(history):
    """The tail after the last event carries the information that nothing happened."""
    for blocks in (1, 2, 5):
        boundaries = block_boundaries(history, blocks)
        assert boundaries[-1] == history.end
        assert boundaries == sorted(boundaries)


def test_explicit_boundaries_are_used_and_completed(history):
    assert block_boundaries(history, [3.0, 6.0]) == [3.0, 6.0, history.end]
    assert block_boundaries(history, [3.0, history.end]) == [3.0, history.end]


@pytest.mark.parametrize(
    ("blocks", "match"),
    [
        (0, "at least 1"),
        ([3.0, 3.0], "strictly increasing"),
        ([3.0, 99.0], r"must lie in"),
        ([], "must not be empty"),
    ],
)
def test_bad_blocks_are_refused(history, blocks, match):
    with pytest.raises(ValueError, match=match):
        block_boundaries(history, blocks)


def test_a_history_with_no_events_still_has_a_block(exp_model):
    empty = History(np.array([]), None, 0.0, end=5.0)
    assert block_boundaries(empty, 4) == [5.0]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_the_diagnostics_record_every_block(history, exp_model, exp_prior):
    smc = fit_smc(
        ExponentialLogLikelihood(exp_model), exp_prior, history, blocks=4, n_particles=32, rng=2
    )
    steps = smc.diagnostics.steps
    assert len(steps) == 4
    assert [s.n_events for s in steps] == sorted(s.n_events for s in steps)
    assert steps[-1].n_events == history.n_events
    assert all(0.0 < s.ess <= 32.0 + 1e-9 for s in steps)
    assert 0.0 <= smc.diagnostics.min_ess_fraction <= 1.0
    assert np.isfinite(smc.diagnostics.log_evidence)
    assert "log evidence" in smc.diagnostics.summary()


def test_a_never_moving_kernel_is_reported(history, exp_model, exp_prior):
    """Guard the guard: a rejuvenation that never moves must be visible.

    Scaling the proposal down by 1e-12 leaves the cloud exactly as the resample
    left it. The posterior is then the resampled prior, and without this the
    whole recovery battery could pass on a sampler that never explores.
    """
    smc = SMCSampler(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        n_particles=32,
        n_move=2,
        scale=2.38,
        rng=4,
    )
    smc.initialise(start=history.start)
    # A proposal factor of zero: every candidate is the current point, so every
    # proposal is accepted and no particle moves. The acceptance rate cannot see
    # that; the distinct-particle count can.
    smc._proposal_factor = lambda cloud: np.zeros((3, 3))
    smc.run(history, blocks=4)
    before = smc.cloud.theta.copy()
    smc.update(history, history.end)
    np.testing.assert_array_equal(smc.cloud.theta, before)


def test_a_degenerate_run_says_so():
    """The warnings list is what stands between a collapsed cloud and a confident plot."""
    from hawkes_package.inference.smc import SMCDiagnostics, StepRecord

    def record(**kwargs):
        fields = {
            "upto": 1.0,
            "n_events": 10,
            "ess": 50.0,
            "resampled": True,
            "move_acceptance": 0.3,
            "move_size": 0.4,
            "unique_fraction": 0.9,
            "log_evidence_increment": -3.0,
            "n_invalid": 0,
        }
        fields.update(kwargs)
        return StepRecord(**fields)

    diagnostics = SMCDiagnostics(n_particles=100)
    # A cloud left concentrated: low ESS and *no* resample, which is the state
    # that carries a handful of particles into the next block.
    diagnostics.steps.append(record(ess=2.0, resampled=False))
    diagnostics.steps.append(record(unique_fraction=0.01))
    notes = diagnostics.warnings()
    assert any("effective sample size" in note for note in notes)
    assert any("distinct" in note for note in notes)
    assert "WARNING" in diagnostics.summary()

    # The frozen-kernel warning keys on *every* move having failed, so it needs
    # a run of its own: one healthy move is enough to say the kernel works.
    frozen = SMCDiagnostics(n_particles=100)
    frozen.steps.append(record(move_acceptance=0.0, move_size=0.0))
    assert any("never moved" in note for note in frozen.warnings())
    assert not any("never moved" in note for note in diagnostics.warnings())


def test_a_move_that_travels_nowhere_is_reported():
    """The acceptance rate cannot see it: a proposal of the current point is accepted."""
    from hawkes_package.inference.smc import SMCDiagnostics, StepRecord

    diagnostics = SMCDiagnostics(n_particles=100)
    diagnostics.steps.append(
        StepRecord(
            upto=1.0,
            n_events=10,
            ess=20.0,
            resampled=True,
            move_acceptance=1.0,
            move_size=1e-12,
            unique_fraction=0.8,
            log_evidence_increment=-3.0,
            n_invalid=0,
        )
    )
    notes = diagnostics.warnings()
    assert any("never moved" in note for note in notes)
    assert diagnostics.min_move_size == 1e-12


def test_a_low_ess_that_is_resampled_is_not_a_warning():
    """Weights concentrate whenever a block is informative; that is not a fault.

    The pre-resample minimum reaches 0.04 of the cloud on ordinary fits. Warning
    on it would fire on healthy runs, and a warning that fires on healthy runs
    is one nobody reads on the unhealthy ones.
    """
    from hawkes_package.inference.smc import SMCDiagnostics, StepRecord

    diagnostics = SMCDiagnostics(n_particles=100)
    diagnostics.steps.append(
        StepRecord(
            upto=1.0,
            n_events=10,
            ess=2.0,
            resampled=True,
            move_acceptance=0.3,
            move_size=0.4,
            unique_fraction=1.0,
            log_evidence_increment=-3.0,
            n_invalid=0,
        )
    )
    assert diagnostics.ess_recovered()
    assert diagnostics.warnings() == []
    assert diagnostics.min_ess_fraction == 0.02


def test_an_empty_diagnostics_object_reports_nothing_wrong():
    from hawkes_package.inference.smc import SMCDiagnostics

    diagnostics = SMCDiagnostics(n_particles=10)
    assert diagnostics.warnings() == []
    assert math.isnan(diagnostics.min_ess_fraction)
    assert diagnostics.log_evidence == 0.0


# ---------------------------------------------------------------------------
# What happens when a likelihood cannot be evaluated
# ---------------------------------------------------------------------------


class Unevaluable(ExponentialLogLikelihood):
    """Fails on the particles whose `mu` exceeds a threshold."""

    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def extend(self, state, theta, history, upto):
        if float(np.asarray(theta).ravel()[0]) > self.threshold:
            raise ArithmeticError("contrived failure")
        return super().extend(state, theta, history, upto)


def test_a_failing_likelihood_raises_by_default(history, exp_model, exp_prior):
    """Reading an exception as "zero posterior mass" is how a typo becomes a posterior.

    The parameter is inside the model's declared support, so the failure is a
    bug in the model rather than a statement about the data -- and the message
    says which particle and which theta, so it can be reproduced.
    """
    smc = SMCSampler(Unevaluable(exp_model, 1.0), exp_prior, n_particles=32, rng=0)
    smc.initialise(start=history.start)
    with pytest.raises(RuntimeError, match=r"failed at particle \d+"):
        smc.update(history, history.end)


def test_rejecting_a_failing_likelihood_counts_and_warns(history, exp_model, exp_prior):
    """Opt-in, and never silent: each rejected particle narrows the posterior."""
    smc = SMCSampler(
        Unevaluable(exp_model, 2.0), exp_prior, n_particles=32, on_invalid="reject", rng=0
    )
    smc.initialise(start=history.start)
    with pytest.warns(UserWarning, match="could not be evaluated"):
        smc.update(history, history.end)
    assert smc.diagnostics.steps[-1].n_invalid > 0
    assert math.isclose(np.exp(smc.cloud.log_weights).sum(), 1.0, rel_tol=1e-9)


def test_a_drifting_particle_that_leaves_the_support_is_held_where_it_was(
    history, exp_model, exp_prior
):
    """Discarding it would leave the cloud a sample of nothing in particular.

    A random walk on an unbounded scale can walk a particle past the
    stationarity boundary. Holding it keeps the cloud a valid sample of
    *something*; the reweighting that follows will move it back if the data
    disagrees.
    """
    smc = SMCSampler(
        ExponentialLogLikelihood(exp_model),
        exp_prior,
        n_particles=64,
        evolution=RandomWalkDrift(2.0),  # large enough to overshoot regularly
        n_move=0,
        rng=0,
    )
    smc.initialise(start=history.start)
    for upto in (3.0, 6.0, history.end):
        smc.update(history, upto)
        assert np.all(smc.model.support(smc.cloud.theta))
