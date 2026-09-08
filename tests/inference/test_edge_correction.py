"""The likelihood must correct exactly as the simulator does.

This is the half of the edge correction that can go wrong silently. If the
simulator renormalises and the likelihood does not, the fit is biased by exactly
the correction — and everything about it looks healthy, because the two halves
disagree about the model rather than about the arithmetic.

So `SpatialComponents` records the policy and the likelihood reads it from
there, rather than each deciding for itself.
"""

import warnings

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import (
    History,
    SpatioTemporalLogLikelihood,
    spatio_temporal_model,
)

#: (mu, alpha, beta, sigma)
TRUTH = np.array([0.5, 0.9, 2.0, 1.0])
DOMAIN = hp.Rectangle(4.0, 3.0)


@pytest.fixture
def corrected():
    return spatio_temporal_model(DOMAIN)


@pytest.fixture
def plain():
    return spatio_temporal_model(DOMAIN, edge_correction="none")


@pytest.fixture
def history(corrected):
    process = corrected(TRUTH, rng=1)
    process.simulate(14)
    return History.from_simulation(process)


def test_the_model_records_the_policy_the_process_will_use(corrected, plain):
    """One decision, made once, read by both halves."""
    assert corrected.components.renormalises is True
    assert plain.components.renormalises is False
    assert spatio_temporal_model(hp.Circle()).components.renormalises is False

    assert corrected(TRUTH, rng=0)._renormalise is True
    assert plain(TRUTH, rng=0)._renormalise is False


def test_the_cached_backend_agrees_with_the_hooks(corrected, history):
    """The hooks are the normative definition; the cached form is a rearrangement.

    On a bounded domain that rearrangement now carries a per-event divisor, so
    this is the check that the division landed on the right axis. Dividing by
    the receiving event rather than the source would still produce a number, and
    a plausible one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hooks = SpatioTemporalLogLikelihood(corrected, backend="hooks").total(TRUTH, history)
        cached = SpatioTemporalLogLikelihood(corrected, backend="cached", homogeneous=False).total(
            TRUTH, history
        )

    assert cached == pytest.approx(hooks, rel=1e-10)


def test_the_uncorrected_backends_also_agree(plain, history):
    """The 'none' path must stay exactly what it was before the correction existed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        hooks = SpatioTemporalLogLikelihood(plain, backend="hooks").total(TRUTH, history)
        cached = SpatioTemporalLogLikelihood(plain, backend="cached", homogeneous=False).total(
            TRUTH, history
        )
    assert cached == pytest.approx(hooks, rel=1e-10)


def test_a_closed_surface_takes_the_identical_path(history):
    """`auto` must change nothing where there is no boundary.

    Bit-identical, not close: the same code runs.
    """
    circle = spatio_temporal_model(hp.Circle())
    process = circle(TRUTH, rng=2)
    process.simulate(12)
    observed = History.from_simulation(process)

    auto = SpatioTemporalLogLikelihood(circle, backend="cached").total(TRUTH, observed)
    explicit = SpatioTemporalLogLikelihood(
        spatio_temporal_model(hp.Circle(), edge_correction="none"), backend="cached"
    ).total(TRUTH, observed)
    assert auto == explicit


def test_the_spread_warning_blames_geometry_on_a_bounded_domain(plain, history):
    """The advice has to be right, not merely present.

    Without the correction the per-event masses genuinely differ by ~40% on this
    domain, and the pre-0.7.0 wording told the reader to raise `n_quad` — which
    cannot help, because the spread is not quadrature error.
    """
    with pytest.warns(UserWarning, match="real geometry rather than quadrature error"):
        SpatioTemporalLogLikelihood(plain, backend="cached").total(TRUTH, history)


def test_the_spread_warning_still_blames_quadrature_on_a_closed_surface():
    """And the old advice must survive where it was correct."""
    model = spatio_temporal_model(hp.Circle())
    likelihood = SpatioTemporalLogLikelihood(model, backend="cached", rtol=1e-12)
    process = model(TRUTH, rng=3)
    process.simulate(10)

    with pytest.warns(UserWarning, match="raising n_quad is the other fix"):
        likelihood.total(TRUTH, History.from_simulation(process))


def test_the_compensator_masses_are_exactly_one_under_correction(corrected, history):
    """Not approximately: by construction.

    Every event's kernel integrates to one over the domain after the division,
    so the compensator's per-event masses are the constant 1 -- which is also
    why the `homogeneous` shortcut becomes trivially exact here.
    """
    likelihood = SpatioTemporalLogLikelihood(corrected, backend="cached", homogeneous=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, masses, _ = likelihood._cached_terms(TRUTH, history)

    np.testing.assert_array_equal(masses, np.ones(history.n_events))


def test_a_kernel_that_integrates_to_nothing_is_refused(corrected, history):
    """`S_i = 0` is a division by zero, and it must name both causes."""
    likelihood = SpatioTemporalLogLikelihood(corrected, backend="cached", homogeneous=False)
    dead = TRUTH.copy()
    dead[3] = 1e-8  # sigma far narrower than a quadrature panel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="cannot be renormalised"):
            likelihood.total(dead, history)


# ---------------------------------------------------------------------------
# The test that justifies the package
# ---------------------------------------------------------------------------

#: Excitation values the profile below maximises over.
ALPHA_GRID = np.linspace(0.3, 2.4, 43)


def profile_alpha(model, history):
    """Return the excitation that maximises the log-likelihood, holding the rest.

    A grid rather than an optimiser: this package holds SciPy to one call site,
    and a coarse argmax is enough to show a bias of this size.
    """
    likelihood = SpatioTemporalLogLikelihood(model, backend="cached", homogeneous=False)
    values = []
    for alpha in ALPHA_GRID:
        theta = TRUTH.copy()
        theta[1] = alpha
        values.append(likelihood.total(theta, history))
    return float(ALPHA_GRID[int(np.argmax(values))])


@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 7, 13])
def test_ignoring_the_edge_inflates_the_excitation(seed, corrected, plain):
    """Simulate with the correction, then fit with and without it.

    This is the whole argument for the package in one assertion. The data have
    every event producing the same expected number of offspring wherever it
    sits. The uncorrected model expects an event near the boundary to produce
    fewer -- its kernel integrates to ``S_i < 1`` there -- so it must raise the
    excitation to explain the offspring it sees, and it does.

    Measured over seeds 0-20 at 40 events: the corrected profile has mean 0.900
    against a truth of 0.900, the uncorrected one has mean 1.474 -- a 64%
    over-estimate -- and the uncorrected exceeds the corrected on **21 of 21
    seeds**, with a minimum ratio of 1.500. The threshold below is 1.2, well
    inside that.

    The corrected half is what stops this being a test of nothing: without it,
    a model that simply preferred larger alpha would pass.
    """
    process = corrected(TRUTH, rng=seed)
    process.simulate(40)
    history = History.from_simulation(process)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with_correction = profile_alpha(corrected, history)
        without = profile_alpha(plain, history)

    assert without > 1.2 * with_correction, (
        f"omitting the edge correction should inflate the excitation, but the two "
        f"profiles are {without:.3f} against {with_correction:.3f}"
    )
    # And the corrected one has to be in the right neighbourhood, or the ratio
    # above could be produced by both being wrong together.
    assert 0.3 < with_correction < 1.8, (
        f"the corrected profile {with_correction:.3f} is nowhere near the truth 0.900"
    )
