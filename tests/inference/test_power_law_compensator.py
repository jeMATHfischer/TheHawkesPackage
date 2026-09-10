"""The compensator of a power-law kernel, against its closed form.

The scope note for the kernel work predicted this and it was right: **a heavy
tail is exactly the shape a fixed panelled Gauss-Legendre rule under-integrates.**
The panels are placed at the events, where the kernel's mass is concentrated in a
core narrower than the panel, and the rule misses part of it.

The direction is what makes it worth a file of its own. The error is
systematically *negative* -- the compensator comes out **too small** -- and a
compensator too small is the inference-side twin of a thinning bound too small:
every unit of `∫λ` that goes missing is a penalty on a high intensity that never
gets applied, so `mu` and the excitation both come back too large and the fit
looks converged.

Nothing here compares one quadrature against another. The Omori-Utsu compensator
has a closed form,

    ∫_0^T λ = mu·T + Σ_i alpha [ c^(1-p) - (T - t_i + c)^(1-p) ] / (p - 1),

so both rules are checked against the answer rather than against each other.
"""

import warnings

import numpy as np
import pytest

from hawkes_package.inference import (
    ExponentialKernel,
    History,
    OmoriUtsuKernel,
    TemporalLogLikelihood,
    monotone_model,
)


def omori_history(c, p=1.8, mu=0.8, ratio=0.68, n_events=300, seed=0):
    """Simulate at a fixed branching ratio, so only the core width changes."""
    alpha = ratio * (p - 1) * c ** (p - 1)
    theta = np.array([mu, alpha, c, p])
    model = monotone_model(kernel=OmoriUtsuKernel())
    process = model(theta, rng=seed)
    process.simulate(n_events)
    events = process.events
    history = History.from_events(events, end=float(events[-1]))
    return model, theta, history


def closed_form(theta, events, times):
    """The exact compensator of ``mu + Σ alpha (t - t_i + c)^-p``."""
    mu, alpha, c, p = (float(v) for v in theta)
    out = []
    for t in np.asarray(times, dtype=float):
        past = events[events < t]
        tail = (c ** (1 - p) - (t - past + c) ** (1 - p)) / (p - 1)
        out.append(mu * t + alpha * float(np.sum(tail)))
    return np.array(out)


def worst_error(likelihood, theta, history, events):
    """Return the worst *signed* relative error against the closed form."""
    grid = history.times[5::20]
    truth = closed_form(theta, events, grid)
    got = likelihood.compensator(theta, history, grid)
    signed = (got - truth) / truth
    return float(signed[np.argmax(np.abs(signed))])


@pytest.mark.statistical
def test_the_power_law_compensator_is_too_small_at_the_old_default():
    """The measurement the family's `quadrature_order` exists for.

    At the package default of 8 nodes per panel the compensator is 2.6e-03 too
    small on a core of 0.2, and every point of it is *low* rather than scattered
    -- so the bias is in the direction that inflates a fitted excitation.
    """
    model, theta, history = omori_history(0.2)
    events = history.times
    at_default = TemporalLogLikelihood(model, order=8, check=False)
    error = worst_error(at_default, theta, history, events)

    assert error < 0.0, "the error should be an under-integration, not noise"
    assert abs(error) > 1e-3, f"expected roughly 2.6e-03, measured {abs(error):.1e}"


@pytest.mark.statistical
def test_the_family_asks_for_enough_nodes_to_fix_it():
    """16 nodes per panel, and the likelihood reads that from the family.

    Measured on the same history: 2.6e-03 at 8, 1.0e-05 at 16 -- a hundredfold,
    for twice the integrand evaluations. The threshold below is 1e-4, between
    the two.
    """
    model, theta, history = omori_history(0.2)
    likelihood = TemporalLogLikelihood(model, check=False)
    assert likelihood.order == 16, "the order should come from the kernel family"
    assert abs(worst_error(likelihood, theta, history, history.times)) < 1e-4


def test_an_exponential_kernel_keeps_the_package_default():
    """The order is per family, because the exponential shape needs nothing.

    Its compensator is exact to 7e-13 at 8 nodes, so raising the default for
    everyone would be paying twice over for the one shape that does not need it.
    """
    assert TemporalLogLikelihood(monotone_model(kernel=ExponentialKernel())).order == 8
    assert not hasattr(ExponentialKernel(), "quadrature_order")


def test_an_explicit_order_still_wins():
    """The family recommends; the caller decides."""
    model = monotone_model(kernel=OmoriUtsuKernel())
    assert TemporalLogLikelihood(model, order=32).order == 32
    assert TemporalLogLikelihood(model, order=4).order == 4


@pytest.mark.statistical
def test_the_resolution_check_catches_what_the_order_cannot():
    """A core narrow enough is a *panel* problem, and no order fixes it.

    The order-`P`-versus-`2P` check measures the whole-window integral, so at a
    core of 0.05 it sees 6.7e-03 at the old default and fires, and 2.1e-04 at
    the family's 16 and does not. At a core of 0.01 it sees **1.3e-02 even at
    16** and fires -- the peak is then a quarter of a panel wide and only more
    panels would help, which is what the warning says.
    """
    model, theta, history = omori_history(0.05)
    with pytest.warns(UserWarning, match="is not resolved by an order"):
        TemporalLogLikelihood(model, order=8, check=True).total(theta, history)

    model, theta, history = omori_history(0.01)
    with pytest.warns(UserWarning, match="is not resolved by an order"):
        TemporalLogLikelihood(model, check=True).total(theta, history)


@pytest.mark.statistical
def test_a_kernel_the_rule_resolves_easily_does_not_warn():
    """The negative control: the check must not fire on every power law.

    At a core of 0.5 the whole-window disagreement is 2.4e-06 -- three orders
    inside the tolerance -- and at 0.2 it is 5.4e-07 at the family's order. A
    check that warned here would be one nobody reads.
    """
    for core in (0.5, 0.2):
        model, theta, history = omori_history(core)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TemporalLogLikelihood(model, check=True).total(theta, history)
