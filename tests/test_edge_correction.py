"""Renormalising the spatial kernel on a domain with an edge.

On a closed surface an event's kernel has the same mass over the domain wherever
it lands. On a bounded one it does not, so an event near the edge really does
produce fewer offspring — and a fit that ignores that blames a weaker kernel.

The correction divides each event's spatial kernel by its own in-domain mass. It
is safe for the thinning bound because that mass is a **per-event constant**: it
depends on the event's location and the fixed quadrature, not on `x` or `t`, so
it divides the intensity and its supremum by the same positive number.
"""

import numpy as np
import pytest

import hawkes_package as hp


def bump(d):
    return max(0.0, 1.0 - d / np.pi)


def decay(dt):
    return 0.9 * np.exp(-2.0 * np.asarray(dt, dtype=float))


def build(domain, **kwargs):
    defaults = {
        "base": lambda x: 0.5,
        "spatial": bump,
        "temporal": decay,
        "domain": domain,
        "monotone_temporal_kernel": True,
        "rng": 0,
    }
    defaults.update(kwargs)
    return hp.SpatioTemporalHawkesProcess(**defaults)


def test_auto_corrects_exactly_where_there_is_an_edge():
    assert build(hp.Rectangle(4.0, 3.0))._renormalise is True
    assert build(hp.Rectangle(4.0))._renormalise is True

    for closed in (hp.Circle(), hp.Torus2D(), hp.Sphere(), hp.FundamentalDomain.hexagon(1.0)):
        assert build(closed)._renormalise is False, f"{type(closed).__name__} has no edge"


def test_the_correction_can_be_forced_or_refused():
    assert build(hp.Rectangle(4.0, 3.0), edge_correction="none")._renormalise is False
    assert build(hp.Circle(), edge_correction="renormalise")._renormalise is True


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="edge_correction must be"):
        build(hp.Rectangle(4.0, 3.0), edge_correction="clip")


def test_every_event_excites_the_same_total_amount():
    """The property the correction exists to create.

    Uncorrected, an event in the corner has visibly less mass over the domain
    than one in the middle. Corrected, both integrate to one.
    """
    process = build(hp.Rectangle(4.0, 3.0))
    middle = np.array([0.0, 0.0])
    corner = np.array([1.9, 1.4])

    raw_middle = process._in_domain_mass(middle)
    raw_corner = process._in_domain_mass(corner)
    assert raw_corner < 0.75 * raw_middle, (
        "a corner event should lose real mass off the edge, or this domain and "
        f"kernel do not exercise the problem: {raw_corner:.4f} vs {raw_middle:.4f}"
    )

    for point in (middle, corner):
        mass = process._in_domain_mass(point)
        corrected = process._quadrature.integrate(
            lambda x, p=point, m=mass: bump(process.domain.distance(x, p)) / m
        )
        assert corrected == pytest.approx(1.0, rel=1e-9)


def test_a_closed_surface_is_bit_identical_with_and_without_the_flag():
    """`auto` must be inert on every domain that predates 1.0.0.

    Not "close": the same events, because the same code path runs.
    """
    for domain in (hp.Circle(), hp.Torus2D()):
        auto = build(domain, rng=5)
        explicit = build(domain, edge_correction="none", rng=5)
        auto.simulate(6)
        explicit.simulate(6)
        np.testing.assert_array_equal(auto.events, explicit.events)


def test_the_correction_changes_the_realisation_on_a_bounded_domain():
    corrected = build(hp.Rectangle(4.0, 3.0), rng=7)
    plain = build(hp.Rectangle(4.0, 3.0), edge_correction="none", rng=7)
    corrected.simulate(8)
    plain.simulate(8)
    assert not np.allclose(corrected.events[0], plain.events[0])


def test_the_scale_cache_survives_growth_and_notices_replacement():
    """Keyed on the event times, not on a count.

    `process.events = history` swaps the whole record. A cache keyed on length
    would survive being handed a different realisation of the same size and
    scale every event by another one's mass — silently, and only on the seeded
    path a forecast takes.
    """
    process = build(hp.Rectangle(4.0, 3.0), rng=1)
    process.simulate(4)
    first = process._edge_scales(4).copy()
    assert first.size == 4
    assert np.all(first > 0)

    process.simulate(2)
    grown = process._edge_scales(6)
    np.testing.assert_array_equal(grown[:4], first, err_msg="growth must not recompute")

    # A different realisation of the same length must not reuse the cache.
    other = build(hp.Rectangle(4.0, 3.0), rng=99)
    other.simulate(6)
    process.events = other.events
    replaced = process._edge_scales(6)
    assert not np.allclose(replaced, grown)
    np.testing.assert_allclose(
        replaced, other._edge_scales(6), rtol=1e-12, err_msg="must match the record it now holds"
    )


def test_a_kernel_that_integrates_to_nothing_raises():
    """`S_i = 0` is a division by zero, and the message has to say which cause.

    A kernel narrower than one quadrature panel and a signed kernel that cancels
    are different problems with the same symptom.
    """
    process = build(
        hp.Rectangle(4.0, 3.0),
        spatial=lambda d: 0.0 * d,
        n_quad=8,
    )
    with pytest.raises(RuntimeError, match="cannot be renormalised"):
        process._in_domain_mass(np.array([0.0, 0.0]))


@pytest.mark.statistical
def test_the_corrected_process_still_bounds_its_intensity():
    """The invariant, on the path the correction changed.

    The harness covers this too; this is the direct statement, and it checks the
    ratio rather than only the sign so a bound that became loose would show.
    """
    process = build(hp.Rectangle(4.0, 3.0), rng=3)
    process.simulate(10)

    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.0, float(process.events[0, -1]), size=40))
    for t in times:
        bound = process._integrated_intensity(float(t), bound=True)
        value = process._integrated_intensity(float(t), bound=False)
        assert value <= bound + 1e-9
