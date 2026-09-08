"""The first domain here with a real boundary.

Every other domain in the package is a closed surface -- periodic, like `Circle`
and `Torus2D`, or a quotient, like `Sphere` and `FundamentalDomain`. `Rectangle`
is neither, and the contract battery in `test_domains.py` (which it joins
through the `domain` fixture) checks that it still satisfies everything a domain
must. This file checks the things that are true *because* it has an edge.
"""

import numpy as np
import pytest
from scipy import stats

import hawkes_package as hp


def test_it_declares_a_boundary_and_no_periodicity():
    """Both flags, and they are not each other's negation.

    `Sphere` and `FundamentalDomain` are also `periodic = False` and have no
    boundary at all -- the first is not a quotient, the second is glued by
    isometries that are not translations. A correction keyed off `periodic`
    would rescale their intensity for nothing, which is why `has_boundary` is
    its own flag.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    assert rectangle.has_boundary is True
    assert rectangle.periodic is False

    for closed in (hp.Sphere(), hp.FundamentalDomain.rectangle(4.0, 3.0), hp.Torus2D()):
        assert closed.has_boundary is False


def test_the_same_box_glued_up_is_a_different_domain():
    """`Rectangle(4, 3)` and `FundamentalDomain.rectangle(4, 3)` share a box only.

    The torus wraps, so its diameter is the half-diagonal and opposite edges are
    adjacent. The rectangle does not, so its diameter is the full diagonal and
    the corners are the farthest pair. Confusing the two is the easiest mistake
    available here, and it is silent: both have the same `volume` and `bounds`.
    """
    bounded = hp.Rectangle(4.0, 3.0)
    glued = hp.FundamentalDomain.rectangle(4.0, 3.0)

    assert bounded.volume == pytest.approx(glued.volume)
    np.testing.assert_allclose(bounded.bounds, glued.bounds)

    corner_a = np.array([-2.0, -1.5])
    corner_b = np.array([2.0, 1.5])
    assert bounded.distance(corner_a, corner_b) == pytest.approx(5.0)
    # On the torus the two corners are the *same point*.
    assert glued.distance(corner_a, corner_b) == pytest.approx(0.0, abs=1e-9)


def test_the_diameter_is_the_full_diagonal():
    """Not the half-diagonal the base class defaults to.

    That default is right for a domain with every axis periodic, where the
    farthest two points are half a period apart. Nothing wraps here.
    """
    assert hp.Rectangle(4.0, 3.0).max_distance == pytest.approx(5.0)
    assert hp.Rectangle(2.0).max_distance == pytest.approx(2.0)


def test_distance_is_plain_euclidean():
    rectangle = hp.Rectangle(4.0, 3.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rectangle.sample_uniform(rng), rectangle.sample_uniform(rng)
        assert rectangle.distance(x, y) == pytest.approx(float(np.linalg.norm(x - y)))


def test_wrap_clips_rather_than_folding():
    """A clipping map is not reversible, which is why `periodic` is False.

    Folding an MCMC proposal through it would pile every out-of-bounds draw onto
    the boundary instead of rejecting it, and the chain would target a density
    with atoms on the edge.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    np.testing.assert_allclose(rectangle.wrap(np.array([9.0, -9.0])), [2.0, -1.5])
    np.testing.assert_allclose(rectangle.wrap(np.array([1.0, 1.0])), [1.0, 1.0])
    # Idempotent, as `wrap` must be for any domain.
    once = rectangle.wrap(np.array([9.0, -9.0]))
    np.testing.assert_array_equal(rectangle.wrap(once), once)


def test_contains_answers_honestly_outside_the_box():
    """The inherited default returns True unconditionally.

    Sound for the quadrature, which only ever asks about nodes drawn from
    `bounds`, but not what the method says -- and a bounded domain is the first
    one where a caller might reasonably ask about a point outside.
    """
    rectangle = hp.Rectangle(4.0, 3.0)
    assert rectangle.contains(np.array([0.0, 0.0]))
    assert rectangle.contains(np.array([2.0, 1.5])), "the closed boundary is inside"
    assert not rectangle.contains(np.array([2.01, 0.0]))
    assert not rectangle.contains(np.array([0.0, -1.51]))


def test_it_has_no_deck_group():
    """No translations to sum over, so `make_periodic` falls back to `distance`."""
    assert hp.Rectangle(4.0, 3.0).orbit(np.array([0.0, 0.0])) is None


@pytest.mark.statistical
def test_sample_uniform_is_uniform():
    """Per axis, against the analytic uniform, by KS."""
    rectangle = hp.Rectangle(4.0, 3.0, origin=[1.0, -2.0])
    rng = np.random.default_rng(3)
    points = np.array([rectangle.sample_uniform(rng) for _ in range(4000)])

    assert np.all(points >= rectangle.lower)
    assert np.all(points <= rectangle.upper)
    for axis, (low, high) in enumerate(rectangle.bounds):
        result = stats.kstest(points[:, axis], stats.uniform(low, high - low).cdf)
        assert result.pvalue > 1e-3, f"axis {axis} is not uniform (p={result.pvalue:.4g})"


def test_an_origin_shifts_the_box_without_changing_its_size():
    shifted = hp.Rectangle(4.0, 3.0, origin=[10.0, 10.0])
    np.testing.assert_allclose(shifted.bounds, [[10.0, 14.0], [10.0, 13.0]])
    assert shifted.volume == pytest.approx(12.0)
    assert shifted.contains(np.array([12.0, 11.0]))
    assert not shifted.contains(np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    ("sides", "kwargs", "match"),
    [
        ((), {}, "at least one side"),
        ((0.0,), {}, "finite and positive"),
        ((-1.0,), {}, "finite and positive"),
        ((np.inf,), {}, "finite and positive"),
        ((4.0, 3.0), {"origin": [1.0]}, "one entry per side"),
        ((4.0,), {"origin": [np.nan]}, "origin must be finite"),
    ],
)
def test_construction_is_validated(sides, kwargs, match):
    with pytest.raises(ValueError, match=match):
        hp.Rectangle(*sides, **kwargs)


def test_a_process_simulates_on_it(flat_base, exp_kernel, bump_spatial):
    """It drops into the existing machinery with no correction yet."""
    process = hp.SpatioTemporalHawkesProcess(
        base=flat_base,
        spatial=bump_spatial,
        temporal=exp_kernel,
        domain=hp.Rectangle(4.0, 3.0),
        monotone_temporal_kernel=True,
        rng=0,
    )
    process.simulate(8)

    assert process.events.shape == (3, 8)
    locations = process.events[1:]
    assert np.all(locations[0] >= -2.0)
    assert np.all(locations[0] <= 2.0)
    assert np.all(locations[1] >= -1.5)
    assert np.all(locations[1] <= 1.5)
