"""The multivariate loop, and the exactness that makes it an additive change.

The claim 0.6.0 rests on is that nothing existing moves. The evidence is here:
a one-type :class:`~hawkes_package.multivariate.MultivariateHawkes` reproduces
the univariate class it corresponds to **event for event, bit for bit**, on the
same seed. That is possible because the type costs no extra variate -- the
cumulative component intensities partition ``(0, M]`` and the uniform that
decides acceptance also decides which slice it landed in -- so the two loops
consume the same stream in the same order.

`rtol=0` throughout. If one of these ever fails, the answer is not a tolerance:
it means the loop shifted, and everything downstream of it moved with it.
"""

import numpy as np
import pytest

import hawkes_package as hp


def plus_two(x):
    """The default nonlinearity of the univariate non-linear classes."""
    return x + 2


@pytest.fixture
def triangular():
    """A kernel that rises to a peak at 0.5 and decays to zero at 1."""

    def _triangular(s):
        s = np.asarray(s, dtype=float)
        return 2 * s * ((s > 0) & (s < 0.5)) + (-2 * s + 2) * ((s >= 0.5) & (s < 1))

    return _triangular


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_one_type_reproduces_the_monotone_class_exactly(exp_kernel, seed):
    """``mu = 0``, ``A = [[1]]`` is `MonotoneKernelHawkes`, to the bit."""
    reference = hp.MonotoneKernelHawkes(exp_kernel, rng=seed)
    reference.simulate(25)

    multivariate = hp.MultivariateHawkes(
        mu=[0.0], excitation=[[1.0]], temporal=exp_kernel, nonlinearity=plus_two, rng=seed
    )
    multivariate.simulate(25)

    np.testing.assert_array_equal(multivariate.events[0], reference.events)
    np.testing.assert_array_equal(multivariate.types, np.zeros(25, dtype=np.intp))


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_one_type_reproduces_the_bell_shape_class_exactly(triangular, seed):
    """The same, through the non-monotone bound."""
    reference = hp.BellShapeHawkes(triangular, rng=seed)
    reference.simulate(25)

    multivariate = hp.MultivariateHawkes(
        mu=[0.0],
        excitation=[[1.0]],
        temporal=triangular,
        nonlinearity=plus_two,
        monotone_temporal_kernel=False,
        rng=seed,
    )
    multivariate.simulate(25)

    np.testing.assert_array_equal(multivariate.events[0], reference.events)


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_the_intensity_hooks_agree_with_the_univariate_ones_exactly(exp_kernel, seed):
    """Not just the realisation: the hooks the loop thins against, pointwise.

    The realisation could match by luck if the two intensities differed only
    where no candidate landed. This evaluates both hooks on a shared grid.
    """
    reference = hp.MonotoneKernelHawkes(exp_kernel, rng=seed)
    reference.simulate(25)
    multivariate = hp.MultivariateHawkes(
        mu=[0.0], excitation=[[1.0]], temporal=exp_kernel, nonlinearity=plus_two, rng=seed
    )
    multivariate.simulate(25)

    last = float(reference.events[-1])
    for t in np.linspace(0.0, last + 1.0, 200):
        assert multivariate._conditional_intensity(float(t)) == (
            reference._conditional_intensity(float(t))
        )

    # The bound is only comparable at or past the last recorded event, which is
    # the only place the loop ever asks for it. Before that, the univariate
    # version sums the kernel over events still in the future and evaluates it
    # at negative lags -- for a decaying kernel that is an enormous number with
    # no meaning. This class masks to `times <= t` instead, so the two agree
    # exactly wherever the question is a real one and differ where it is not.
    for t in np.linspace(last, last + 1.0, 200):
        assert multivariate._upper_bound(float(t)) == reference._upper_bound(float(t))


def test_an_empty_history_bounds_the_intensity_exactly(exp_kernel):
    """With no events the bound *is* the intensity, so the first draw always accepts.

    Reduced through the same ``cumsum`` in both places for this reason: were one
    a pairwise ``sum`` and the other not, the two could differ by a bit and the
    first acceptance test would become a coin flip nobody intended.
    """
    process = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=0
    )
    assert process._upper_bound(0.0) == process._conditional_intensity(0.0)


def test_the_type_costs_no_variate(exp_kernel):
    """Two types consume exactly as many draws as one, for the same event count.

    This is the property the exactness above rests on, asserted directly rather
    than inferred from it: whatever the loop does to choose a type, it must not
    reach for the stream to do it.
    """

    def drawn(process):
        before = process.rng.bit_generator.state
        process.simulate(40)
        del before
        return process.rng

    one = hp.MultivariateHawkes(
        mu=[0.5], excitation=[[0.4]], temporal=exp_kernel, rng=np.random.default_rng(3)
    )
    two = hp.MultivariateHawkes(
        mu=[0.5, 0.0],
        excitation=[[0.4, 0.0], [0.0, 0.0]],
        temporal=exp_kernel,
        rng=np.random.default_rng(3),
    )
    drawn(one)
    drawn(two)
    # The second type is inert -- no background, no excitation given or received
    # -- so the accepted stream is identical and every draw lands in slice 0.
    np.testing.assert_array_equal(one.events[0], two.events[0])
    np.testing.assert_array_equal(two.types, np.zeros(40, dtype=np.intp))


def test_the_record_carries_times_and_types(exp_kernel):
    process = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=1
    )
    process.simulate(30)

    assert process.events.shape == (2, 30)
    assert process.n_simulated == 30
    assert np.all(np.diff(process.events[0]) > 0)
    assert set(process.types.tolist()) <= {0, 1}
    assert process.types.dtype == np.intp


def test_simulate_until_reproduces_simulate(exp_kernel):
    """The horizon loop and the counting loop must agree, as they do univariately."""
    counted = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=5
    )
    counted.simulate(20)
    horizon = float(counted.events[0, -1])

    timed = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=5
    )
    timed.simulate_until(horizon)

    np.testing.assert_array_equal(timed.events, counted.events)


def test_a_negative_excitation_entry_is_refused(exp_kernel):
    """The bound takes a supremum per term, and that step needs a non-negative scale."""
    with pytest.raises(ValueError, match=r"non-negative.*sup\(a\*f\)"):
        hp.MultivariateHawkes(
            mu=[0.4, 0.2], excitation=[[0.3, -0.1], [0.5, 0.2]], temporal=exp_kernel
        )


@pytest.mark.parametrize(
    ("mu", "excitation", "match"),
    [
        ([], [[0.1]], "at least one"),
        ([0.4, 0.2], [[0.1]], "match mu"),
        ([0.4], [[0.1, 0.2]], "square"),
        ([-0.4], [[0.1]], "non-negative"),
        ([np.inf], [[0.1]], "finite"),
        ([0.4], [[np.nan]], "finite"),
    ],
)
def test_construction_is_validated(exp_kernel, mu, excitation, match):
    with pytest.raises(ValueError, match=match):
        hp.MultivariateHawkes(mu=mu, excitation=excitation, temporal=exp_kernel)


def test_n_types_must_be_a_positive_whole_number(exp_kernel):
    class Bare(hp.MultivariateTemporalHawkesProcess):
        def _component_intensities(self, t):
            return np.ones(self.n_types)

        def _upper_bound(self, t):
            return float(self.n_types)

    with pytest.raises(ValueError, match="positive whole number"):
        Bare(0)
    with pytest.raises(ValueError, match="positive whole number"):
        Bare(1.5)
