"""Log-space weights, effective sample size, and the two resamplers.

Unbiasedness is the property a resampler has to have and the one that is
invisible when it is missing: a biased resampler still returns N indices, and
the cloud it produces still looks like a cloud. It is checked here over 20 000
repeats against four standard errors, which a correct implementation passes for
every seed and a biased one fails for all of them.
"""

import numpy as np
import pytest

from hawkes_package.inference.resample import (
    effective_sample_size,
    log_normalise,
    log_sum_exp,
    multinomial,
    systematic,
    unique_fraction,
)

# ---------------------------------------------------------------------------
# Log-space arithmetic
# ---------------------------------------------------------------------------


def test_log_sum_exp_survives_where_exp_underflows():
    """A block of a few hundred events moves the log weights by hundreds of nats."""
    values = [-10000.0, -10001.0]
    assert np.exp(values).sum() == 0.0, "the naive computation really does underflow"
    assert log_sum_exp(values) == pytest.approx(-10000.0 + np.log1p(np.exp(-1.0)), rel=1e-12)


def test_log_sum_exp_of_all_minus_infinity_is_minus_infinity():
    """And not `nan`, which subtracting -inf from itself would give."""
    assert log_sum_exp([-np.inf, -np.inf]) == -np.inf


def test_log_sum_exp_matches_the_direct_computation(rng):
    values = rng.normal(0.0, 3.0, size=200)
    assert log_sum_exp(values) == pytest.approx(np.log(np.sum(np.exp(values))), rel=1e-12)


def test_log_sum_exp_of_an_empty_array_is_minus_infinity():
    assert log_sum_exp([]) == -np.inf


def test_normalising_makes_the_weights_a_distribution(rng):
    values = rng.normal(0.0, 50.0, size=64)
    normalised = log_normalise(values)
    assert log_sum_exp(normalised) == pytest.approx(0.0, abs=1e-12)
    # Only the shift changed: the ratios are untouched.
    np.testing.assert_allclose(normalised - values, normalised[0] - values[0], atol=1e-12)


def test_a_weightless_cloud_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="zero weight"):
        log_normalise([-np.inf, -np.inf, -np.inf])


def test_nan_weights_are_named_rather_than_propagated():
    with pytest.raises(ValueError, match="nan"):
        log_normalise([0.0, np.nan, -1.0])


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


def test_equal_weights_give_the_full_sample_size():
    n = 128
    assert effective_sample_size(np.full(n, -np.log(n))) == pytest.approx(n, rel=1e-12)


def test_one_dominant_particle_gives_one():
    log_weights = log_normalise(np.array([0.0, -800.0, -800.0, -800.0]))
    assert effective_sample_size(log_weights) == pytest.approx(1.0, rel=1e-9)


def test_ess_lies_between_one_and_n(rng):
    for _ in range(20):
        log_weights = log_normalise(rng.normal(0.0, 4.0, size=32))
        ess = effective_sample_size(log_weights)
        assert 1.0 <= ess <= 32.0 + 1e-9


def test_unique_fraction_counts_distinct_ancestors():
    assert unique_fraction(np.array([0, 0, 0, 0]), 4) == 0.25
    assert unique_fraction(np.array([0, 1, 2, 3]), 4) == 1.0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


RESAMPLERS = {"systematic": systematic, "multinomial": multinomial}


@pytest.fixture(params=sorted(RESAMPLERS), ids=sorted(RESAMPLERS))
def resampler(request):
    return RESAMPLERS[request.param]


def test_resampling_returns_n_indices_in_range(resampler, rng):
    log_weights = log_normalise(rng.normal(0.0, 2.0, size=50))
    indices = resampler(log_weights, rng)
    assert indices.shape == (50,)
    assert np.all((indices >= 0) & (indices < 50))


def test_resampling_is_unbiased(resampler):
    """Particle i must be drawn N * w_i times in expectation.

    Four standard errors over 20 000 repeats: a correct resampler passes for
    every seed, and one that is off by even a fraction of a slot does not.
    """
    weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    log_weights = np.log(weights)
    n = weights.size
    repeats = 20_000
    rng = np.random.default_rng(2026)

    counts = np.zeros(n)
    for _ in range(repeats):
        counts += np.bincount(resampler(log_weights, rng), minlength=n)
    observed = counts / repeats
    expected = n * weights
    # Multinomial counts have variance n*w*(1-w) per repeat; systematic's is at
    # most that (Douc and Cappe 2005), so this bound holds for both.
    standard_error = np.sqrt(n * weights * (1 - weights) / repeats)
    assert np.all(np.abs(observed - expected) < 4 * standard_error)


def test_systematic_consumes_exactly_one_variate():
    """Which makes the number of draws a fit makes independent of the cloud size.

    Two runs that resample at the same steps then stay on the same stream,
    whatever N is -- and reproducibility across cloud sizes is what makes a
    diagnostic comparable between them.
    """
    log_weights = log_normalise(np.zeros(64))
    before = np.random.default_rng(0)
    systematic(log_weights, before)
    after = np.random.default_rng(0)
    after.uniform()
    assert before.uniform() == after.uniform()


def test_systematic_keeps_every_heavy_particle():
    """A particle with weight above 1/N cannot be dropped by systematic resampling.

    Not true of multinomial, which is the point of preferring it.
    """
    weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    log_weights = np.log(weights)
    rng = np.random.default_rng(7)
    for _ in range(200):
        kept = set(systematic(log_weights, rng).tolist())
        assert {0, 1, 2} <= kept


def test_systematic_is_no_noisier_than_multinomial():
    """Douc and Cappe: its conditional variance is no worse. Measured, not assumed."""
    weights = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
    log_weights = np.log(weights)
    repeats = 4000
    variances = {}
    for name, resampler in RESAMPLERS.items():
        rng = np.random.default_rng(11)
        counts = np.array(
            [np.bincount(resampler(log_weights, rng), minlength=5) for _ in range(repeats)]
        )
        variances[name] = counts.var(axis=0).sum()
    assert variances["systematic"] < variances["multinomial"]
