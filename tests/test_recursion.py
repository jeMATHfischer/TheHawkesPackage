"""The exponential intensity carried forward instead of rebuilt.

`ExponentialHawkes` used to re-sum every past event on both of the loop's
evaluations, 2.36 O(n) reductions per accepted event. Since 1.0.0 it carries
``S(t) = sum exp(-beta (t - t_i))`` and advances it with one multiply, which
turns a quadratic simulation into a linear one -- 157 µs per event at 8 000
events before, 6.0 µs after, and flat out to 16 000.

Two things are asserted here that nothing else would catch.

**The recursion is not a second definition of the intensity.** `CLAUDE.md`'s
strongest rule is that an accessor written separately from the hook the
simulator thins against is how `ExponentialHawkes` came to plot a curve a
constant `mu` below its own intensity. So the carried sum is checked against the
hook at every event of a long run, over seeds 0-20.

**The continuation guarantees are what force the state to persist.** Rebuilding
the sum from the record at the start of the next call gives a value a few ulps
away from the one an uninterrupted loop was carrying, and `simulate(1);
simulate(1)` is asserted elsewhere to be `simulate(2)` bit for bit. Those
assertions are repeated here at greater length, because here they are load
bearing rather than incidental.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.exponential import _DecayCursor

PARAM = np.array([1.0, 0.5, 1.0])  # branching ratio 0.5


def test_the_carried_sum_agrees_with_the_hook_at_every_event():
    """Over seeds 0-20 the worst relative disagreement is 4.4e-16, about two ulps.

    The tolerance below is three orders above that, and it is a tolerance rather
    than an equality for a reason worth stating: a product of decays and a
    single exponential of the total lag are different arithmetic, and no amount
    of care makes them the same bits.
    """
    worst = 0.0
    for seed in range(21):
        process = hp.ExponentialHawkes(PARAM, rng=seed)
        process.simulate(400)
        events = process.events

        cursor = _DecayCursor(hp.ExponentialHawkes(PARAM, rng=0), 0.0)
        for t in events:
            cursor.move_to(float(t))
            carried = cursor.intensity()
            direct = process._conditional_intensity(float(t))
            worst = max(worst, abs(carried - direct) / direct)
            cursor.accept()

    assert worst < 1e-13, f"the recursion drifted from the hook by {worst:.3e}"


def test_the_loop_no_longer_rebuilds_the_sum():
    """The complexity claim, asserted deterministically rather than by a clock.

    A wall-clock threshold on a shared runner is theatre. What actually changed
    is that the loop stopped calling the two O(n) hooks at all: 2.36 reductions
    per accepted event before, zero now. A regression that reintroduced them
    would show here on any machine, at any speed.
    """
    process = hp.ExponentialHawkes(PARAM, rng=0)
    calls = {"intensity": 0, "bound": 0}
    original_intensity = process._conditional_intensity
    original_bound = process._upper_bound

    def counting_intensity(t):
        calls["intensity"] += 1
        return original_intensity(t)

    def counting_bound(t):
        calls["bound"] += 1
        return original_bound(t)

    process._conditional_intensity = counting_intensity
    process._upper_bound = counting_bound
    process.simulate(300)

    assert len(process.events) == 300
    assert calls == {"intensity": 0, "bound": 0}


@pytest.mark.slow
def test_the_cost_per_event_stops_growing_with_n():
    """Complexity, not constants: the ratio survives a noisy machine.

    Quadratic work quadruples when `n` doubles -- measured at 4.14 and 4.09 for
    the last two doublings before this change. Linear work does not, and the
    threshold sits nearer the quadratic answer than the linear one so that only
    a genuine return to re-summing trips it.
    """
    import time

    def seconds(n):
        process = hp.ExponentialHawkes(PARAM, rng=0)
        start = time.perf_counter()
        process.simulate(n)
        return time.perf_counter() - start

    small = min(seconds(4000) for _ in range(3))
    large = min(seconds(8000) for _ in range(3))
    assert large / small < 3.0, (
        f"doubling n multiplied the work by {large / small:.2f}; linear is 2 and "
        "the pre-1.0.0 quadratic loop was 4.09"
    )


def test_repeated_single_steps_equal_one_long_call():
    """Forty calls of one against one call of forty, bit for bit.

    `test_base.py` asserts this for two events across all three temporal
    classes. Forty is here because the carried sum is what makes it fragile:
    every continuation is an opportunity to rebuild the state slightly
    differently, and the error would compound rather than cancel.
    """
    stepwise = hp.ExponentialHawkes(PARAM, rng=3)
    for _ in range(40):
        stepwise.simulate(1)

    whole = hp.ExponentialHawkes(PARAM, rng=3)
    whole.simulate(40)

    np.testing.assert_array_equal(stepwise.events, whole.events)


def test_a_horizon_at_the_kth_event_still_reproduces_the_count():
    """The two loops share one cursor implementation, so they still agree exactly."""
    counted = hp.ExponentialHawkes(PARAM, rng=17)
    counted.simulate(40)
    horizon = float(counted.events[-1])

    timed = hp.ExponentialHawkes(PARAM, rng=17)
    timed.simulate_until(horizon)

    np.testing.assert_array_equal(timed.events, counted.events)


def test_replacing_the_record_discards_the_carried_sum():
    """The failure the invalidation exists for, and it is a quiet one.

    Seeding a realisation with ``process.events = history`` is supported. A
    replacement with the same length and the same last time would otherwise
    satisfy the reuse test -- same position, same count -- while the sum it
    carries describes a history that is no longer there. The bound would then be
    computed for the wrong past, with nothing raising.
    """
    process = hp.ExponentialHawkes(PARAM, rng=1)
    process.simulate(6)
    original = process.events.copy()

    # A different history with the same length, and a last event at the same
    # time, so every cheap identity check would pass.
    replacement = np.concatenate([original[:-1] - 5.0, original[-1:]])
    process.events = replacement

    cursor = process._cursor(float(replacement[-1]))
    rebuilt = 1.0 + 0.5 * float(np.exp(-1.0 * (replacement[-1] - replacement)).sum())
    assert cursor.bound() == pytest.approx(rebuilt, rel=1e-12)
    assert cursor.bound() != pytest.approx(
        1.0 + 0.5 * float(np.exp(-1.0 * (original[-1] - original)).sum()), rel=1e-6
    )


def test_a_non_positive_bound_still_raises_from_the_cursor():
    """The check moved into a shared helper; the message must not have moved with it."""
    process = hp.ExponentialHawkes(np.array([0.0, 0.5, 1.0]), rng=0)
    cursor = process._cursor(0.0)
    with pytest.raises(RuntimeError, match="Non-positive thinning bound"):
        cursor.bound()


def test_the_intensity_hook_is_unchanged():
    """The hook is still the definition, and still the direct sum.

    It is what `intensity_over_interval`, the likelihood and every diagnostic
    read, and it must not acquire the recursion's rounding -- a compensator and
    an intensity that disagree by even 1e-15 is not a problem, but two different
    expressions for one quantity eventually is.
    """
    process = hp.ExponentialHawkes(PARAM, rng=2)
    process.simulate(50)
    events = process.events
    mu, alpha, beta = PARAM

    for t in (0.0, float(events[10]) + 0.3, float(events[-1]) * 0.5):
        past = events[events < t]
        expected = mu + alpha * float(np.exp(-beta * (t - past)).sum())
        assert process._conditional_intensity(t) == expected
