"""`History` carrying an event type, and the trap that makes it delicate.

`from_events` reads any two-dimensional record as spatio-temporal. That is the
right reading for every record 0.5.0 could produce, and the wrong one for a
multivariate record, whose last row is a type index rather than a coordinate.
Nothing raises: the types become coordinates, `ndim` counts one too many, and
the fit comes back converged on a process nobody described. So the multivariate
layout gets its own constructor, and this file is where that separation is
pinned.
"""

import numpy as np
import pytest

import hawkes_package as hp
from hawkes_package.inference import History

TIMES = np.array([0.4, 1.1, 2.9])
TYPES = np.array([0, 1, 0])


def test_the_documented_example_still_constructs():
    """The two new fields are keyword-defaulted, so every existing call survives."""
    history = History(np.array([0.4, 1.1, 2.9]), None, 0.0, end=4.0)
    np.testing.assert_array_equal(history.upto(1.5).times, np.array([0.4, 1.1]))
    assert history.types is None
    assert history.n_types is None
    assert history.ndim == 0


def test_a_typed_history_reports_its_types():
    history = History(TIMES, None, 0.0, 4.0, TYPES, 2)
    np.testing.assert_array_equal(history.types, TYPES)
    assert history.n_types == 2
    assert history.types.dtype == np.intp
    # The type row is not a coordinate, and must not be counted as one.
    assert history.ndim == 0


def test_types_and_n_types_must_be_given_together():
    with pytest.raises(ValueError, match="must be given together"):
        History(TIMES, None, 0.0, 4.0, TYPES, None)
    with pytest.raises(ValueError, match="must be given together"):
        History(TIMES, None, 0.0, 4.0, None, 2)


def test_n_types_is_not_inferred_from_the_observed_types():
    """A type that produced no event still exists, and must survive the window.

    Inferring ``n_types = types.max() + 1`` would drop it here, and with it the
    model's background for that type and its column of the excitation matrix --
    silently, and only on the windows where it happened to be quiet.
    """
    history = History(TIMES, None, 0.0, 4.0, TYPES, 5)
    assert history.n_types == 5


@pytest.mark.parametrize(
    ("types", "n_types", "match"),
    [
        (np.array([0, 1]), 2, "one entry per event"),
        (np.array([0, 1, 2]), 2, r"\[0, n_types\)"),
        (np.array([0, -1, 0]), 2, r"\[0, n_types\)"),
        (np.array([0.0, 1.5, 0.0]), 2, "whole numbers"),
        (np.array([0.0, np.nan, 0.0]), 2, "finite"),
        (TYPES, 0, "positive whole number"),
        (TYPES, 2.5, "positive whole number"),
    ],
)
def test_types_are_validated(types, n_types, match):
    with pytest.raises(ValueError, match=match):
        History(TIMES, None, 0.0, 4.0, types, n_types)


def test_upto_slices_times_and_types_together():
    history = History(TIMES, None, 0.0, 4.0, TYPES, 2)
    cut = history.upto(1.5)
    np.testing.assert_array_equal(cut.times, TIMES[:2])
    np.testing.assert_array_equal(cut.types, TYPES[:2])
    assert cut.n_types == 2


def test_upto_slices_consistently_at_every_cut():
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.01, 9.99, size=60))
    types = rng.integers(0, 3, size=60)
    history = History(times, None, 0.0, 10.0, types, 3)

    for cut in rng.uniform(0.0, 10.0, size=100):
        sliced = history.upto(float(cut))
        keep = times <= cut
        np.testing.assert_array_equal(sliced.times, times[keep])
        np.testing.assert_array_equal(sliced.types, types[keep])


def test_the_record_round_trips_through_a_process(exp_kernel):
    """`as_process_events` must produce exactly what the process records."""
    process = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=2
    )
    process.simulate(25)
    end = float(process.events[0, -1])

    history = History.from_multivariate_events(process.events, n_types=2, end=end)
    np.testing.assert_array_equal(history.times, process.events[0])
    np.testing.assert_array_equal(history.types, process.types)
    assert history.points is None

    rebuilt = history.as_process_events()
    np.testing.assert_array_equal(rebuilt, process.events)

    # And back through the seeding path the simulator actually uses.
    fresh = hp.MultivariateHawkes(
        mu=[0.4, 0.2], excitation=[[0.3, 0.1], [0.5, 0.2]], temporal=exp_kernel, rng=2
    )
    fresh.events = rebuilt
    np.testing.assert_array_equal(fresh.events, process.events)


def test_a_spatio_temporal_multivariate_record_round_trips():
    """Times in row 0, the type in the last row, coordinates in between."""
    times = np.array([0.4, 1.1, 2.9])
    points = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    record = np.vstack([times[None, :], points, TYPES.astype(float)[None, :]])

    history = History.from_multivariate_events(record, n_types=2, end=4.0)
    np.testing.assert_array_equal(history.times, times)
    np.testing.assert_array_equal(history.points, points)
    np.testing.assert_array_equal(history.types, TYPES)
    assert history.ndim == 2
    np.testing.assert_array_equal(history.as_process_events(), record)


def test_from_multivariate_events_needs_at_least_two_rows():
    with pytest.raises(ValueError, match="at least 2 rows"):
        History.from_multivariate_events(TIMES, n_types=2, end=4.0)


def test_from_events_misreads_a_multivariate_record():
    """The trap, pinned rather than left to be rediscovered.

    This is not desirable behaviour and it is not a bug that can be fixed in
    place: `from_events`' two-dimensional reading is correct for every record a
    single-type process produces, and the shape alone cannot distinguish a
    one-dimensional multivariate record from a one-dimensional spatio-temporal
    one. The separation is the constructor, and this test exists so that anyone
    tempted to merge the two sees what merging them would cost.
    """
    record = np.vstack([TIMES[None, :], TYPES.astype(float)[None, :]])

    misread = History.from_events(record, end=4.0)
    assert misread.ndim == 1, "the type row was read as a coordinate"
    assert misread.types is None
    np.testing.assert_array_equal(misread.points[0], TYPES.astype(float))

    correct = History.from_multivariate_events(record, n_types=2, end=4.0)
    assert correct.ndim == 0
    np.testing.assert_array_equal(correct.types, TYPES)


def test_from_events_accepts_types_held_separately():
    history = History.from_events(TIMES, end=4.0, types=TYPES, n_types=2)
    np.testing.assert_array_equal(history.types, TYPES)
    assert history.n_types == 2


def test_the_buffer_error_names_the_type_row(exp_kernel):
    """A multivariate record's second row is a type, not a coordinate."""
    process = hp.MultivariateHawkes(mu=[0.4], excitation=[[0.3]], temporal=exp_kernel, rng=0)
    with pytest.raises(ValueError, match="holding the event type"):
        process.events = np.zeros((3, 5))
