"""Record where the time goes, so a performance claim can be checked later.

Run locally and paste the tables into ``RESULTS.md`` beside this file, with the
machine they came from. **Not a test.** Wall-clock thresholds on a shared runner
measure the runner, and the suite's performance assertions are written to be
machine-independent instead: `tests/test_recursion.py` asserts that the thinning
loop makes *no* ``O(n)`` reductions, which is the same claim without a clock.

What each table is for:

``simulate``
    The temporal loop. Since 0.9.0 `ExponentialHawkes` carries its intensity
    forward, so its cost per event is flat in ``n`` while the other two classes
    stay quadratic -- running all three is what makes that visible.
``log-likelihood``
    `ExponentialLogLikelihood` is the Ozaki recursion and linear;
    `TemporalLogLikelihood` goes through the intensity hook and is ``O(n^2 P)``.
    The gap is what makes a general-kernel fit a different size of job.
``spatio-temporal``
    The known slow path: every candidate costs a space integral, and every node
    of it costs a geodesic distance per past event. Note which side is slow -- a
    *fit* of 60 events is seconds against a minute to generate them.
``fit``
    An SMC fit, where rejuvenation dominates at ``n_particles x n_move`` full
    likelihood evaluations per resample.

Usage::

    python benchmarks/run.py            # the quick set, about a minute
    python benchmarks/run.py --full     # adds the sizes that take minutes
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior,
    ExponentialLogLikelihood,
    GaussianSpatial,
    History,
    IndependentPrior,
    LogNormal,
    TemporalLogLikelihood,
    exponential_model,
    fit_smc,
)

PARAM = np.array([1.0, 0.5, 1.0])  # branching ratio 0.5


def timed(fn: Callable[[], Any], repeats: int = 1) -> float:
    """Return the *best* of `repeats` runs, in seconds.

    The minimum rather than the mean: a slow run measures whatever else the
    machine was doing, and there is no such thing as a spuriously fast one.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def decay(lags: Any) -> np.ndarray:
    """Return the shared exponential kernel, mass 0.5."""
    return np.asarray(0.5 * np.exp(-1.0 * np.asarray(lags, dtype=float)), dtype=float)


def rising(lags: Any) -> np.ndarray:
    """Return a bell-shaped kernel, for the class whose bound needs a peak."""
    values = np.asarray(lags, dtype=float)
    return np.asarray(values * np.exp(-2.0 * values), dtype=float)


def make_temporal(name: str) -> hp.HawkesProcess:
    """Build one of the three temporal classes at a comparable excitation."""
    if name == "ExponentialHawkes":
        return hp.ExponentialHawkes(PARAM, rng=0)
    if name == "MonotoneKernelHawkes":
        return hp.MonotoneKernelHawkes(decay, rng=0)
    return hp.BellShapeHawkes(rising, rng=0)


def simulation_rows(full: bool) -> list[tuple[str, int, float, float]]:
    """Return the cost per event for each temporal class, at growing `n`."""
    sizes = [500, 2000, 8000] + ([32000] if full else [])
    quadratic_ceiling = 8000 if full else 2000
    rows = []
    for name in ("ExponentialHawkes", "MonotoneKernelHawkes", "BellShapeHawkes"):
        for n in sizes:
            # The quadratic classes stop earlier: at 8 000 events they already
            # take minutes, which is the finding rather than a shortcoming of
            # the benchmark.
            if name != "ExponentialHawkes" and n > quadratic_ceiling:
                continue
            seconds = timed(lambda name=name, n=n: make_temporal(name).simulate(n))
            rows.append((name, n, seconds, 1e6 * seconds / n))
    return rows


def likelihood_rows(full: bool) -> list[tuple[str, int, float]]:
    """Return one log-likelihood evaluation, closed form against the hook path."""
    rows = []
    model = exponential_model()
    for n in [500, 2000] + ([8000] if full else []):
        process = hp.ExponentialHawkes(PARAM, rng=0)
        process.simulate(n)
        history = History.from_events(process.events, end=float(process.events[-1]))

        closed = ExponentialLogLikelihood(model)
        rows.append(
            (
                "ExponentialLogLikelihood",
                n,
                timed(lambda closed=closed, history=history: closed.total(PARAM, history), 5),
            )
        )
        if n <= 2000:
            hooks = TemporalLogLikelihood(model, check=False)
            rows.append(
                (
                    "TemporalLogLikelihood",
                    n,
                    timed(lambda hooks=hooks, history=history: hooks.total(PARAM, history), 3),
                )
            )
    return rows


def spatial_process() -> hp.SpatioTemporalHawkesProcess:
    """Build the spatio-temporal process the rows below time."""
    return hp.SpatioTemporalHawkesProcess(
        base=lambda _: 0.5,
        spatial=GaussianSpatial(1).build(np.array([0.6])),
        temporal=decay,
        domain=hp.Circle(),
        monotone_temporal_kernel=True,
        rng=0,
    )


def spatio_temporal_rows(full: bool) -> list[tuple[str, int, float]]:
    """Return the cost of generating events on a surface."""
    rows = []
    for n in [20] + ([60] if full else []):
        rows.append(("simulate on a Circle", n, timed(lambda n=n: spatial_process().simulate(n))))
    return rows


def fit_rows(full: bool) -> list[tuple[str, int, float]]:
    """Return the cost of a whole SMC fit, where rejuvenation dominates."""
    rows = []
    model = exponential_model()
    prior = ConstrainedPrior(
        IndependentPrior((LogNormal(0.0, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
        model.support,
    )
    for n in [500] + ([2000] if full else []):
        process = hp.ExponentialHawkes(PARAM, rng=0)
        process.simulate(n)
        history = History.from_events(process.events, end=float(process.events[-1]))
        likelihood = ExponentialLogLikelihood(model)
        rows.append(
            (
                "fit_smc, 128 particles, 4 blocks",
                n,
                timed(
                    lambda likelihood=likelihood, history=history: fit_smc(
                        likelihood, prior, history, blocks=4, n_particles=128, rng=0
                    )
                ),
            )
        )
    return rows


def main() -> int:
    """Run the benchmarks and print them as markdown tables."""
    parser = argparse.ArgumentParser(description="Record where the time goes.")
    parser.add_argument("--full", action="store_true", help="include the sizes that take minutes")
    args = parser.parse_args()

    print(
        f"{platform.python_implementation()} {platform.python_version()} on "
        f"{platform.system()} {platform.machine()}, numpy {np.__version__}, "
        f"hawkes_package {hp.__version__}\n"
    )

    print("| simulate | n | seconds | us/event |")
    print("|---|---|---|---|")
    for name, n, seconds, per_event in simulation_rows(args.full):
        print(f"| `{name}` | {n} | {seconds:.3f} | {per_event:.1f} |")

    print("\n| one log-likelihood | n | seconds |")
    print("|---|---|---|")
    for name, n, seconds in likelihood_rows(args.full):
        print(f"| `{name}` | {n} | {seconds:.4f} |")

    print("\n| spatio-temporal | n | seconds |")
    print("|---|---|---|")
    for name, n, seconds in spatio_temporal_rows(args.full):
        print(f"| {name} | {n} | {seconds:.2f} |")

    print("\n| fit | n | seconds |")
    print("|---|---|---|")
    for name, n, seconds in fit_rows(args.full):
        print(f"| {name} | {n} | {seconds:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
