# Recorded benchmark results

Produced by `python benchmarks/run.py`. **Numbers from one machine, recorded by
hand.** There is deliberately no wall-clock assertion in CI: a threshold on a
shared runner measures the runner, and the suite's performance claims are
written to be machine-independent instead — `tests/test_recursion.py` asserts
that the thinning loop makes *no* `O(n)` reductions per step, which is the same
claim without a clock.

Recorded 2026-09-10, CPython 3.12.10 on Windows AMD64, numpy 1.26.4.

## Simulation

| simulate | n | seconds | µs/event |
|---|---|---|---|
| `ExponentialHawkes` | 500 | 0.007 | 13.7 |
| `ExponentialHawkes` | 2 000 | 0.020 | 10.1 |
| `ExponentialHawkes` | 8 000 | 0.063 | 7.8 |
| `MonotoneKernelHawkes` | 500 | 0.028 | 55.9 |
| `MonotoneKernelHawkes` | 2 000 | 0.162 | 81.0 |
| `BellShapeHawkes` | 500 | 0.043 | 86.9 |
| `BellShapeHawkes` | 2 000 | 0.236 | 118.0 |

Read the last column, not the middle one. `ExponentialHawkes` gets **cheaper**
per event as `n` grows — fixed costs amortising over a linear loop — while the
other two get dearer, which is the quadratic intensity sum showing through. That
contrast is the whole content of the 1.0.0 recursion.

Before 1.0.0 `ExponentialHawkes` read 28.4, 37.1 and 157.2 µs/event at those
three sizes, and 1.258 s in total at 8 000.

## One log-likelihood evaluation

| | n | seconds |
|---|---|---|
| `ExponentialLogLikelihood` | 500 | 0.0003 |
| `TemporalLogLikelihood` | 500 | 0.0695 |
| `ExponentialLogLikelihood` | 2 000 | 0.0011 |
| `TemporalLogLikelihood` | 2 000 | 0.5431 |

**494× at 2 000 events**, and the gap widens with `n` because one is `O(n)` and
the other `O(n²P)`. This is why a general-kernel fit is a different size of job
from an exponential one, and why `HawkesEstimator` picks the closed form when it
can: an SMC fit is thousands of these.

## Spatio-temporal

| | n | seconds |
|---|---|---|
| simulate on a `Circle` | 20 | 21.5 |

Roughly a second per event at twenty events, and worse from there: every
candidate costs a space integral and every node of it costs a geodesic distance
per past event. Note which side is slow — *fitting* a spatio-temporal history is
milliseconds through the cached backend, against minutes to generate one.

## A whole fit

| | n | seconds |
|---|---|---|
| `fit_smc`, 128 particles, 4 blocks | 500 | 0.22 |

With the closed-form likelihood a fit is a fifth of a second. Substituting
`TemporalLogLikelihood` multiplies it by the ratio in the table above, which is
the honest way to size a general-kernel fit before starting one.
