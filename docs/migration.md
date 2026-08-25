# Migrating to 0.2.0

Version 0.2.0 is the first packaged release. The import name changed, the API
was unified, and several genuine bugs were fixed — including some that change
previously produced numbers **without raising**. Read the last section before
re-running an old script.

## Import name

```diff
- import TheHawkesPackage as THP
+ import hawkes_package as hp
```

`import TheHawkesPackage` still works and forwards to identical objects (same
classes, same module objects, so `isinstance` and pickles are unaffected), but
emits a `DeprecationWarning`. The shim is present in 0.2.x and 0.3.x and is
**removed in 0.4.0**.

The distribution on PyPI is `the-hawkes-package`; only the import name is
`hawkes_package`.

## Renamed modules

| Old | New |
|---|---|
| `TheHawkesPackage.ExponentialHawkes` | `hawkes_package.exponential` |
| `TheHawkesPackage.MonotoneKernelHawkes` | `hawkes_package.monotone` |
| `TheHawkesPackage.BellShapeHawkes` | `hawkes_package.bell_shape` |
| `TheHawkesPackage.MCMC_sampler` | `hawkes_package.mcmc` |
| `TheHawkesPackage.SpatioTemporal_Hawkes_Monotone` | `hawkes_package.spatio_temporal.legacy` |
| `TheHawkesPackage.spatio_temporal.sampler` | `hawkes_package.mcmc` |

Class names are unchanged apart from the one below. The old dotted paths keep
working through the shim.

## Renamed methods

`simulate(k)` is the canonical method on every process class. The three older
spellings remain as aliases that warn and are removed in 0.4.0.

| Old | New | Removed in |
|---|---|---|
| `propagate_by_amount(k)` | `simulate(k)` | 0.4.0 |
| `propagate_by_k_events(k)` | `simulate(k)` | 0.4.0 |
| `propogate_by_amount(k)` *(typo)* | `simulate(k)` | 0.4.0 |
| `Spatio_Temporal_Hawkes_Process` | `LegacySpatioTemporalHawkesProcess` | 0.4.0 |

:::{note}
This table is kept in lockstep with the test suite: `RENAMES` in
`tests/test_deprecations.py` drives tests that assert each old name still warns
and still forwards, so the documentation cannot drift from the behaviour.
:::

## The ambiguous class name

`Spatio_Temporal_Hawkes_Process` used to mean **two different classes**:

- at top level, the legacy periodic-interval implementation;
- inside `TheHawkesPackage.spatio_temporal`, the domain-aware class.

Two different algorithms, and two different shapes of `Events`, behind one
identifier. In 0.2.0:

- `hawkes_package.Spatio_Temporal_Hawkes_Process` still resolves to the
  **legacy** class, so existing top-level code keeps its behaviour. It warns.
- `hawkes_package.spatio_temporal.Spatio_Temporal_Hawkes_Process` is **removed**
  and raises `AttributeError`. Repointing it would have silently switched those
  callers to a different algorithm; failing loudly is the safer trade.

```diff
- from TheHawkesPackage.spatio_temporal import Spatio_Temporal_Hawkes_Process
+ from hawkes_package import SpatioTemporalHawkesProcess     # the domain-aware class
+ from hawkes_package import LegacySpatioTemporalHawkesProcess  # the old one
```

## Randomness

**`np.random.seed(...)` no longer controls simulations.** Every process now
takes `rng=`, accepting `None`, an integer seed, or an existing
{class}`numpy.random.Generator`.

```diff
- np.random.seed(42)
- G = THP.ExponentialHawkes(np.array([2.0, 0.5, 1.0]))
+ G = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
```

An old script that seeded globally still runs — it just produces a different
realisation. Nothing warns, because there is no way to detect the intent.

## Changes that alter results

Each of these produces different numbers from the same code, with no error.

**`ExponentialHawkes.intensity_over_interval` omitted the background rate.**
The returned intensity excluded $\mu$, while the simulator's bound included it,
so plots sat a constant $\mu$ below the intensity actually being simulated. The
accessor and the simulator now share one implementation.

**Five thinning bounds were invalid.** Thinning is only correct while
$M \geq \lambda$; where that fails, candidates are accepted that should have
been rejected, biasing the simulated distribution towards Poisson.

- `BellShapeHawkes` inflated the whole intensity by a single kernel peak, which
  is not enough when two or more events are rising at once. The invariant failed
  in roughly 5% of steps.
- `SpatioTemporalHawkesProcess` and the legacy class excluded the event at
  exactly $t$ — and at the start of a thinning step $t$ *is* the most recent
  event time, so its entire excitation was missing. The invariant failed in
  about 70% of steps.

- The kernel peak was located by a local search started at lag 0. On a kernel
  that is flat there — the standard delayed-excitation shape — it returned 0, so
  the bell-shaped bound collapsed to the monotone one and the invariant failed in
  **46.3%** of steps.
- On a domain of two or more dimensions the spatial integral was Monte Carlo,
  redrawn per call, so the bound was an estimate rather than a bound and was
  compared against a second independent estimate: `P(lambda_hat > M_hat) = 0.437`.
- The bound multiplied `sup(kappa_t)` by `kappa_s`, which is the supremum of the
  product only where the spatial kernel is non-negative.

Series generated with 0.0.1 under any of these configurations were not draws
from the intended process. Regenerate them.

**The phantom event at `t = 0`.** Every process was seeded with a fictitious
event that contributed to the intensity until the first `simulate` call deleted
it. `E[T1]` was 12.77 where the model gives `1/mu = 20.0`, and
`simulate(1); simulate(1)` did not equal `simulate(2)`. Both are now correct.
`Events` starts empty, so `process.Events[-1]` before the first `simulate` raises
`IndexError` instead of returning `0.0`.

**Every location in the legacy class came from the wrong density**, through an
array-shape broadcast: `intensity(1.0, 0.15)` gave 1.191369 but
`intensity(1.0, [0.15])` gave 1.717217, and the sampler used the latter.

**The MCMC chain now stays inside the domain.** Proposals outside it are
rejected, and `proposal_std` defaults to a tenth of each axis's width instead of
a fixed 1.0, so locations on a large or anisotropic domain change — for the
better: on `Circle(radius=10)` only 30.3% of draws had landed in a peak holding
52.2% of the mass.

**User callables now always receive a shape-`(ndim,)` point.** A `base` or
`spatial` written against the old float-on-one-path behaviour may need adjusting;
`base=lambda x: 0.5 + 0.2*np.cos(x[0])` is the spelling that works everywhere.

**The legacy class ignored a non-default `Space`.** The spatial sampler
hard-coded $[-\pi, \pi]$, so a custom interval was accepted and silently
disregarded. It is now honoured, and the parameter is renamed:

```diff
- G = THP.Spatio_Temporal_Hawkes_Process(base, spatial, temporal, Space=[-2.0, 2.0])
+ G = hp.LegacySpatioTemporalHawkesProcess(base, spatial, temporal, space=(-2.0, 2.0))
```

`Space=` still works for one release, with a warning.

**Importing no longer reseeds your `random` module.** The legacy module called
`random.seed(42)` at import time, silently resetting the caller's global stream
as a side effect of an `import`. If you were (unknowingly) relying on that, seed
it yourself.

## New in 0.2.0

Both spatio-temporal classes gained `intensity(t, x)` and
`intensity_over_interval(times, points)`. Previously there was no way to
evaluate the field intensity without re-implementing it by hand:

```diff
- def intensity_G(time, space):
-     t = np.array([G.temporal(time - G.Events[0, i]) for i in ... ])
-     s = np.array([G.spatial(periodizer(space - G.Events[1, i])) for i in ... ])
-     return 0.5 + np.multiply(t, s).sum()
- z = np.vectorize(intensity_G)(time_grid, space_grid)
+ time, space, z = G.intensity_over_interval(times, points=points)
```

`z` is `(n_space, n_time)` — rows index space, columns index time, the layout
{func}`matplotlib.pyplot.contourf` expects.

An exploding process now raises `RuntimeError` instead of looping forever once
inter-arrival times underflow to zero.
