# Migration

One section per release that requires action. Newest first.

## Migrating to 0.3.0

**Nothing here changes what an existing script produces.** `Circle` and
`Torus2D` fill their bounding boxes, so no quadrature node is masked, no weight
is rescaled and no extra draw is consumed — the same seed runs the same code
path it ran in 0.2.0. This section is for authors of **custom `SpatialDomain`
subclasses**, where one contract did change.

### The domain contract

0.2.0 required a domain to *be* its bounding box, to a relative tolerance of
`1e-9`:

```text
volume == prod(bounds widths)
```

0.3.0 requires only that the declared volume match the measure the domain
actually carries:

```text
volume == integral of volume_element over {x in bounds : contains(x)}
```

`contains` defaults to `True` and `volume_element` to `1.0`, which is the old
requirement written out — so **a subclass that overrides neither needs no
change.**

### The volume check now warns where it used to raise

The old equality was checked against the bounding box and raised. It is now
checked against the summed quadrature weights, with two thresholds:

| `volume` vs. the rule's own measure | 0.2.0 | 0.3.0 |
|---|---|---|
| within `1e-9` relative | passes | passes |
| up to 1% apart | `ValueError` | passes |
| 1% to 10% apart | `ValueError` | `UserWarning` |
| more than 10% apart | `ValueError` | `ValueError` |

Two consequences worth acting on:

- **A domain that used to be rejected may now simulate.** Both faults the check
  catches — a domain that misstates its measure, and a rule too coarse to
  resolve its boundary — scale the simulated event rate by the same factor they
  get the measure wrong by, and neither has any other symptom. If you have a
  subclass whose `volume` was approximate, make it exact, or raise `n_quad`
  until the warning clears.
- **The `UserWarning` is fatal under `-W error`.** A downstream suite
  configured with `filterwarnings = ["error"]` — as this package's own is —
  turns the 1%-to-10% band back into a hard failure at the same construction
  call, with `UserWarning` in place of `ValueError`. That is the intended
  reading: the rule is telling you the event rate will be wrong.

### New optional hooks on `SpatialDomain`

`contains`, `volume_element`, `orbit` and `interior_point`, each defaulting to
pre-0.3.0 behaviour. A domain that is a proper subset of its bounding box must
override `contains`, and `interior_point` as well — the box centre need not lie
inside such a domain, and that point is used to probe whether the quadrature
rule resolves the spatial kernel. Override `volume_element` if the chart is not
flat.

`make_periodic` now periodises **any** domain that declares `orbit`, by summing
the kernel over the image points. Before 0.3.0 it recognised only `Circle` and
`Torus2D` and handed every other domain the unperiodised kernel.

### New in 0.3.0

{class}`~hawkes_package.FundamentalDomain`: a convex
Euclidean polygon plus the side-pairing isometries that identify its boundary,
presenting a flat orientable surface. `FundamentalDomain.hexagon` gives the
hexagonal torus — the first quotient in the package no rectangular domain
expresses — and `FundamentalDomain.rectangle` reproduces `Torus2D` through the
general machinery. Only orientation-preserving pairings are accepted. See
[theory](theory.md) for the construction and why the masked rule leaves the
Ogata bound intact.

## Migrating to 0.2.0

Version 0.2.0 is the first packaged release. The import name changed, the API
was unified, and several genuine bugs were fixed — including some that change
previously produced numbers **without raising**. Read *Changes that alter
results*, below, before re-running an old script.

### Import name

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

### Renamed modules

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

### Renamed methods

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

### The ambiguous class name

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

### Randomness

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

### Changes that alter results

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

### New in 0.2.0

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
