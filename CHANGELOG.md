# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Remove the aliases 0.4.0 deprecated — `Events`, `Sim_num`, `L1`, `L2`, and the `n_images`
  argument of `FundamentalDomain` — in 0.5.0. `REMOVED_IN` in `_deprecation.py` is the date.
- Make the intensity incremental for the exponential kernel, which is the classic `O(n)` Hawkes
  simulation. This is now the whole of the quadratic term: 0.4.0's buffer made the event record
  grow linearly, and measuring showed the record was never where the time went.
- Hyperbolic surfaces past twelve sides — genus 4, seven crosscaps — are refused at
  construction, and reaching them needs a different search rather than a bigger budget. A
  certified distance enumerates a deck-group window whose size grows like `exp(R)`, and the
  radius scales with the polygon; the answer stays tiny (193 elements for genus 3) while the
  search that certifies it does not. Searching outward from the *pair of points* instead of
  from the polygon's centre would size the work to the answer.
- Double precision is a second ceiling behind that one: a deck element at displacement 18 has
  hyperboloid coordinates near `5e7`, where the spacing of doubles exceeds the gap between the
  sheet and its asymptotic cone.

## [0.4.0] — 2026-08-27

The breaking release every deferred rename was deferred *to*. Two halves: the
removals and renames below, and the surfaces work that made the release worth
cutting now.

### Removed

- **The `TheHawkesPackage` import shim.** `import TheHawkesPackage` is an
  `ImportError`; the import name is `hawkes_package`. Deprecated since 0.2.0.
- **The three `simulate` aliases** — `propagate_by_amount`,
  `propagate_by_k_events` and the `propogate_by_amount` typo. `simulate(k)` is
  the method.
- **`Spatio_Temporal_Hawkes_Process`**, the top-level name that used to mean two
  different classes depending on the import path.
- **`LegacySpatioTemporalHawkesProcess`** and `hawkes_package.spatio_temporal.legacy`,
  frozen since 0.2.0 for bit-compatibility with results published before it.
  `SpatioTemporalHawkesProcess` on a `Circle` is the replacement, and it is not
  bit-compatible: it integrates deterministically where the legacy class used
  Monte Carlo. Its `Base`/`Space` argument spellings go with it.
- `DeprecatedAlias` and `deprecated_module_getattr` from the internal
  `_deprecation` module, which lost their last callers. An unused deprecation
  helper reads as supported machinery, which is worse than none.

### Added

- **Every closed surface.** `FundamentalDomain` is no longer restricted to flat orientable
  quotients: a polygon may now live in any of three constant-curvature model spaces, and the
  sign of the Euler characteristic decides which. New presentations:
  `FundamentalDomain.klein_bottle`, `FundamentalDomain.projective_plane`,
  `FundamentalDomain.genus(g)` for the orientable surface of genus `g`, and
  `FundamentalDomain.crosscaps(k)` for the non-orientable surface `N_k`. Together with the new
  `Sphere` domain and the existing `rectangle`/`hexagon`, that is the whole classification of
  closed surfaces.
- **`Sphere`**: the round 2-sphere, the one closed surface that is not a quotient — it is simply
  connected, so a deck group is the wrong tool for it. It needs no machinery beyond the curved
  measure, which is why it also serves as the end-to-end proof that the curved-measure path works.
- **`FundamentalDomain.topology`** reports the orientability, Euler characteristic, genus and
  plain-language name of the surface a presentation actually glues to — read off the corner
  cycles, not declared by the caller. `FundamentalDomain.cycles` exposes the cycles themselves.
- `SpatialDomain` gains three optional hooks, all with backwards-compatible defaults: `lift_distance`
  (default the chart norm — the distance between two *lifts*, which an image sum needs and the
  quotient distance is not) and `max_distance` (default the box half-diagonal — an upper bound on
  `distance`, which the chart cannot supply on a curved domain), plus `nodes_per_axis`, the
  quadrature resolution the domain knows it needs.
- `hawkes_package.spatio_temporal.kernels.check_image_sum` warns when a periodised kernel is
  truncated where it has not yet decayed.

### Changed

- **`Events` is now `events`, and `Sim_num` is now `n_simulated`.** Both old
  spellings still work, warn, and are removed in 0.5.0 — including assignment,
  because `process.Events = history` is how a realisation is conditioned on
  events it did not simulate, and an alias that only supported reading would let
  that assignment silently shadow the real attribute.
- **`Torus2D(L1=, L2=)` is now `Torus2D(width=, height=)`**, as arguments and as
  attributes, and likewise for `FundamentalDomain.rectangle` and
  `FundamentalDomain.klein_bottle`. Old spellings warn and work until 0.5.0; an
  unrecognised keyword still raises, so a typo cannot silently fall back to the
  default.
- With the last of the frozen names gone, ruff's `N` (pep8-naming) is selected
  again. That was the point of doing the rename, and it is what stops the names
  coming back.
- **The event record grows by doubling** instead of reallocating on every
  accepted event, and `events` is a view onto it. In isolation that is 7x faster
  at 5 000 events and 17x at 50 000, and linear rather than quadratic — but it
  changes no simulation's running time measurably, because the record was never
  where the time went: the intensity sums are themselves `O(n)` per thinning
  step and dominate by two orders of magnitude. What it removes is a term that
  would become dominant the moment the intensity is made incremental. The
  event-lookup on the spatio-temporal path was vectorised at the same time,
  19x faster at 5 000 events and equally invisible end to end.
- Existing scripts produce **identical numbers**. The rename and the buffer
  consume the same draws in the same order; the event times of all four
  simulators are bit-identical to 0.3.0.

- **Orientation-reversing side pairings are accepted.** Through 0.3.0 a pairing with determinant
  `-1` raised. That was a policy rather than a correctness check, and it cost the entire
  non-orientable half of the classification. What replaces it is the check that was actually
  missing — freeness — which a glide reflection passes and a pure reflection does not.
- **A presentation is validated at construction, against Poincaré's polygon theorem.** The side
  correspondence must be complete, every corner cycle must close with an interior-angle sum of
  exactly `2*pi` and a trivial cycle transformation, and Gauss-Bonnet must tie the area, the
  curvature and the Euler characteristic together. Previously a bad presentation surfaced — if at
  all — as a `ValueError` from `wrap`, mid-simulation and far from the mistake. Code that built
  a domain which never in fact tiled now fails at the constructor instead of later or never.
- **`FundamentalDomain.distance` truncates the deck group by displacement radius, certified per
  call**, rather than by word length. Both points are reduced into the polygon first, after which
  the triangle inequality bounds the displacement of any element that could improve on the best
  distance found; once the window covers that, the answer is the exact minimum. Word length was a
  heuristic, and on a hyperbolic surface — where the element count grows like `exp(R)` — the
  difference is between a kernel that stays periodic and one that quietly decays to zero. Flat
  domains produce identical numbers: the old window already contained the minimiser.
- `make_periodic` sums over `lift_distance` rather than the chart norm. Identical on every flat
  domain, where the chart *is* the universal cover; on a curved one the chart norm is not a
  distance at all. It also now warns when the last ring of images still contributes more than 1%
  of the sum, which on a hyperbolic surface is the difference between a convergent image sum and
  a silently truncated one.
- **`contains` no longer depends on how far `distance` has searched.** The boundary convention
  reads the orbit of a boundary point out of the deck-group window, and `distance` widens that
  window on demand — so a corner that was the representative of its cycle could stop being one
  because an unrelated earlier call had widened the search. Shared domains made it
  order-dependent, and the CI matrix duly failed on five of ten jobs and passed on the other
  five with identical code. The window used for the boundary is now the one the polygon was
  built with, and is never replaced.
- **The deck-group search budget counts the answer and the search separately.** They differ by
  orders of magnitude in negative curvature — genus 3 certifies a 193-element window by visiting
  tens of thousands — and counting them against one cap rejected a legitimate surface for a cost
  its answer never incurs, from inside `distance`, on whichever pair of points happened to need
  the wider search first.
- The boundary convention is now "one representative per orbit, chosen lexicographically" rather
  than a closed/open flag per side. The flag rule reproduces `Torus2D`'s `[-L/2, L/2)` convention
  on the rectangle and the hexagon and is wrong in general: on the projective plane's hemisphere
  every assignment of flags leaves one corner cycle with two representatives and the other with
  none. Which boundary points a flat domain admits is unchanged.
- `SpatioTemporalHawkesProcess` takes its default `n_quad` from the domain rather than from the
  dimension. A flat polygon still asks for 32 nodes per axis; a hyperbolic one asks for 128,
  because 32 misses its area by 5% — and mismeasuring the area scales the simulated event rate
  by exactly that factor.

### Fixed

- **The location sampler was handed the intensity without the measure.** The event location is
  distributed as `lambda dA` on the surface, but `mcmc_sampler` walks in *chart* coordinates with
  a symmetric proposal and accepts on the raw ratio, so the density it must be given is
  `lambda * volume_element` — the same factor `restrict()` already applied to the quadrature
  weights, and which the sampler was never given. **No previously produced number moves**: every
  domain that had shipped carried `volume_element == 1`, so the factor is identically one on all
  of them. It is not one on the first curved domain, where the omission would have piled events
  wherever the chart compresses area — at the poles, on a sphere.
- **Freeness of the side pairings was never checked.** A rotation pairing is an isometry with
  determinant `+1`, so it passed every test the package made; it has a fixed point, and quotients
  to an *orbifold* — a cone point — rather than to a surface. Nothing downstream could tell the
  difference. Pairings are now classified and a non-free one is refused, with the motion named.

### Deprecated

- The `n_images` argument of `FundamentalDomain`, removed in 0.5.0. It tuned a truncation that
  now bounds itself, so nothing replaces it; `FundamentalDomain.orbit` still takes its own
  `n_images`, and the attribute still exists.

## [0.3.0] — 2026-08-26

### Added

- **`FundamentalDomain`**: a convex Euclidean polygon together with the side-pairing isometries
  that identify its boundary, presenting a flat orientable surface. `FundamentalDomain.hexagon`
  gives the hexagonal torus — the first quotient in the package that no rectangular domain
  expresses — and `FundamentalDomain.rectangle` reproduces `Torus2D` through the general
  machinery, which is how that machinery is checked. Only orientation-preserving pairings are
  accepted; a reflection or glide raises.
- `SpatialDomain` gains three optional hooks and one property, all with defaults that leave an
  existing subclass behaving exactly as before: `contains` (default `True` — the domain fills its
  bounding box), `volume_element` (default `1.0` — the flat chart), `orbit` (default `None` — no
  deck group) and `interior_point` (default the centre of `bounds`).
- `make_periodic` periodises any domain that implements `orbit`, by summing the kernel over the
  image points. `Circle` and `Torus2D` keep their existing hand-written branches unchanged.

### Changed

- **A spatial domain may now be a proper subset of its bounding box.** Integration masks the
  quadrature rule by `contains` and weights it by `volume_element`. The Ogata bound is unaffected:
  it needs the bound and the acceptance test to share one node set with strictly positive weights,
  which masking and a positive metric factor both preserve, so `M >= lambda` still holds pathwise.
- `SpatioTemporalHawkesProcess` no longer raises when `volume != prod(bounds widths)`. In its place
  the summed quadrature weights are checked against the domain's declared `volume`: more than 1%
  apart warns that the rule does not resolve the domain boundary and the event rate will be wrong
  by about as much; more than 10% apart still raises, since that means `volume`, `bounds` and
  `contains` describe different regions. For a domain that fills its box nothing is masked and the
  weights sum to the box volume exactly, so the old guarantee is subsumed rather than dropped.
- The location sampler targets the intensity restricted to the domain, rather than to its bounding
  box. On a domain that is a proper subset this matters: `_full_intensity` off the domain is the
  periodic extension, so the box covers parts of the domain twice and others once, and folding a
  box-distributed draw back in inherits that unevenness.
- Existing scripts produce identical numbers: `Circle` and `Torus2D` fill their bounding boxes, so
  nothing is masked and no extra draw is consumed. `docs/migration.md` says what changed for the
  author of a custom `SpatialDomain` subclass, which is the only audience this release asks
  anything of.

## [0.2.0] — 2026-08-26

First packaged release. The distribution is `the-hawkes-package`; the import name is
`hawkes_package`.

### Changed

- **Breaking: `np.random.seed(...)` no longer controls simulations.** Every process now takes
  `rng=`, accepting `None`, an `int` seed, or an existing `numpy.random.Generator`, and draws from
  that stream only. Replace `np.random.seed(42); ExponentialHawkes(param)` with
  `ExponentialHawkes(param, rng=42)`.
- **Import name is now `hawkes_package`.** `import TheHawkesPackage` still works and forwards to
  identical objects, but emits a `DeprecationWarning`. The shim is present in 0.2.x and 0.3.x and
  **removed in 0.4.0**.
- Modules renamed to PEP 8 snake_case: `ExponentialHawkes.py` → `exponential.py`,
  `MonotoneKernelHawkes.py` → `monotone.py`, `BellShapeHawkes.py` → `bell_shape.py`,
  `MCMC_sampler.py` → `mcmc.py`, `SpatioTemporal_Hawkes_Monotone.py` →
  `spatio_temporal/legacy.py`. The old dotted paths keep working through the shim.
- `simulate(k)` is the canonical method on every process class. `propagate_by_amount`,
  `propagate_by_k_events` and the `propogate_by_amount` typo remain as aliases that emit
  `DeprecationWarning`; they are removed in 0.4.0.
- `hawkes_package.spatio_temporal.Spatio_Temporal_Hawkes_Process` is **removed**. The name
  previously meant the domain-aware class in this subpackage but the legacy periodic-interval class
  at top level — one identifier, two different algorithms and two different `Events` shapes.
  Accessing it from the subpackage now raises `AttributeError` instead of silently resolving to
  whichever class the import path picked.
- `spatio_temporal/sampler.py` removed; it was a re-export of `mcmc_sampler`, still reachable at
  `hawkes_package.mcmc.mcmc_sampler`.
- The vestigial `PoissEvent` attribute is gone from both spatio-temporal classes. It accumulated
  exponential draws whose values were never read — only their count mattered — so it consumed
  randomness to no effect.
- `Events` starts **empty** rather than holding a fictitious event at `t = 0`, so
  `process.Events[-1]` before the first `simulate` now raises `IndexError` instead of returning 0.0.
- User callables (`base`, `spatial`) always receive a shape-`(ndim,)` point, on every code path.
- The spatial integral is a deterministic quadrature rule at every dimension, replacing both
  `scipy.integrate.quad` and Monte Carlo. `SpatialDomain` implementations must now satisfy
  `volume == prod(bounds widths)`, and gain a `periodic` flag (default `False`) that governs whether
  MCMC proposals may be folded rather than rejected.

### Fixed

- **Four invalid Ogata thinning bounds, all of which silently biased the simulated distribution.**
  Thinning is only correct while `M >= lambda`; where it fails, candidate events are accepted that
  should have been rejected — silently, since a too-tight bound raises nothing. None of these were
  reachable by the pre-0.2.0 tests, which exercised only `Circle()`, monotone temporal kernels and
  non-negative spatial kernels; the invariant harness now covers a delayed kernel, `Torus2D`, a
  sign-changing spatial kernel and `make_periodic`.
  - `BellShapeHawkes` added a single peak's worth of headroom to the whole intensity. That is not
    enough when two or more events are in their rising phase at once, and the invariant failed in
    roughly 5% of steps. Each event is now bounded by its own future supremum — the peak value if it
    has not yet peaked, its current value if it has.
  - `SpatioTemporalHawkesProcess` and the legacy class excluded the event at exactly `t` from the
    bound, which is precisely the `MonotoneKernelHawkes` bug the codebase already documented,
    reproduced in the newer classes. At the start of a thinning step `t` *is* the most recent event
    time, so its entire excitation was missing from the bound; the invariant failed in about 70% of
    steps. Both classes now integrate the per-event suprema, which also removes a dimensionally
    inconsistent correction term that added a bare temporal-kernel value to a space-integrated
    intensity.
  - The kernel's peak was located by `scipy.optimize.fmin` started at lag 0, with no validation of
    the result. On a kernel that is flat near zero — the standard delayed-excitation shape — it
    returned 0, so the peak value collapsed to `temporal(0) = 0` and the bell-shaped bound silently
    degraded to the monotone one. The invariant failed in **46.3%** of steps, worst excess 4.278.
    Replaced by a global scan with an adaptively expanded window; `peak_lag=` bypasses it.
  - For a domain of two or more dimensions the spatial integral was a 500-point Monte Carlo estimate
    redrawn on every call, so the bound was an unbiased estimate rather than an upper bound — and
    the acceptance test drew a second, independent estimate to compare against. On `Torus2D`,
    `P(lambda_hat > M_hat) = 0.437` where Ogata's algorithm requires 0. It also stole 500·ndim
    variates per evaluation from the stream driving the simulation. Replaced by a deterministic
    Gauss-Legendre tensor rule, which makes `M >= lambda` exact by construction and is ~8x faster
    than the `quad` it also replaces in one dimension.
  - The bound took `sup(kappa_t)` and multiplied by `kappa_s`, which is the supremum of the product
    only where `kappa_s >= 0`. With an inhibitory spatial kernel the invariant failed in 3 of 71
    steps. The spatial factor is now clipped at zero in bound mode, which *is* the correct supremum.

- **The phantom `t = 0` event.** `Events` was seeded with a fictitious event to bootstrap the first
  thinning step, and that event contributed to every intensity sum until the first `simulate` call
  deleted it. `E[T1]` was 12.77 where the model gives `1/mu = 20.0`, and `simulate(1); simulate(1)`
  differed from `simulate(2)` (mean second gap 28.17 against 22.58, KS p = 4.2e-06) despite the
  docstring promising they continue one realisation. `Events` now starts empty and holds only real
  events at every moment; `Sim_num` is counted per event, so it still agrees with `len(Events)` after
  a caught failure.
- **Every event location in the legacy class was drawn from the wrong density.** `spatial` applied to
  a shape-`(1,)` offset returns a shape-`(1,)` value, so the spatial factors formed an `(n, 1)`
  column; multiplied by the `(n,)` temporal factors that broadcasts to an `(n, n)` outer product, and
  the sum computed `(Σ kappa_t)(Σ kappa_s)` instead of `Σ kappa_t·kappa_s`. Since `mcmc_sampler`
  always passes an array, this was the sampling density — while the temporal thinning, fed scalars by
  `quad`, used the correct one. With three past events, `intensity(1.0, 0.15)` gave 1.191369 and
  `intensity(1.0, [0.15])` gave 1.717217.
- **No non-constant background could be written.** User callables received a Python float from the
  quadrature path and a shape-`(ndim,)` array from Monte Carlo and the MCMC sampler, so
  `base=lambda x: 0.5 + 0.2*np.cos(x[0])` raised on one path and
  `base=lambda x: 0.5 + 0.2*np.cos(x)` on the other. They now always receive a shape-`(ndim,)` point.
- **`make_periodic` could not be used as a `spatial` kernel.** It returns a two-point callable while
  `spatial` was called with a single geodesic distance, so passing it raised `TypeError` on the
  second event — although `README.md` presents it as the way to build a domain-respecting kernel.
  Such a kernel now declares itself with `pairwise = True`. Its image sum was also taken about the
  raw difference, so beyond `n_images` periods the nearest image fell outside the window and the
  kernel decayed to zero instead of staying periodic.
- **The MCMC chain was not confined to the domain.** The proposal was an unbounded random walk that
  never rejected an out-of-domain move; `space` bounded only the initial draw. Correct for a target
  periodic with the domain, wrong otherwise: with a non-periodic background the marginal was
  indistinguishable from uniform (chi-square p = 7.6e-07 against the true target, 0.21 against
  uniform), and on a domain whose `wrap` clips, **all** event locations landed on a boundary.
  Proposals outside `space` are now rejected; folding is opt-in through `transform=` and used only
  where `domain.periodic`.
- **`mcmc_sampler` could return silent garbage.** A failed search for a starting point fell through
  into `density(proposal) / density(x)` with a zero denominator: `ZeroDivisionError` for 2 of 30
  seeds with a Python float, and with a NumPy float `0/0 -> nan`, where `min(1.0, nan) == 1.0` made
  the chain accept every proposal. It now raises, and the acceptance test is written without
  division. `proposal_std` also defaults to a tenth of each axis's width rather than a fixed 1.0,
  which could not equilibrate on a wide domain (30.3% of draws in a peak holding 52.2% of the mass)
  and was wrong on both axes of an anisotropic one.
- `quad`'s error estimate was discarded while `IntegrationWarning` was silenced globally, so a failed
  integration was invisible: with a width-0.005 spatial kernel `quad` returned exactly the
  background-only value, making the excitation invisible to the temporal thinning while the spatial
  sampler still saw it. A construction-time resolution check replaces it.
- `Circle.distance` silently measured only the first component of a longer vector;
  `Torus2D.distance` raised on a scalar and on a `(2, 1)` column. `simulate(2.7)` truncated silently.


- **NumPy 2.x compatibility.** `float()` on a shape-`(1,)` array raises `TypeError` since NumPy 2.0,
  which broke three code paths that no test reached:
  - `SpatioTemporalHawkesProcess` failed in its constructor with the **default**
    `monotone_temporal_kernel=False`, because `scipy.optimize.fmin` returns a shape-`(1,)` array.
  - `make_periodic` on a `Circle` failed whenever it was given array coordinates — which is exactly
    how `SpatioTemporalHawkesProcess` calls it.
  - `BellShapeHawkes.ext` and the legacy class's `temporal_extremum` were shape-`(1,)` arrays rather
    than floats, so comparisons against them produced arrays.
- `ExponentialHawkes.intensity_over_interval` omitted the baseline `mu` from the returned intensity,
  while the thinning bound included it. Plots produced with 0.0.1 were shifted down by `mu`. The
  accessor and the simulator now share one `_conditional_intensity` implementation, so they cannot
  diverge again.
- Importing the spatio-temporal module no longer calls `random.seed(42)`, which silently reseeded
  the *caller's* global `random` module as a side effect of `import`.
- The legacy spatio-temporal process ignored a non-default `Space=`, hard-coding `[-pi, pi]` as the
  MCMC domain. It now honours the value passed.
- The mutable default argument `Space=[-np.pi, np.pi]` is now an immutable tuple, renamed `space=`.
  `Space=` is still accepted for one release with a `DeprecationWarning`.
- An exploding process no longer hangs. When the expected offspring count reaches one the intensity
  diverges, inter-arrival times underflow to exactly zero and time stops advancing, so `simulate`
  looped forever with no diagnostic. It now raises `RuntimeError` naming the cause. This is easy to
  trigger with a fast-growing nonlinearity — `nonlinearity=np.exp` over a unit-mass kernel is
  enough — and the documentation notebook that shipped with 0.0.1 did exactly that.

### Added

- `peak_lag=` and `peak_value=` on `BellShapeHawkes` and both spatio-temporal classes, to bypass the
  numerical peak search for a kernel with a spike narrower than the search grid.
- `n_quad=` on both spatio-temporal classes: quadrature nodes per axis.
- `proposal_std=` and `n_iter=` on `SpatioTemporalHawkesProcess`, forwarded to the spatial sampler.
- `x0=`, `transform=` and `max_init_tries=` on `mcmc_sampler`.
- `PairwiseKernel` and the `pairwise = True` protocol, so a kernel can consume both endpoints
  instead of a geodesic distance.
- `pyproject.toml` (hatchling), `LICENSE` (MIT), `README.md`, this changelog, and a `src/` layout.
- `py.typed` marker — the package ships inline type information.
- `intensity` and `intensity_over_interval` on both spatio-temporal classes. Previously there was no
  way to evaluate the field intensity without re-implementing it by hand.
- A `HawkesProcess` / `TemporalHawkesProcess` base class carrying the shared Ogata thinning loop.
- Test suite expanded with domain-contract, periodic-kernel, deprecation and statistical
  correctness tests, at a 90% coverage gate.
- CI (lint, 3.10–3.14 on Linux and Windows, coverage, wheel-import check), a trusted-publishing
  release workflow, and a Sphinx documentation site.

## 0.0.1 — 2019-03-20

Initial internal version. Never published.

[Unreleased]: https://github.com/jeMATHfischer/TheHawkesPackage/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.4.0
[0.3.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.3.0
[0.2.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.2.0
