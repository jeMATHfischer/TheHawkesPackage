# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Rename the public attributes `Events` → `events` and `Sim_num` → `n_simulated` (0.3.0).
  Deferred from 0.2.0 because every existing notebook cell touches them.
- `mcmc_sampler` currently proposes an unbounded random walk and never rejects out-of-domain
  proposals; the caller merely wraps the final draw. Harmless for periodic kernels, wrong for a
  non-periodic `base(x)`. Minimal fix: accept a `transform` callback and pass `domain.wrap` (0.3.0).

## [0.2.0] — unreleased

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

### Fixed

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

### Added

- `pyproject.toml` (hatchling), `LICENSE` (MIT), `README.md`, this changelog, and a `src/` layout.
- `py.typed` marker — the package ships inline type information.
- `intensity` and `intensity_over_interval` on both spatio-temporal classes. Previously there was no
  way to evaluate the field intensity without re-implementing it by hand.
- A `HawkesProcess` / `TemporalHawkesProcess` base class carrying the shared Ogata thinning loop.
- Test suite expanded with domain-contract, periodic-kernel, deprecation and statistical
  correctness tests, at a 90% coverage gate.
- CI (lint, 3.9–3.13 on Linux and Windows, coverage, wheel-import check), a trusted-publishing
  release workflow, and a Sphinx documentation site.

## [0.0.1] — 2019-03-20

Initial internal version. Never published.
