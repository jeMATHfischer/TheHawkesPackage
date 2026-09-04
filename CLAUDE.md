# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`CONTRIBUTING.md` is the normative rulebook. This file is the orientation layer: it
says what the rules *cost you if you break them* and points at the rulebook for the
full statement. Where the two overlap, `CONTRIBUTING.md` wins.

## What this package is

Simulation of temporal and spatio-temporal Hawkes processes via Ogata's thinning
algorithm. A Hawkes process is self-exciting: every event raises the probability of
further events for a while afterwards.

```
λ(t | H_t) = φ( μ + Σ_{t_i < t} κ(t − t_i) )
```

Background rate `μ`, excitation kernel `κ`, monotone increasing nonlinearity `φ`
(the identity in the linear case). The sum runs over `t_i < t` **strictly**, so the
intensity reported *at* an event time is the left limit. The spatio-temporal
intensity is separable, `λ(t, x) = μ(x) + Σ κ_t(t − t_i)·κ_s(d(x, x_i))`, with `d`
the **geodesic distance on the domain** — the shorter arc on a `Circle`, the wrapped
Euclidean distance on a `Torus2D`.

**Scope boundaries.** Simulation, plus — since 0.5.0 — **inference**, which lives
entirely in `hawkes_package.inference` and is built on the same intensity hooks the
simulator thins against. Still out of scope: multivariate or mutually-exciting
processes, and discrete marks — space is the only mark. Partially observed data is
out too, and for a reason rather than by omission: it creates a genuine latent
state and needs a different algorithm, not a different setting of this one.
`hawkes_package.mcmc` remains the spatial location sampler on the Ogata correctness
path and nothing else; inference has its own chain in `inference/mcmc.py`, named so
it cannot read as a drop-in. Adding anything in the out-of-scope list is new
territory, not a gap to fill in by analogy with what is already here.

**Visualization** (`hawkes_package.viz`, 0.5.0) is a third area, and was a
deliberate scope addition rather than a roadmap item — it is not on the `###
Planned` list. It draws four of the closed surfaces, colours them by the
intensity and animates them to one HTML page. It is bound by the same rule as
inference: the field is evaluated through the simulator's hooks, never from a
second expression. Its backend is optional and lives behind one lazy import.

**Three names for one project**: the distribution is `the-hawkes-package`, the
import name is `hawkes_package`, and `TheHawkesPackage` was a deprecated import
shim, removed in 0.4.0. Import `hawkes_package`.

Theory, including the full bound argument: `docs/theory.md`.

## Setup and commands

The `src/` layout means **the package must be installed before it is importable**.
`conftest.py` deliberately carries no `sys.path` hack — do not add one. A fresh
clone running `pytest` fails with an import error that reads like a broken repo; the
fix is to install.

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

```bash
pytest                       # everything
pytest -m "not slow"         # the fast suite, ~25 s — what the CI matrix runs
pytest --cov                 # adds the 90% coverage gate
ruff check . && ruff format --check .
mypy
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`mypy` and `pytest` take no path argument — configuration drives both (`mypy` is
pinned to `files = ["src"]`, `strict`).

Running one test, and defeating the random ordering while iterating on a failure:

```bash
pytest tests/test_domains.py::TestFundamentalDomain -q
pytest tests/statistical/test_thinning_invariant.py -q -p no:randomly
```

The docs build is a real check, not a formality: `docs/conf.py` reads the version
from **installed** distribution metadata, and every notebook under
`docs/examples/` executes on every build under `-W`. An API change that breaks an
example fails the Docs job. Reproduce it with the `sphinx-build` line above.

Pre-commit is not run by any workflow — CI re-runs only `ruff` and `mypy`. So
`nbstripout`, `validate-pyproject` and the large-file guard are local-only
guarantees; run `pre-commit run --all-files` before pushing.

## Architecture

**The two-hook contract.** Every process supplies an intensity hook —
`_conditional_intensity` (temporal) or `_integrated_intensity` (spatio-temporal) —
plus `_upper_bound`, and inherits the Ogata thinning loop. `TemporalHawkesProcess`
in `src/hawkes_package/base.py` owns that loop for the three temporal classes;
`SpatioTemporalHawkesProcess` implements its own `_propagate` because it thins
against a *space-integrated* intensity. Throughout the spatio-temporal code a
`bound=True` flag switches the same helpers (`_temporal_factors`, `_dist_spatial`)
between the value and its per-event future supremum.

| Path | Go here for |
|---|---|
| `src/hawkes_package/base.py` | the Ogata loop, `simulate`, `events`/`n_simulated`, `_EventBuffer`, `SeedLike` |
| `exponential.py`, `monotone.py`, `bell_shape.py` | the three temporal classes and their bounds |
| `mcmc.py` | random-walk Metropolis–Hastings — the spatial location sampler only |
| `_numerics.py` | `as_point` (the coordinate contract), `locate_peak` |
| `spatio_temporal/domains.py` | `SpatialDomain` ABC, `Circle`, `Torus2D`, `FundamentalDomain` |
| `spatio_temporal/_integration.py` | `TensorQuadrature`, `build`, `restrict`, `check_resolution` |
| `spatio_temporal/process.py` | the domain-aware simulator |
| `inference/likelihood.py` | `History`, the three log-likelihoods, `_bind_history` |
| `inference/smc.py` | the IBIS loop, `ParticleCloud`, `SMCDiagnostics` |
| `inference/models.py` | `ProcessModel` — theta to a process, and where it is defined |
| `inference/_geometry.py` | the theta-independent distance tensors |
| `inference/_compensator.py` | panelled Gauss–Legendre for `∫ λ` |
| `viz/_embedding.py` | chart → R³, one immersion per surface, derived from its own pairings |
| `viz/_field.py` | the hoisted λ frames, the colour range, the event fade |
| `viz/_plotly.py` | the only module that names a plotting library, inside a function body |

**Geometry and quadrature.** `SpatialDomain` requires `distance`, `wrap`,
`sample_uniform`, `volume` and `bounds`; it optionally supports `contains`,
`volume_element`, `orbit` and `interior_point`, all with backwards-compatible
defaults. `restrict()` masks the tensor Gauss–Legendre rule by `contains` and
reweights the surviving nodes by `volume_element`, which is what lets a domain be a
**proper subset of its bounding box**. This does not weaken the Ogata bound: that
argument only ever required the bound and the acceptance test to share one node set
with strictly positive weights, never that the nodes fill a box. `_confined_density`
returning 0 off-domain is what confines the location sampler to such a domain.

**`periodic` is a correctness switch, not an optimisation.** When it is true the
sampler folds proposals through `wrap` (`transform=domain.wrap`), which is
reversible only where the deck group acts by *translations*. `FundamentalDomain`
sets `periodic = False` deliberately, because the Möbius pairings that hyperbolic
domains will need do not satisfy that. Rejection costs only mixing near the
boundary and is correct for any pairing.

**Kernel protocol.** `temporal` receives a non-negative lag and `spatial` a
non-negative distance — unless the kernel carries `pairwise = True`
(`PairwiseKernel`), in which case it receives both endpoints, which is what image
sums need. Every user callable is handed a shape-`(ndim,)` point via `as_point`.

## Invariants that fail silently

This is the section that matters. None of the following raises when violated.

- **`M >= λ` is the whole game.** A bound that is too *small* makes every candidate
  accepted, so the output is a Poisson process wearing a Hawkes costume. Seven such
  defects shipped during 0.2.0, and every one lived in a configuration the test
  harness did not then reach. **Add a case to
  `tests/statistical/test_thinning_invariant.py` before changing how a bound is
  computed**, not after. Two recurring root causes worth checking first: an
  unvalidated numerical peak search, and a *randomised* quantity used as a bound (a
  Monte Carlo estimate is unbiased, not dominating).
- **Never define an intensity accessor separately from the hook the simulator thins
  against.** That is exactly how `ExponentialHawkes` came to plot a curve a constant
  `mu` below its own intensity.
- **Never call `np.random.seed`; pass `rng=`.** The suite runs under
  `pytest-randomly`, which shuffles order and reseeds globals between tests, so any
  dependence on the global stream surfaces as an intermittent failure.
  `tests/test_base.py` is the one deliberate exception, and it exists to prove the
  global stream no longer influences simulations.
- **`filterwarnings = ["error"]`, with no blanket exemptions.** Intentional warnings
  must be wrapped in `pytest.warns`; a module reading `process.Events` at import
  time errors at collection.
- **Exact-value reproducibility may be asserted only for the three temporal
  classes.** The spatio-temporal path branches on floating-point comparisons whose
  last bits vary across SciPy and BLAS builds; a different branch consumes a
  different number of draws and the streams diverge. Assert distributional and
  structural properties there instead.
- `xfail_strict` is on, so `pytest.mark.xfail(..., strict=True)` for a bug you are
  about to fix cannot rot — the suite fails when it starts passing.
- **A behaviour change that alters previously produced numbers without raising is
  the worst failure mode in a scientific package.** It goes under `### Changed` or
  `### Fixed` in `CHANGELOG.md` *and* into `docs/migration.md`, in terms a user can
  act on.
- **The package currently carries no deprecations**, and `_deprecation.py` was
  deleted with the last of them in 0.5.0 — an unused deprecation helper reads as
  supported machinery. Adding one means recreating the module, and adding the
  old/new pair to the removal tables in `tests/test_deprecations.py`, which
  `docs/migration.md` is written from so the documentation cannot drift from
  behaviour.
- Simulation is O(n²) in the intensity sum: eight events on a 2-D domain already
  takes ~12 s, and 60 events on a `Circle` takes ~65 s. Budget test sizes
  accordingly, and reach for `@pytest.mark.slow` for anything over a second. Note
  which side is slow: a spatio-temporal *fit* of 60 events is 3 s against 65 s to
  generate them.
- **A compensator computed too small is the inference-side twin of a bound
  computed too small.** Every unit of `∫ λ` that goes missing is a penalty on a
  high intensity that never gets applied, so `mu` and the excitation both come
  back too large and the fit looks converged. Guarded by exact quadrature at the
  event jumps, an order-`P`-versus-`2P` resolution check, and the time-rescaling
  test — and the residuals for that test must use a compensator that does *not*
  share the estimator's bug, because a fit made with a compensator 20% too small
  inflates the intensity by 25% and the two errors cancel exactly.
- **`SpatioTemporalLogLikelihood`'s cached backend has a precondition, and it
  raises rather than degrading.** `_full_intensity` floors *after* summing, so the
  separability identity `∫_D λ = ∫_D μ + Σ κ_t·S_i` holds only where the pre-floor
  integrand is non-negative at every node. `backend="hooks"` is the normative
  definition; `"auto"` falls back to it once, with a warning, and records what ran.
- **Particle degeneracy reads as confidence.** A collapsed cloud reports a very
  tight posterior centred wherever the resampling noise left it. Neither obvious
  diagnostic catches a frozen rejuvenation kernel: the effective sample size is
  perfect for N copies of one particle, and the *acceptance rate* reads 1.000,
  because a proposal scaled to 1e-12 proposes the point it starts from.
  `StepRecord.move_size` — the distance travelled in units of the cloud's own
  width — is what tells them apart, 0.5 against 5.7e-12.
- **Never assert a statistical threshold without sweeping seeds 0–20 first.** The
  sweep for `test_ibis_agrees_with_an_independent_metropolis_run` is what found
  that the Metropolis proposal could shrink irrecoverably: the step reached
  2e-159, every proposal became the point it started from, the acceptance rate
  read 1.000 and the chain returned 40 000 copies of one sample while looking
  healthy.

## Conventions

- **The frozen public names were renamed in 0.4.0** and ruff's `N` (pep8-naming)
  is selected again, which is what keeps them renamed. `Events` → `events`,
  `Sim_num` → `n_simulated`, `L1`/`L2` → `width`/`height`; `Base` and `Space`
  went with the legacy class. Each old spelling survives as a
  `DeprecatedAttribute` until 0.5.0 — including its setter, because
  `process.Events = history` is how a realisation is seeded and a read-only
  alias would let that assignment silently shadow the real attribute.
- `TID` is also off on purpose: `from ..mcmc import mcmc_sampler` is the sanctioned
  way for a subpackage to reach a sibling.
- Python floor 3.10; `from __future__ import annotations` in every module; PEP 604
  unions and PEP 585 generics; `collections.abc.Callable`.
- mypy `strict` over `src`, and `py.typed` ships — downstream typing must not
  regress, and since 0.4.0 there is no module it is loosened for.
- numpydoc docstrings, line length 100. `D105`/`D107` are ignored because
  constructor parameters are documented on the class. Carry `.. versionadded::` /
  `.. versionchanged::` / `.. deprecated::` directives with the version.
- Runtime dependencies are **numpy and scipy only** — scipy in exactly one place
  (`minimize_scalar` in `_numerics.py`), which is why `ks_exponential` hand-rolls
  the Kolmogorov series rather than importing `scipy.stats`. Tests may use scipy
  freely, and check the hand-rolled versions against it. Do not reach for pandas, numba or jax;
  matplotlib is a docs extra, not a runtime dependency, and **plotly is a `[viz]`
  extra** on the same footing: `viz/_plotly.py` imports it inside `_backend()` and
  nothing else in `src` names it, so `import hawkes_package.viz` works without it
  and a render says what to install. `tests/viz/test_public_surface.py` checks that
  in a subprocess. NumPy 2 compatibility is an active concern.
- `ValueError` for bad construction input, `RuntimeError` for a simulation that
  cannot proceed (non-positive bound, explosion, sampler that cannot find support),
  `UserWarning` for silently-wrong-numerics risk. Messages state the offending value
  and what to do about it.
- **Comment culture is the strongest local convention.** Non-obvious lines explain
  *the bug they prevent*, usually with the measured failure rate. The module
  docstrings of `_numerics.py` and `spatio_temporal/_integration.py` are the model.
  Match that density rather than writing what-not-why comments.
- **Conventional Commits**, lowercase imperative subject, no trailing period:
  `feat:`, `fix:`, `docs:`, `chore:`, `chore(release):`, `ci:`, `build:`, `style:`,
  `refactor:`, `feat!:` for a breaking change. The body quantifies the defect and
  names what would have failed silently.
- Branch from `master`; linear history is required; keep formatting changes in their
  own commit; add a `CHANGELOG.md` entry under `[Unreleased]` for anything
  user-visible.

## Where it pushes to

| Target | Trigger | Mechanism |
|---|---|---|
| `github.com/jeMATHfischer/TheHawkesPackage` (`master`) | manual push | HTTPS |
| GitHub Pages — `jeMATHfischer.github.io/TheHawkesPackage/` | push to `master` | `docs.yml`, environment `github-pages`; PRs build but never deploy |
| PyPI `the-hawkes-package` | push of a **final** `v*` tag | `release.yml`, OIDC trusted publishing, environment `pypi` |
| TestPyPI | push of a **prerelease** tag (`v0.2.0rc1`) | `release.yml`, environment `testpypi`, `skip-existing` |
| GitHub Releases | after whichever publish job ran succeeds | notes cut from `CHANGELOG.md` |

A push to `master` deploys the documentation. That Pages deploy is the only
outward-facing effect of a plain push — **nothing publishes a package except a tag.**

Releasing (runbook in `CONTRIBUTING.md § Releasing`):

1. Move the `[Unreleased]` entries under a new version heading in `CHANGELOG.md`.
2. Bump `__version__` in `src/hawkes_package/__init__.py` — the single source of
   truth; hatchling reads it from there.
3. Tag `vX.Y.Z` and push.

Two gates stop a mistagged release *before* it uploads: the build job asserts
tag == wheel == sdist version, and both trusted publishers are bound to the
literal filename `release.yml`.

- **The `CHANGELOG.md` check is not one of them.** The `github-release` job
  hard-fails when there is no `## [<version>]` section, but it is
  `needs: [build, publish-testpypi, publish-pypi]` — it runs *after* the upload,
  so on a final tag PyPI has already accepted the files and the version is burnt.
  The rc rehearsal does not cover it either: the prerelease path falls back to a
  "Rehearsal build of …" note and passes. **Verify the section by reading
  `CHANGELOG.md` before tagging**, not by a green rc.

- **Never rename `release.yml`.** Both publishers are bound to that filename, and
  renaming it silently breaks every future release on both indexes.
- Tags must be canonical PEP 440 with a leading `v`. `v0.2.0-rc1` and `v0.2.0RC1`
  are rejected by design.
- There is no `workflow_dispatch`, deliberately: every upload traces to a tag.
- **A published version can never be re-uploaded, on either index.** A broken
  release is yanked and superseded, never replaced.
- Required status checks are matched on **display names**, so renaming any workflow
  `name:` or job `name:` silently breaks branch protection (fourteen checks, listed
  in `CONTRIBUTING.md`).
- `ci.yml` does not run on tag pushes; `release.yml`'s `verify` job is the tag-time
  equivalent and runs `pytest -m "not slow"`. The slow tests run only in `ci.yml`'s
  coverage job — on the master push, never on the tag.

## Current state and roadmap

Released: **0.5.0** — `hawkes_package.inference`,
`HawkesProcess.simulate_until`, `HawkesEstimator`, `hawkes_package.viz`, and the
removal of every name 0.4.0 dated for that release. 0.4.0 before it reached every
closed surface through `FundamentalDomain` and the three constant-curvature model
spaces. The plan the inference work was built from is
`docs/plans/bayesian_module.md`; where the plan and the code differ, the code
records the measurement that decided it.

Nothing is outstanding. What follows is the roadmap **past** 0.5.0 — each item
is also under `### Planned` in `CHANGELOG.md`, which is the source:

- **The quadratic term is the intensity, not the record.** 0.4.0 replaced
  `np.append` with a doubling buffer — 7× faster at 5 000 events, 17× at 50 000,
  and linear rather than quadratic — and it changed no simulation's running time
  measurably, because the record was never where the time went. The remaining
  `O(n²)` is the intensity sum, `O(n)` per thinning step. Making it incremental
  is possible for the exponential kernel and is the real fix; do not reach for
  the buffer again.
- **Hyperbolic surfaces stop at twelve sides** — genus 3, six crosscaps — and
  `_MAX_HYPERBOLIC_SIDES` refuses the rest at construction. The limit is the
  deck-group search, not the geometry: a certified `distance` enumerates every
  element within a radius that scales with the polygon, and the element count
  grows like `exp(R)`, so genus 3 certifies a 193-element answer by visiting
  tens of thousands and genus 4 cannot be searched at all. Searching outward
  from the *pair of points* rather than from the polygon's centre would size the
  work to the answer; a bigger budget would not.
- **`contains` must never read the grown window.** The boundary convention takes
  the orbit of a boundary point and keeps its lexicographic minimum, and
  `distance` widens the window on demand — so reading the current window makes a
  pure predicate depend on search history, and two callers sharing a domain get
  different answers depending on order. `_boundary_window` is snapshotted at
  construction for exactly this reason. It shipped wrong once and the CI matrix
  caught it by failing five of ten jobs on identical code.
