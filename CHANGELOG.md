# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Power-law and compact-support kernels.** `OmoriUtsuKernel` is
  `alpha (s + c)**-p`, the applied standard for aftershock decay; `ParetoSpatial` is
  `(r**2 + d**2)**-q` in space; both are normalised or parameterised so the existing
  branching bound keeps working unchanged. The package had only light tails before, and the
  difference is not decorative: matched at the peak and at the branching ratio, the power law
  at lag 20 is more than a **million times** the exponential's value, which is where a real
  catalogue's late aftershocks live.

  Both bound their exponent where the integral *converges* — `p > 1` in time, `q > ndim/2`
  in space — and return `inf` from `mass` below it rather than a plausible finite number.
  `ProcessModel.support` evaluates the branching callable on every row of a batch before the
  bounds filter it, so a finite answer there would admit a parameter with infinitely many
  offspring per event, and the failure would surface as an explosion during simulation rather
  than as a rejected proposal.

  `CompactSpatial` is `1 - (r/R)**2` inside `R` and **exactly** zero past it. A Gaussian at
  ten sigma is 1e-22 and a power law at ten scales is 1e-3 — both small, neither zero — so
  dropping such a pair changes the answer by an amount somebody has to bound. Here there is
  nothing to bound.

  How hard each is to *integrate* differs by five orders, which is what to know before
  choosing `n_quad`. The two spatio-temporal backends compute one number two ways, and at
  `n_quad = 64` they agree to 2.2e-10 for the Gaussian, 2.3e-07 for the power law and 3.1e-05
  for the compact kernel, whose kink at the radius is the hardest of the three. All three
  converge as the rule is refined, which is what says the gap is quadrature error rather than
  two implementations disagreeing.
- **A recorded benchmark harness**, `benchmarks/run.py` and `benchmarks/RESULTS.md`. Not a CI
  check: a wall-clock threshold on a shared runner measures the runner. The first recording
  says that one log-likelihood at 2 000 events is 1.1 ms closed-form against 543 ms through
  the intensity hook — **494×**, widening with `n` — which is how to size a general-kernel fit
  before starting one.

- **A background that varies over the domain.** `LogLinearBase` is
  `mu(x) = exp(b0 + sum_k b_k z_k(x))`, per unit measure, as `ConstantBase` already means
  it. A constant background says events are equally likely everywhere, which is false for
  every applied dataset, and the failure that causes is the one this package exists to
  prevent: **a constant-background fit to data whose background is clustered attributes
  the clustering to self-excitation.** Measured on data with no self-excitation at all —
  locations drawn from two fixed lumps, times uniform — the constant background peaks at
  `alpha` 0.614 over seeds 0–20, a branching ratio near 0.31 invented out of nothing,
  against 0.062 for a background that can express the lumps, and it is higher on 21 of 21
  seeds. Neither fit raises; both look converged.

  Nothing else needed changing, which was checked rather than assumed: the simulator
  already takes a callable base rate, `spatio_temporal_model` already takes a `BaseFamily`,
  and the cached likelihood backend already evaluates the background at the quadrature
  nodes and at the events.

  The link is **logarithmic rather than affine**, so the background is positive everywhere
  by construction and cannot trip the cached backend's non-negativity precondition — which
  raises rather than degrading, and would do so mid-fit on a particle that looked fine a
  move ago. The coefficients are the first parameters here on the whole real line: a sign
  is a direction, not a rate.

  There is **no separate kernel-density family**, and the reason is arithmetic rather than
  scope. `exp(b0 + b log f(x))` is `exp(b0) f(x)**b`, so a density passed as a covariate
  *is* a kernel-density background at `b = 1`, with the normaliser absorbed by the fitted
  intercept and `b` a free reading of how strongly the background follows the density. A
  fitted bandwidth and stochastic declustering are a different thing — an alternating
  scheme that absorbs the clusters into the background if the same events feed both — and
  belong with the EM baseline.

  One caution the class docstring states and the tests measure: `log_mu0` and a coefficient
  trade off through the total event count. Profiling a coefficient at a fixed intercept
  peaks in [0.53, 0.62] against a truth of 0.9 over seeds 0–20, never once near it, where
  concentrating the intercept out recovers 0.933. On real data that reads as a weaker
  spatial trend rather than as a bad fit.
- **Marks with mark-dependent productivity**, in the ETAS shape: every event carries a magnitude
  and a larger magnitude produces more offspring. `MarkedHawkes` takes any non-negative
  productivity and mark sampler; `ExponentialMarkedHawkes` is the classic case, with an
  exponential kernel and Gutenberg–Richter marks. `marked_model` and `MarkedLogLikelihood` fit
  them. A new class beside the existing ones, so nothing that exists changes behaviour.

  **An unbounded productivity is not a threat to the thinning bound**, which is worth stating
  because it looks like one. The intensity sums over `t_i < t` strictly and the bound over
  `t_i ≤ t₀`, so both range over marks *already drawn* — the supremum of `g` over the mark
  distribution never enters, and a candidate accepted at `t` draws its mark afterwards.

  **A divergent expected productivity is**, and it is the opposite kind of failure: it looks
  harmless. `E[g(m)] = b/(b − a)` is infinite for `scale ≥ b_value`, while every realised mark
  stays finite and every simulated catalogue looks ordinary. The constructor refuses it, because
  nothing at run time would.

  The mark density is part of the log-likelihood by default, and the measurement is the argument:
  **without it, `b_value` does not move the likelihood at all** — three very different rates give
  one identical value, since the parameter enters the ground process not at all. Its only other
  appearance is the stationarity boundary, so omitting the term identifies it by a constraint
  rather than by data. `TemporalLogLikelihood` refuses a marked model for exactly this reason,
  and that hole is subtler than the multivariate one: the intensity term would have come out
  *right*, and only the marks would have gone missing.
- `History` gains a `marks` column and `History.from_marked_events`. A separate constructor
  rather than a flag, for the reason `from_multivariate_events` is one: `from_events` reads a
  `(2, n)` record as one-dimensional spatio-temporal, and the shape alone cannot say which of
  the three layouts it is.

- **A validation and diagnostics harness**, in a new `hawkes_package.inference.validation`
  subpackage. Four pieces, answering four questions the existing diagnostics could not.

  `independent_compensator` and `compensator_agreement` answer *is the compensator itself
  right*. `residuals` takes its integral from the likelihood it is handed, which is correct
  for checking two implementations against each other and exactly wrong for validation: a fit
  made with a compensator 20% too small inflates the intensity, and rescaling through that same
  broken integral gives unit-rate gaps. Measured on 400 events — the inflated fit puts `alpha`
  at 0.690 against a truth of 0.500, the residuals through the broken compensator pass at
  **p = 0.46**, the same residuals through an honest one reject at **p = 3.3e-05**, and the
  agreement number reports 0.2, the defect at its actual size.

  `cell_residuals` answers *does it fit where and when the events are*. Time rescaling collapses
  the window before it starts, so it says only that something is wrong, never which part. Cells
  compare observed counts against `∫_C λ`; over seeds 0–20 the standardised residuals sit in
  [−0.343, 0.343] at the truth and never above −1.480 at a wrong excitation.

  `compare_with_baseline` answers *is the model worth having*. The bar is the homogeneous
  Poisson process at its **own maximum likelihood**, not at a convenient rate — a baseline
  handed a bad parameter is worse than none, because beating it reads as evidence. Over seeds
  0–20 a Hawkes fit beats it on 21 of 21 runs of Hawkes data and on 0 of 21 runs with the
  excitation switched off.

  `rolling_origin` answers *would it have been any use prospectively*. Expanding window,
  refitting at each origin and scoring only what followed. The no-leakage guarantee is asserted
  arithmetically rather than by reading the code: a spy fitter records every history it is
  handed and no event in any of them lies past its origin.
- A sixth executed example, `docs/examples/validating_a_fit.ipynb`, which works all four through
  on one fit and *demonstrates* the cancellation rather than describing it. It runs in 34 s and
  needs a longer window for the backtest section than for the rest, for a reason the notebook
  states: five blocks of fifty events cannot separate a branching ratio of 0.5 from a constant
  rate, and even the true parameters score negative skill at three of five origins there. That
  is the backtest being underpowered, not the model being useless — and it is exactly the result
  that gets misread as the second.

- **Bounded, non-periodic domains.** `Rectangle` and `Polygon` are the first domains here with a
  real boundary; everything else is a closed surface, periodic or a quotient. `Polygon` is convex,
  masks the quadrature by half-planes fixed at construction, and carries its own `nodes_per_axis`
  because a diagonal edge cuts every quadrature panel and the area error falls only like `1/n` --
  3.4% at the flat default, 0.85% at the 128 it asks for, and the area error is exactly the factor
  the simulated event rate is wrong by.
- **An explicit edge-correction policy**, through `edge_correction=` on
  `SpatioTemporalHawkesProcess` and `spatio_temporal_model`. On a bounded domain an event near the
  edge spreads offspring into space that is partly outside, so it produces fewer of them, and a fit
  that ignores this attributes the missing offspring to a weaker kernel. `'renormalise'` divides
  each event's spatial kernel by its own in-domain mass so every event excites the same total
  amount wherever it sits; `'none'` leaves the intensity as written; `'auto'` -- the default --
  corrects exactly where the domain sets the new `SpatialDomain.has_boundary`.

  Measured on a 4x3 rectangle at 40 events, over seeds 0-20: the corrected profile recovers the
  excitation with mean 0.900 against a truth of 0.900, while omitting the correction gives mean
  1.474 -- a **64% over-estimate** -- and exceeds the corrected estimate on 21 of 21 seeds.

  The correction is safe for the thinning bound because the divisor is a *per-event constant*: it
  depends on the event's location and the fixed quadrature, not on position or time, so it divides
  the intensity and its supremum by the same positive number. It is computed on the same quadrature
  the bound and the acceptance test already share, and the likelihood reads the policy off the
  model rather than deciding for itself -- if the simulator corrected and the likelihood did not,
  every fit would be biased by exactly the correction with nothing raising.

  It also makes the stationarity condition exact rather than conservative: the branching ratio
  assumes the spatial kernel has unit mass, which renormalisation is what makes true.
- `SpatialDomain.has_boundary`, defaulting to `False`. **Not** the negation of `periodic`: `Sphere`
  and `FundamentalDomain` are already non-periodic and have no edge, so keying the correction off
  `periodic` would rescale their intensity for nothing. Every domain that predates 0.7.0 takes the
  identical path.

- **Multivariate, mutually-exciting processes**, in both the simulator and the inference
  subpackage. `MultivariateHawkes` and `MultivariateExponentialHawkes` simulate a finite set of
  event types against one shared kernel shape and a non-negative `(d, d)` excitation matrix;
  `multivariate_model` fits them, with `MultivariateLogLikelihood` through the intensity hooks and
  `MultivariateExponentialLogLikelihood` as the `O(n·d)` closed form. The record is `(2, n)` —
  times in row 0, the type in row 1 — and `History` carries `types` and `n_types` alongside.

  **Temporal only.** Space and event type do not combine: a spatio-temporal multivariate
  process needs its own thinning bound, drawn against a space-integrated *vector* intensity,
  with the floor moving from after-the-sum to per-component and the type drawn before the
  location. That is a second bound argument rather than a wider version of the first, and
  the reordering costs the exact-value safety net every other claim here rests on. The
  multivariate likelihoods refuse a history carrying locations and say so.
- **Additive: no previously produced number moves.** The existing classes are untouched, including
  `ExponentialHawkes`'s scalar `alpha/beta` guard, which is correct for a scalar class. A one-type
  multivariate process reproduces `MonotoneKernelHawkes`, `BellShapeHawkes` and `ExponentialHawkes`
  **exactly** on the same seed — asserted at `rtol=0` on the realisation and pointwise on both
  intensity hooks — because the type costs no extra variate: the cumulative component intensities
  partition `(0, M]`, and the uniform that decides acceptance also decides which slice it landed in.
  The one-type likelihood is bit-identical to `ExponentialLogLikelihood`, compensator included.
- **Stationarity is now a spectral radius.** `A ∫κ` is the branching matrix, and the long-run rate
  is the vector `(I − G)⁻¹ μ`. Nothing downstream changed to accommodate it: `ProcessModel.support`
  and the `stationarity` prior take the ratio through an injected callable and never inspected how
  it was computed.
- `ExcitationMatrix`, `MultivariateBase` and `UnitExponentialKernel` as parameter families.
  The last exists for identifiability: `ExponentialKernel` carries `alpha`, and `A[i, j] * alpha`
  would make only the product identifiable — scale the matrix by `c`, divide `alpha` by `c`, and it
  is the same process. The posterior would wander that ridge with every diagnostic reporting health
  and the matrix it finally reported would be arbitrary. The amplitude now lives in one place.

### Changed

- **`ExponentialHawkes` is linear in the number of events, and its seeded realisations move
  in the last bits.** The loop re-summed every past event twice per thinning step, 2.36 `O(n)`
  reductions per accepted event, so a simulation was quadratic in its own output. It now
  carries `S(t) = Σ e^{-β(t - t_i)}` and advances it with one multiply: 1.258 s → 0.048 s at
  8 000 events, and 157 µs → 6.0 µs per event, flat out to 16 000 rather than growing.

  A product of decays rounds differently from one exponential of the total lag, so the
  realisation is **not bit-identical** to 0.8.0's. The size of that is the point, and is
  measured over seeds 0–20 at 2 000 events under both stopping rules: **0 of 42 realisations
  changed length**, 1 983 of 2 000 event times in seed 0 are bit-identical, and the worst
  relative move in any event time is **3.7e-16** — under two units in the last place. Same
  seed, same number of events at the same horizon, same clustering, different floats.
  `docs/migration.md` carries the caveat that cannot be measured away: an acceptance test
  could in principle flip, and one that did would differ from that point on.

  Nothing else moves. `MonotoneKernelHawkes`, `BellShapeHawkes`, `MarkedHawkes`, every
  multivariate class and the whole spatio-temporal path are **byte-identical** to 0.8.0,
  asserted by fingerprint rather than by argument.
- The temporal thinning loop reads its intensity through a cursor rather than calling
  `_upper_bound` and `_conditional_intensity` inline. A subclass implementing only the two
  hooks is unaffected — the default cursor calls them at the same times in the same order,
  which is asserted byte-for-byte. A harness that *instruments* those hooks is affected, and
  the package's own said so: every `ExponentialHawkes` case in
  `tests/statistical/test_thinning_invariant.py` failed on `assert len(pairs) > 0` rather
  than passing vacuously.

- `SpatioTemporalHawkesProcess` now checks that its quadrature rule **resolves the
  background**, not only the spatial kernel, and warns when doubling the node count moves
  the background integral by more than 1%. Nothing in the package could produce a varying
  background before 0.8.0, but a caller has always been able to pass one as a plain
  callable — and a background lump narrower than a quadrature panel loses its mass between
  the nodes, in exactly the places the events are, so the simulated event rate comes out
  wrong by that fraction with nothing said. No number changes; a construction that was
  silently wrong now says so. The check is skipped when the background is constant across
  the nodes, since this constructor runs once per particle per rejuvenation move.

### Fixed

- **A power-law kernel's compensator was too small at the package's default quadrature
  order**, which is the direction that matters: every unit of `∫λ` that goes missing is a
  penalty on a high intensity that never gets applied, so the excitation comes back too large
  and the fit looks converged. Measured against the closed-form Omori–Utsu compensator on a
  300-event history, worst relative error, and every point *low* rather than scattered:

  | core `c` | `P=8` | `P=12` | `P=16` | `P=20` |
  |---|---|---|---|---|
  | 0.5 | 1.5e-04 | 2.4e-06 | 3.4e-08 | 4.5e-10 |
  | 0.2 | 2.6e-03 | 1.7e-04 | 1.0e-05 | 5.6e-07 |
  | 0.05 | 3.0e-02 | 7.6e-03 | 1.7e-03 | 3.5e-04 |

  An exponential kernel at the same default is exact to 7e-13, so this is a property of the
  shape rather than of the rule. A kernel family may now carry `quadrature_order`, which the
  likelihoods read when the caller does not name one; `OmoriUtsuKernel` asks for 16. Nothing
  that predates 0.9.0 changes — a family without the attribute still gets 8. The order does
  not rescue a core four times narrower than the median inter-event gap, and there the
  order-`P`-versus-`2P` check fires and says the panel is the problem.

- `_EventBuffer`'s error message told a two-row record that its second row should be a coordinate.
  Multivariate records put the event type there.
- The cached spatio-temporal backend's spread warning told the reader to raise `n_quad`. That is
  right on a domain that fills its bounding box, where the spread *is* quadrature error, and
  useless on one with an edge, where it is real geometry -- 40.1% on a 4x3 rectangle -- and no node
  count reduces it. It now names the cause it has and points at `edge_correction`.

### Planned

- Make the intensity incremental for the two temporal classes that still rebuild it.
  `ExponentialHawkes` has it as of 0.9.0; `MonotoneKernelHawkes` and `BellShapeHawkes` take an
  arbitrary kernel and have no recursion to carry, so this needs a *kernel-aware* path rather
  than a loop change — an exponential mixture is the case that would work.
- Hyperbolic surfaces past twelve sides — genus 4, seven crosscaps — are refused at
  construction, and reaching them needs a different search rather than a bigger budget. A
  certified distance enumerates a deck-group window whose size grows like `exp(R)`, and the
  radius scales with the polygon; the answer stays tiny (193 elements for genus 3) while the
  search that certifies it does not. Searching outward from the *pair of points* instead of
  from the polygon's centre would size the work to the answer.
- Double precision is a second ceiling behind that one: a deck element at displacement 18 has
  hyperboloid coordinates near `5e7`, where the spacing of doubles exceeds the gap between the
  sheet and its asymptotic cone.

Beyond those three, the point-process capabilities the package does not have yet, listed in
the order they would be built. Each is scoped as a work package under `docs/extensions/`, which is
kept beside the docs and not published; the summaries here are the roadmap.

1. **A periodic time background** for diurnal, weekly and seasonal structure. Blocked by a
   signature rather than by mathematics: the background is a function of position only, and
   adding time to it also moves the thinning bound, which must then use the supremum over the
   remaining interval rather than the current value.
2. **An MLE/EM baseline beside the sequential machinery**, so the package can be benchmarked
   against other libraries on equal terms. `LogLikelihood.total` is already a scalar objective
   and `ParameterSpec` already supplies the unconstrained transform; the real cost is widening
   SciPy past the single call site it is deliberately held to.
3. **Reproducibility**: serialisation of a fitted model with a version stamp, and a coverage
   test that simulates from known parameters, refits and checks the credible intervals. Seeding
   is already done. Coverage is a statistical threshold like any other — a collapsed particle
   cloud reports a tight posterior, and only `StepRecord.move_size` tells it from a real one.

## [0.5.0] — 2026-09-04

### Added

- **`hawkes_package.viz`**, an optional subpackage that renders the spatio-temporal intensity as
  a 3-D surface coloured by `λ(t, x | H_t)` and animated over time, written out as one
  self-contained interactive HTML page with a play button and a frame slider. Four surfaces:
  `Sphere`, `FundamentalDomain.projective_plane()`, `Torus2D` / `FundamentalDomain.rectangle()`,
  and `FundamentalDomain.klein_bottle()`.

  **Visualization was not on the `### Planned` roadmap**, so this is a deliberate scope
  addition rather than a deferred item coming due. It earns its place because the package now
  reaches every closed surface and had no way to *look* at one: `intensity_over_interval` merges
  the realised event times into its own time axis and sorts them, so it cannot produce
  fixed-cadence frames at all, and the existing examples plot 2-D scatters of event locations
  with the field itself never drawn above one dimension.

  It adds **no runtime dependency**. The backend is plotly behind a new `[viz]` extra;
  `hawkes_package.viz._plotly` is the only module that names it and imports it inside a function
  body, so every `viz` module imports with numpy and scipy alone — which `tests/test_api_surface.py`
  asserts and `tests/viz/test_public_surface.py` re-checks in a subprocess.

  Three things worth knowing before reading a picture off it.

  *The colour scale is global across every frame*, and the realised range is returned on the
  result and written into the caption. A per-frame rescale would make a quiet frame look as hot
  as a burst, which destroys the one thing an animation of a self-exciting process is for. The
  ceiling is measured from the frames rather than derived from `_upper_bound`, which is the space
  *integral* of a dominating field and so a spatial average — for a sharp kernel on a large
  domain it sits below the peak and would clip the very bursts being drawn.

  *The intensity is evaluated through the simulator's own hooks*, with the time-independent
  factors hoisted out of the frame loop. That turns `n_grid × n_frames × n_events` distance
  evaluations into `n_grid × n_events` — 22 minutes down to 33 seconds for a 64×64 grid over 40
  frames on the projective plane, whose `distance` costs 327 µs a call. The hoist is
  `_full_intensity` rearranged and nothing else: it is **bit-exact** against `process.intensity`,
  which is asserted with `==` rather than a tolerance, and it re-checks a sample of finished
  values against the hook at build time and raises rather than degrading. The spatial factor is
  built through `process._spatial_at`, so pairwise kernels — everything `make_periodic` returns —
  are hoisted by the same code and not excluded.

  *Two of the four pictures distort distance, and say so.* The sphere is drawn as itself and the
  projective plane as its double cover, both exact — the covering map is a local isometry, and
  the antipodal symmetry of the colouring is the identification made visible. The flat torus and
  the Klein bottle are drawn as a donut and a figure-8 immersion, because neither admits an
  isometric embedding in three-space and the Klein bottle admits no embedding at all. The Klein
  bottle's immersion is derived from the domain's *own* side pairings rather than from a textbook
  one: `embed` reads the glide reflection `(x, y) -> (-x, y + h)` off the pairing matrices and
  lines the figure-8's base circle up with the translated axis, and the gluing closes to 7e-16.
  Getting that axis backwards would still render, still look like a Klein bottle, and tear the
  field across one seam — so `tests/viz/test_embedding.py` carries a negative control that fails
  if the axes are swapped. Hyperbolic surfaces are refused at construction rather than
  approximated.

  See `docs/visualization.md` for the reference and
  `docs/examples/intensity_surfaces.ipynb` for all four surfaces drawn and animated -- the
  latter executes on every docs build, so the documented API cannot rot silently.

- **Bayesian inference.** `hawkes_package.inference` fits the parameters of any process this
  package simulates, from observed events, in blocks as they arrive. The algorithm is an SMC
  sampler over the data-tempered posterior sequence (Chopin's IBIS) with resample–move
  rejuvenation, not a bootstrap filter: with a static parameter and no transition noise a
  bootstrap filter degenerates to a single point carrying weight one, reporting empty credible
  intervals around wherever the resampling noise left it, and reporting nothing about having
  done so.

  The likelihood is computed from **the simulator's own intensity hooks** — the same functions
  the Ogata loop thins against — so what is fitted is what would be drawn. Temporal and
  spatio-temporal, any kernel, any nonlinearity, any `SpatialDomain`.
  `SpatioTemporalLogLikelihood` carries two backends that compute the same number and always
  records which one ran: `"hooks"` is the definition, `"cached"` precomputes the geometry that
  does not depend on the parameters and is roughly 10⁵ times faster, and it **raises** rather
  than degrading where its precondition fails. Also: time-rescaling residuals with a hand-rolled
  Kolmogorov–Smirnov test, posterior-predictive forecasting, an independent Metropolis chain for
  reference posteriors, and drifting parameters behind an `evolution=` switch.

  `hawkes_package.__all__` gains exactly one name, `"inference"`. See `docs/inference.md` for the
  guide and `docs/theory.md` for why each of those choices is the one it is.
- **`HawkesProcess.simulate_until(t_end, *, start=None)`** — simulate to a time horizon rather
  than to an event count. The complement of `simulate(k)`, and the one a forecast needs: a
  fixed-count simulation cannot express "no events at all in the horizon", which is an outcome.
  `start` may be *later* than the last recorded event, which conditions on the observed fact
  that nothing happened in between. Truncating the thinning loop this way is exact rather than
  approximate. Both loops implement it; `_propagate_until` is a new abstract hook on
  `HawkesProcess`, so a class subclassing it directly (rather than `TemporalHawkesProcess`) must
  implement it.
- **`HawkesEstimator`**, a scikit-learn-shaped front door to the same fit: `fit`, `partial_fit`,
  `predict`, `score`, `forecast`, in one object where the subpackage otherwise asks for four.
  It infers nothing new, and three exact equalities say so and are tested — a `fit` is
  `fit_smc` bit-for-bit at the same seed, a `partial_fit` per block is `fit(blocks=k)`
  bit-for-bit, and a `score` is the log-evidence increment the next `partial_fit` records.

  It **inherits nothing from scikit-learn and does not import it at module scope.** `clone`,
  `Pipeline` and `GridSearchCV` reach an estimator through `get_params`/`set_params` and never
  through `isinstance` — `clone`'s own gate is `hasattr(estimator, "get_params")` — so
  inheriting `BaseEstimator` buys no behaviour, while a base class chosen by whichever packages
  happen to be installed would make `repr`, parameter ordering and pickling differ between
  environments. It is also refused by `mypy --strict`, since scikit-learn ships no `py.typed`
  and `disallow_subclassing_any` applies. `tests/inference/test_sklearn_interop.py` pins the
  reimplementation against `BaseEstimator`'s own, which is what stops the two drifting.

  Three choices worth knowing before reading a number off it. `predict` returns the conditional
  intensity **averaged over the particles**, not evaluated at the posterior mean; the intensity
  is convex in the decay rate, so the plug-in is biased low wherever the posterior has width.
  It **refuses times past `history.end`**, where the intensity computed from the observed record
  is the intensity given that nothing has happened since, and understates the truth by exactly
  the excitation of the events that would have occurred — `forecast` answers that question by
  simulating forward. And `blocks` defaults to **8**, not `fit_smc`'s 1: a single block is IBIS
  with one tempering step, which is importance sampling from the prior and degenerates on any
  history long enough to be worth fitting.

  `end` is a required keyword on `fit` with no default, for the reason `History.end` has none.
  There is no `GridSearchCV` support, and not for a technical reason: a point-process history
  cannot be sliced into folds when every fold's likelihood depends on the events before it.
- `hawkes_package.inference.block_boundaries` is now exported from the subpackage — it is how a
  `partial_fit` loop reproduces `fit`'s blocking.
- `README.md` covers `HawkesEstimator`, `simulate_until`'s `start=` argument, the `[viz]` extra
  and the surfaces it draws, and links all five executed notebooks. Its intro said the package
  "extends the construction to a spatial domain with periodic boundaries", which has undersold it
  since 0.4.0 — it reaches every closed surface.
- **`docs/examples/surfaces.ipynb`**, a fourth executed notebook, on the one part of the package
  that had no runnable example: `FundamentalDomain`. It walks the six constructors and the
  surfaces they present, shows a point's images under the gluing and the resulting quotient
  distance — two points at opposite edges of the Klein bottle's polygon are 8.2 times closer on
  the surface than on the page — checks Gauss–Bonnet by reading each hyperbolic area back as
  `-2*pi*chi`, and draws the tensor quadrature masked by `contains` on a hexagon that fills only
  75% of its bounding box. It simulates on the hexagonal torus, which no `Torus2D` expresses.

  The cost section reports **structural** numbers only — sides, `nodes_per_axis`, and the
  quadrature grid as its square, so a genus-3 surface reads as 64× a flat one — and no timings.
  A measured microsecond cost swung by a factor of 240 across sampling protocols on one machine,
  because `distance` grows and caches its search window on demand, so a printed timing would be
  noise that differs on every build. The section ends on the `genus(4)` refusal, which states the
  reason better than a benchmark could.

  A hyperbolic *simulation* is deliberately absent: genus-2 needs a 128×128 quadrature grid at
  roughly a millisecond per distance call, which is minutes per intensity integration.
- **Two new sections in `docs/examples/online_inference.ipynb`**, both executed on every docs
  build. "The same fit in one object" runs `HawkesEstimator` beside the four-object form it is
  a front door to, and plots the posterior intensity band — the figure that shows what
  averaging over the particles buys, which a plug-in at the posterior mean cannot draw. "When
  the parameter moves" contrasts `RandomWalkDrift` against `Static` across a regime change in
  `mu`: the drifting filter climbs toward the new level while the static fit *falls*, because
  shrinking the background is the only way it can reconcile events it has no mechanism to
  follow. That section is also the first runnable example of `simulate_until(..., start=)`,
  whose whole purpose — conditioning on an observed empty gap — had no code anywhere in the
  docs.
- `ProcessModel` has a readable `__repr__`. The generated dataclass one printed three closures by
  address plus every `Parameter` in full, and `HawkesEstimator`'s repr embeds it.
- `hawkes_package.spatio_temporal.kernels.image_distance_fn`, the map from a pair of points to
  the distances a periodised kernel sums over. Factored out of `make_periodic` so that the
  simulator and the cached likelihood cannot disagree about which images they see — a
  disagreement that would show up as a plausible but wrong posterior rather than as an error.

### Changed

- **A periodised spatial kernel on `Torus2D` moves by about one unit in the last place.**
  `make_periodic` now sums its image contributions with the built-in `sum`, which since CPython
  3.12 sums floats with Neumaier compensation; two of its four branches already did. `Circle` is
  bit-identical either way — its lattice has seven terms — and the 49-term torus lattice moves by
  1.1e-16. That is enough to flip a thinning acceptance, so a seeded spatio-temporal simulation
  using `make_periodic` on a `Torus2D` produces a different realisation than it did in 0.4.0. It
  is the same process, sampled more accurately; exact reproducibility was never guaranteed on the
  spatio-temporal path, as `CONTRIBUTING.md` states. `docs/migration.md` has the details.
- `hawkes_package.base._stalled_message` takes a progress phrase rather than an
  accepted/requested pair, since a run stopped by a time horizon has no requested count. The
  message `simulate` produces is unchanged.

### Fixed

- **`SpatioTemporalLogLikelihood` reused a geometry cache built for a *different* history**
  whenever the two happened to hold the same number of events. The prefix-consistency check
  lives inside `extend_geometry`, and `geometry_for` only called it when the event count
  changed — so a second history of equal length was answered with the distance tensors built
  for the first, and the log-likelihood came back for data nobody had passed, with nothing
  raised. The check now runs on every reuse, which costs nothing: `extend_geometry` already
  returns the cache unchanged once the prefix matches. Reachable before only by reusing a
  likelihood object across two fits by hand.
- **`docs/examples/temporal_processes.ipynb` taught the deprecated `.Events` spelling**, in five
  places, so the published page rendered three `DeprecationWarning` boxes telling readers that
  the attribute the tutorial itself uses is going away in 0.5.0. It now uses `.events`. This
  mattered beyond tidiness: `REMOVED_IN` is `0.5.0`, so removing the aliases would have broken
  the Docs job on a notebook nobody had reason to look at. `sphinx-build -W` does not catch it,
  because `-W` promotes Sphinx warnings and not Python ones raised inside a notebook.
- `docs/examples/spatio_temporal.ipynb` described a `monotone_temporal_kernel=False` argument
  its code did not pass, relying on the default instead. The code now passes it, since being
  explicit is the point the surrounding prose is making.
- **`README.md` documented a migration path that no longer exists.** It said `import
  TheHawkesPackage` "still works but emits a `DeprecationWarning`" and that
  `propagate_by_amount`, `propagate_by_k_events` and `propogate_by_amount` "remain as deprecated
  aliases" — all five were removed in 0.4.0, so the shim raises `ImportError` and the methods are
  gone. The section now records what 0.4.0 actually did, and names the aliases still standing
  (`Events`, `Sim_num`, `L1`, `L2`, and `FundamentalDomain`'s `n_images`) with their removal
  version.

### Removed

- **The aliases 0.4.0 deprecated, on the date 0.4.0 named.** `Events` → `events`, `Sim_num` →
  `n_simulated`, `L1`/`L2` → `width`/`height` as both keywords and attributes, and the `n_images`
  argument of `FundamentalDomain`. Reading a removed name raises `AttributeError` and passing a
  removed keyword raises `TypeError`.

  **One case Python cannot refuse for you:** `process.Events = history` now binds a plain
  attribute rather than seeding the realisation. That setter was the whole reason the alias was a
  descriptor rather than a read-only property in 0.4.0, and with the descriptor gone there is no
  warning left to catch the assignment — the symptom is a seeded run that ignored its history.
  Assign to `process.events`. `docs/migration.md` states it under 0.5.0.

  `FundamentalDomain`'s third parameter is **keyword-only** as a consequence, so
  `FundamentalDomain(vertices, pairings, 4)` now raises on the argument count. The `n_images`
  *attribute* survives, as 0.4.0 said it would: it is the default word length `orbit` reads.
- **`hawkes_package._deprecation`**, which lost its last caller to the removals above. An unused
  deprecation helper reads as supported machinery, which is worse than none — the same reasoning
  0.4.0 applied to `DeprecatedAlias` and `deprecated_module_getattr`, now applying to the module
  itself. The package carries no deprecations at all, and `tests/test_deprecations.py` asserts
  that rather than describing it. The next deprecation recreates the module, which is cheap.

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

[Unreleased]: https://github.com/jeMATHfischer/TheHawkesPackage/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.5.0
[0.4.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.4.0
[0.3.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.3.0
[0.2.0]: https://github.com/jeMATHfischer/TheHawkesPackage/releases/tag/v0.2.0
