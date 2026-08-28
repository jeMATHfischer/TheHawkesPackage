# Fitting a process to data

`hawkes_package.inference` estimates the parameters of any process this package
simulates, from observed events, in blocks as the data arrives. It is Bayesian:
the answer is a posterior distribution over parameters, not a point.

The likelihood it maximises is computed from **the simulator's own intensity
hooks** — the same functions the Ogata loop thins against. That is the design
constraint everything else follows from. A second, faster expression for the
intensity would be a second model, and the discrepancy would surface as a
plausible-but-wrong posterior rather than as an error.

The mathematics is in [Theory](theory.md#inference); this page is how to use it.

## A first fit

```{code-block} python
import numpy as np
import hawkes_package as hp
from hawkes_package.inference import (
    ConstrainedPrior, ExponentialLogLikelihood, History, IndependentPrior,
    LogNormal, SMCSampler, exponential_model,
)

truth = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=7)
truth.simulate(800)
history = History.from_simulation(truth)

model = exponential_model()
prior = ConstrainedPrior(
    IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
    model.support,
)

smc = SMCSampler(ExponentialLogLikelihood(model), prior, n_particles=256, rng=0)
cloud = smc.run(history, blocks=8)

print(cloud.summary())
print(smc.diagnostics.summary())
```

Four objects, and each answers one question.

{class}`~hawkes_package.inference.likelihood.History` is **the data and the
window it was observed on**. `end` is keyword-required and has no default,
because "observed on `[0, T]`" and "stopped at the 800th event" are different
experiments: they differ by the compensator over the empty tail, which is the
information that nothing happened after the last event, and dropping it biases
the background rate and the excitation upward with nothing raised.
`from_simulation` sets `end` to the last event time, which is the correct window
for the output of `simulate(k)` and the wrong one for `simulate_until(T)` — use
`from_events` with the horizon there.

{func}`~hawkes_package.inference.models.exponential_model` is **the model**: the
map from a parameter vector to a process, plus the set of parameters that map is
defined on. Its `support` excludes `alpha >= beta`, which is exactly what
{class}`~hawkes_package.ExponentialHawkes` refuses to be built at.

The prior is **where you think the truth might be**. Wrapping it in
{class}`~hawkes_package.inference.priors.ConstrainedPrior` with the model's own
support keeps the cloud where the process exists; without it the sampler raises
at `initialise` rather than quietly filtering, because a prior that includes
parameters the process cannot be simulated at is worth correcting.

{class}`~hawkes_package.inference.smc.SMCSampler` is **the fit**.

## The same fit in one object

{class}`~hawkes_package.inference.estimator.HawkesEstimator` holds all four behind
the scikit-learn method names, for when you want a fit rather than a construction:

```{code-block} python
from hawkes_package.inference import HawkesEstimator

est = HawkesEstimator(model, prior, n_particles=256, rng=0).fit(history)
print(est.theta_)                       # posterior mean
print(est.diagnostics_.warnings())      # [] is the health check
```

It infers nothing of its own. A `fit` is `fit_smc` bit-for-bit at the same seed, a
`partial_fit` per block is `fit(blocks=k)` bit-for-bit, and a `score` is the
log-evidence increment the next `partial_fit` records — all three are asserted as
exact equalities in the suite, because a wrapper that started computing its own
answer would otherwise do it quietly.

`end` is a required keyword with no default, for the reason
{class}`~hawkes_package.inference.likelihood.History` gives; passing a `History`
supplies it instead. `blocks` defaults to **8**, not `fit_smc`'s 1 — one block is
importance sampling from the prior, and degenerates on any history worth fitting.

### Predicting

`predict` returns the conditional intensity at the times you ask for, one value
per time:

```{code-block} python
grid = np.linspace(0.0, history.end, 400)
lam = est.predict(grid)
lower, median, upper = est.predict_intensity_band(grid, level=0.9)
```

It averages the intensity **over the particles** rather than evaluating it at the
posterior mean. The two differ: the intensity is convex in the decay rate, so by
Jensen the plug-in sits systematically below the marginal wherever the posterior
has width. It is the same argument
{func}`~hawkes_package.inference.forecast.posterior_predictive` makes for drawing a
fresh particle per path.

At an observed event time the value is the **left limit** `λ(t⁻)` — the intensity
hook excludes events at `t`, which is the convention the log-likelihood's sum uses.

`predict` **refuses times past `history_.end`**. There, the intensity computed from
the observed record is the intensity conditional on nothing having happened since,
and it understates the truth by exactly the excitation of the events that would have
occurred. `forecast`, `predict_counts` and `predict_interval` answer that question
properly, by simulating forward.

### Fitting as the data arrives

`partial_fit` takes **the new events and the new window**, not the history so far —
the estimator accumulates that itself:

```{code-block} python
est = HawkesEstimator(model, prior, n_particles=256, rng=0)
for block, upto in arriving_blocks:
    est.partial_fit(block, end=float(upto))
```

Passing the cumulative history by mistake raises on the first event it has already
seen. The other convention would have accepted it and quietly counted a block twice,
which tightens the posterior around whatever it already believed.

An empty block is legitimate: it says nothing happened, which is information.

`score` is the log posterior-predictive density of the next window, higher being
better, and it must **abut** the fitted one — the process has memory, so a segment
scored as though it began fresh is a different quantity.

### What it is not

- Not a `sklearn.base.BaseEstimator` subclass, and scikit-learn is not imported at
  module scope. `clone`, `Pipeline` and `GridSearchCV` reach an estimator through
  `get_params`/`set_params` and never through `isinstance`, so inheritance buys
  nothing — while a base class chosen by what happens to be installed would make
  `repr`, parameter ordering and pickling differ between environments. If you need
  the `isinstance`, subclass it yourself:
  `class SklearnHawkes(BaseEstimator, HawkesEstimator): ...`
- It does not pass `check_estimator`: `X` is a 1-D array of event times with no
  feature axis, `fit` requires a keyword `end`, and there is no `n_features_in_`.
  `parameter_names_` is the honest name for what `feature_names_in_` would hold.
- **No `GridSearchCV` or `cross_val_score`**, for a statistical reason rather than a
  technical one: a point-process history cannot be sliced into folds when every
  fold's likelihood depends on the events before it. Sweep with an explicit loop over
  `clone`, `set_params`, `fit` and a chronological `score`.
- Nested `set_params` (`prior__sd_log`) does not resolve — the priors and models here
  are frozen objects, not estimators, and are replaced wholesale.
- `fit` after `partial_fit` **discards** the online state, as scikit-learn's semantics
  require. It is the one way to lose a long online fit.

## Choosing a prior

A log-normal on each positive parameter is the default worth reaching for: it
has the right support, its tail is heavy enough that a prior guessed an order of
magnitude wrong still puts mass where the data is, and its logarithm is the
scale the sampler moves on anyway.

Note that `LogNormal(m, s)` parameterises the **median** as `exp(m)`, not the
mean. `LogNormal(0.5, 1.0)` has median 1.6 and mean 2.7.

For a parameter you know a scale for, {class}`~hawkes_package.inference.priors.Gamma`
in the shape–rate parameterisation is the other natural choice.
{class}`~hawkes_package.inference.priors.Uniform` and
{class}`~hawkes_package.inference.priors.Normal` are there for bounded and
unbounded parameters respectively. Anything with `log_pdf` and `sample` works —
the protocol is two methods and inherits nothing.

## Reading the diagnostics

```
SMC over 256 particles, backend=cached
      upto   events       ESS  resamp   accept      move   unique       dlogZ
     25.16      100      25.6     yes    0.305  4.41e-01    0.258       35.91
     52.94      200     101.3     yes    0.240  3.82e-01    0.594       29.36
     79.72      300     212.1      no        -         -    1.000       31.81
...
log evidence 349.06 (up to the prior's constant)
```

| Column | What it says | What is wrong |
|---|---|---|
| `ESS` | Effective sample size **before** any resample | Nothing on its own: weights concentrate whenever a block is informative, and 4% of the cloud is ordinary. What matters is whether a resample followed — {meth}`~hawkes_package.inference.smc.SMCDiagnostics.ess_recovered` |
| `accept` | Metropolis acceptance during rejuvenation | Near 0 means the proposal is far too large. Near 1 means it is far too small, which looks like success |
| `move` | Distance the cloud travelled, in units of its own width | **The number to read.** Around 0.5 is healthy; near zero means the cloud is exactly where the resample left it, so the posterior is a resampled prior |
| `unique` | Distinct ancestors after resampling | How much diversity the resample took. Rejuvenation is what puts it back |
| `dlogZ` | The block's contribution to the log evidence | — |

{meth}`~hawkes_package.inference.smc.SMCDiagnostics.warnings` reports the
combinations that mean something, and `summary()` prints them. An empty list is
the thing to check for; a tight posterior with an empty warning list is a fit
worth believing, and a tight posterior without one is not.

The `log evidence` is exact only up to the prior's normalising constant. A
`ConstrainedPrior` is unnormalised by design — the truncation constant cancels
everywhere else — so differences between models sharing a prior are meaningful
and absolute values are not.

## Fitting online

`run(history, blocks=k)` is a convenience over the real entry point, which is
{meth}`~hawkes_package.inference.smc.SMCSampler.update`. Call it as each block of
events arrives:

```{code-block} python
smc = SMCSampler(ExponentialLogLikelihood(model), prior, n_particles=256, rng=0)
smc.initialise()

for upto in observation_times:          # as the data comes in
    smc.update(grown_history, float(upto))
    if smc.diagnostics.warnings():
        print(smc.diagnostics.summary())
```

`upto` is the time the process has been **observed** to, which is not the same as
the last event time. The difference is data: it says nothing happened in between,
and dropping it biases the background rate upward.

{meth}`~hawkes_package.inference.estimator.HawkesEstimator.partial_fit` is the same
call with the bookkeeping done for you: it takes the new events and the new `end`,
and grows the history itself.

Eight blocks of a hundred reach the same posterior as one block of eight hundred
— the increments telescope exactly under a static parameter — so the block
structure is a matter of when you want answers, not of what answer you get. It
does change *cost*: rejuvenation is by far the most expensive thing here, and it
can only happen at a block boundary.

## Checking the fit

A posterior can be tight, stable and wrong. The check that catches that is the
time-rescaling theorem:

```{code-block} python
from hawkes_package.inference import ks_exponential, residuals

gaps = residuals(ExponentialLogLikelihood(model), cloud.mean(), history)
print(ks_exponential(gaps))        # p > 1e-3 is consistent
```

{meth}`~hawkes_package.inference.estimator.HawkesEstimator.residuals` and
{meth}`~hawkes_package.inference.estimator.HawkesEstimator.report` are the same two
calls off a fitted estimator.

Under the true parameter the rescaled gaps are independent `Exp(1)`, so this one
test catches a wrong kernel, a missing background and an under-counted
compensator at once. {func}`~hawkes_package.inference.diagnostics.posterior_report`
prints it alongside the marginals, because the combination — confident posterior,
rejected fit — is the one worth noticing and the one that goes unnoticed when
they are read separately.

## Forecasting

```{code-block} python
from hawkes_package.inference import posterior_predictive, predictive_counts

paths = posterior_predictive(model, cloud, history, horizon=20.0, n_paths=200, rng=1)
print(np.quantile(predictive_counts(paths), [0.05, 0.5, 0.95]))
```

{meth}`~hawkes_package.inference.estimator.HawkesEstimator.forecast`,
`predict_counts` and `predict_interval` are the same three off a fitted estimator.

Each path draws a fresh particle, so the result carries the uncertainty about
*which* process as well as the process's own variability. Simulating many paths
from the posterior mean instead gives a band that is too narrow, and at a few
hundred events the difference is not subtle.

The forecast starts at `history.end`, not at the last event — see above. This is
what {meth}`~hawkes_package.base.HawkesProcess.simulate_until` exists for, and
also why forecasting cannot be done by simulating a fixed count and truncating:
"no events at all in the horizon" is an outcome, and a count-based simulation
cannot produce it.

## Spatio-temporal fits

The same four objects, with a domain:

```{code-block} python
from hawkes_package.inference import SpatioTemporalLogLikelihood, spatio_temporal_model

model = spatio_temporal_model(hp.Circle(), n_quad=64)   # (mu, alpha, beta, sigma)
likelihood = SpatioTemporalLogLikelihood(model)
smc = fit_smc(likelihood, prior, history, blocks=6, n_particles=128, rng=5)
print(likelihood.backend_used)      # 'cached'
```

Two backends compute the same number. `backend="hooks"` is the definition: it
builds the process, conditions it on the history, and calls `_full_intensity` and
`_integrated_intensity`. It is also unusable for a fit — one space integral costs
114 ms on a `Circle` at 256 nodes, so a single log-likelihood at 200 events is
about twelve minutes.

`backend="cached"` rearranges the same quantity around the separability of the
intensity. Because $\lambda(t,x) = \mu(x) + \sum_i \kappa_t(t-t_i)\kappa_s(d(x,x_i))$
is a sum, the space integral distributes into a background term plus one
per-event spatial mass $S_i$, and every $S_i$ is a quadrature sum over distances
that **do not depend on the parameters**. Computed once, a full log-likelihood
becomes about three milliseconds.

That rearrangement has a precondition. The process floors the intensity *after*
summing, so the identity holds only where the pre-floor integrand is non-negative
at every node. Below zero the cached form over-counts, the compensator comes out
too small, and the excitation is biased upward — so the cached backend checks and
**raises** rather than degrading. `backend="auto"` falls back to the hooks once,
with a warning, and records which path ran in `backend_used`.

:::{note}
The cost of a spatio-temporal fit is dominated by the geometry cache, which is
built once and costs `nodes × events × images` distance calls. On a `Circle` at
200 events that is a couple of seconds; on a hyperbolic fundamental domain,
where one `distance` call is 79 µs, it is minutes. The fit that follows is
milliseconds either way.
:::

## Drifting parameters

By default the parameter is static and the sampler is exact. Passing an
`evolution` makes it a filter over a parameter that changes:

```{code-block} python
from hawkes_package.inference import RandomWalkDrift

smc = SMCSampler(likelihood, prior, evolution=RandomWalkDrift(0.1), n_move=0, rng=0)
```

`n_move=0` is required, and the constructor raises otherwise: an MCMC move
invariant for a posterior the model says changes every block is incoherent, and
it is exactly the configuration reached by tuning until the output looks
plausible.

Which kernel to use is decided by what the parameter does.
{class}`~hawkes_package.inference.evolution.LiuWest` preserves the cloud's mean
and variance exactly, so a genuinely constant parameter does not look
increasingly uncertain the longer it is watched — but its jitter is proportional
to a variance that may already have contracted, so it follows a slow *drift* well
and a *jump* badly. {class}`~hawkes_package.inference.evolution.RandomWalkDrift`
adds an absolute amount of variance per block whether or not the data supports
it, and can therefore re-expand a collapsed cloud to follow a step.

**The block likelihood is an approximation under any drift**, and it does not
announce itself: the increment is computed with the current parameter applied to
the whole intensity, including excitation left by events generated under earlier
ones. Good when the drift is slow compared with the kernel's memory, poor when it
is not, and silent in both cases.

## What it costs

| Path | Full `ℓ(θ)` | Incremental |
|---|---|---|
| temporal, exponential | `O(n)` — 0.55 ms at n = 800 | `O(new events)` |
| temporal, general kernel | `O(n²P)` — hundreds of ms at n = 2000 | `O(nP)` |
| spatio-temporal, hooks | `O(n²mP)` — minutes at n = 200 | — |
| spatio-temporal, cached | vectorized — 2.8 ms at n = 200 | — |

Rejuvenation dominates a fit, at `n_particles × n_move` full evaluations per
resample. For the exponential family that is well under a second; for a general
temporal kernel it is minutes, so size such a fit at a few hundred events or use
{class}`~hawkes_package.inference.likelihood.ExponentialLogLikelihood`.

## What is not here

Partially observed or thinned data. That creates a genuine latent state and
needs a different algorithm — a bootstrap filter or a branching-structure
augmentation — rather than a different setting of this one. Also out of scope:
multivariate and mutually-exciting processes, and discrete marks.

`hawkes_package.mcmc` is untouched by any of this. It remains the spatial
location sampler on the Ogata correctness path; inference has its own chain in
{func}`~hawkes_package.inference.mcmc.metropolis_chain`, named so it cannot read
as a drop-in.
