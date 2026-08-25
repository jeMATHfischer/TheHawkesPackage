# Theory

## The conditional intensity

A Hawkes process is a point process whose conditional intensity depends on its
own history $H_t$:

$$
\lambda(t \mid H_t) = \varphi\!\left( \mu + \sum_{t_i < t} \kappa(t - t_i) \right),
$$

with background rate $\mu$, excitation kernel $\kappa$, and a monotone
increasing nonlinearity $\varphi$ (the identity in the linear case). Each event
raises the intensity, which raises the chance of further events — the
self-exciting feedback that produces clustering.

The sum runs over $t_i < t$ **strictly**, so the intensity reported *at* an
event time is the left limit, the value just before that event's contribution
lands.

## Ogata's thinning algorithm

Every simulator in this package uses Ogata's thinning (a Lewis–Shedler
construction). The idea is to dominate the true intensity $\lambda^*(t)$ with a
bound $M$, generate candidates from a Poisson process at rate $M$, and accept
each with probability $\lambda^*(t)/M$:

```text
1. Start at the current time t0 (the last accepted event).
2. Compute an upper bound  M >= lambda*(s)  for all s >= t0.
3. Draw U ~ Uniform(0,1), set tau = -log(U)/M, advance t = t0 + tau.
4. Draw V ~ Uniform(0,1).
5. If V <= lambda*(t)/M  -> accept: record an event at t, set t0 = t, go to 2.
   Else                  -> reject: go to 2, reusing t as the new t0.
```

### The bound is the whole game

$M$ must satisfy $M \geq \lambda^*(s)$ for **all** $s \geq t_0$, including at
$t_0$ itself. Both ways of getting this wrong are silent:

- **$M$ too small.** The acceptance ratio exceeds 1, so every candidate is
  accepted and the output is a Poisson process at rate $M$, not a Hawkes
  process. Nothing raises.
- **$M$ too large.** Still correct, just wasteful — more rejections per event.

Locating the peak is part of the bound, and is easy to get wrong: a local
search started at lag 0 sees nothing on a kernel that is flat there — the
standard *delayed excitation* shape — and reports a peak of zero, silently
collapsing the bell-shaped bound to the monotone one. The package scans a grid
over an adaptively expanded window and never returns a value below anything it
observed; `peak_lag=` bypasses the search for a kernel with a spike too narrow
to be seen on a grid.

The temporal kernel must be non-negative for this bound to be valid, and that is
checked. The *spatial* kernel may change sign — inhibition is supported — and the
bound clips its negative values, which is the correct supremum rather than a
patch: with $\kappa_t \geq 0$ decaying, the supremum of $\kappa_t \kappa_s$ is
$\sup(\kappa_t)\,\kappa_s$ where $\kappa_s \geq 0$ and $0$ where it is not.

Because a wrong bound produces plausible-looking output, the package's test
suite checks the invariant directly: it records every $(M, \lambda)$ pair that
the acceptance test compares, asserts $\lambda \leq M$ throughout, and asserts
that *some* candidates are rejected. Four bounds that violated it shipped
undetected before 0.2.0, each in a configuration the harness did not then
cover — which is why it now spans delayed kernels, two-dimensional domains,
sign-changing spatial kernels and periodised kernels.

### How this package bounds

Each past event's future contribution is bounded **separately**, by the
supremum its own kernel can still reach:

$$
M = \varphi\!\left( \mu + \sum_{t_i \le t_0} \sup_{s \ge t_0} \kappa(s - t_i) \right).
$$

For a monotone-decreasing $\kappa$ that supremum is just $\kappa(t_0 - t_i)$, so
the bound is the current intensity. For a bell-shaped $\kappa$ it is the peak
value $\kappa(\text{ext})$ for any event that has not yet peaked. Bounding
per-event is what makes the result correct when several events are rising at
once — inflating the total by a single peak is not enough.

Note the $t_i \le t_0$: the event *at* $t_0$ must be counted. Omitting it is the
classic failure, and it is exactly what makes $M < \lambda^*(t_0 + \epsilon)$.

## Stability

A Hawkes process is stable — it does not produce infinitely many events in
finite time — when the expected number of offspring per event is below one.

| Class | Stability condition |
|---|---|
| `ExponentialHawkes([μ, α, β])` | $\alpha/\beta < 1$ |
| `MonotoneKernelHawkes(κ, φ)` | depends on the kernel mass $\int \kappa$ and on how fast $\varphi$ grows |
| `BellShapeHawkes(κ, φ)` | as above |

For the exponential case the branching ratio is exactly $\alpha/\beta$, and the
constructor rejects $\alpha/\beta \ge 1$ outright. The long-run event rate is
then

$$
\text{rate} = \frac{\mu}{1 - \alpha/\beta}.
$$

The nonlinear classes have no such closed form, so nothing is checked up front.
An explosive choice — $\varphi = \exp$ with a unit-mass kernel, say — makes the
intensity diverge, the inter-arrival times underflow to zero, and time stop
advancing. The simulator detects that and raises `RuntimeError` rather than
looping forever.

## Verifying a simulator

The sharpest available check is the **time-rescaling theorem**: if events really
come from intensity $\lambda^*$, then mapping them through their own
compensator

$$
\Lambda(t) = \int_0^t \lambda^*(s)\,\mathrm{d}s
$$

yields a unit-rate Poisson process, so the transformed gaps
$\Lambda(t_{i+1}) - \Lambda(t_i)$ are $\mathrm{Exp}(1)$. For the exponential
kernel the compensator is closed-form,

$$
\Lambda(t) = \mu t + \frac{\alpha}{\beta} \sum_{t_i < t}
             \left( 1 - e^{-\beta (t - t_i)} \right),
$$

and a Kolmogorov–Smirnov test against $\mathrm{Exp}(1)$ catches a wrong kernel,
a wrong bound and a missing baseline at once — none of which a shape or
monotonicity assertion would notice. The package tests this on every build.

## Spatio-temporal processes

Events carry a location $x$ in a spatial domain, and the intensity becomes
separable:

$$
\lambda(t, x \mid H_t) = \mu(x)
    + \sum_{t_i < t} \kappa_t(t - t_i)\, \kappa_s\!\left( d(x, x_i) \right),
$$

where $d(\cdot,\cdot)$ is the geodesic distance on the domain — the shorter arc
on a `Circle`, the wrapped Euclidean distance on a `Torus2D`.

Simulation runs in two stages:

1. **Time.** Ogata thinning against the space-integrated intensity
   $\int_{\mathcal{X}} \lambda(t, x) \,\mathrm{d}x$, computed by a fixed
   Gauss–Legendre tensor rule at every dimension.

   The rule being *fixed* is what makes the bound a bound. An estimate redrawn
   per call — Monte Carlo, as before 0.2.0 — is unbiased, not dominating, so it
   falls below the truth about half the time; and comparing two independent
   estimates let the acceptance ratio exceed 1. Sharing one node set with
   strictly positive weights means a pointwise-dominating integrand integrates
   to a dominating value, so $M \geq \lambda$ holds by construction.
2. **Space.** Given an accepted event time, the location is drawn by
   Metropolis-Hastings from the conditional spatial density
   $\lambda(t, \cdot) / \int \lambda(t, x)\,\mathrm{d}x$.

:::{note}
The sampler is confined to the domain. Proposals outside it are rejected, which
is the correct Metropolis move for the density restricted to the domain and is
valid whatever the geometry. On a genuinely periodic domain the proposal may
instead be *folded* back — also reversible, since folding is a symmetric move on
the quotient — which mixes better across the seam; that is what `Circle` and
`Torus2D` do, via their `periodic` flag.

Before 0.2.0 the walk was unbounded and out-of-domain proposals were never
rejected. That is correct only when the target is periodic with the domain: with
a non-periodic background the sampled marginal was indistinguishable from
uniform, and on a domain whose `wrap` clips rather than folds, every event
landed exactly on a boundary.
:::

## References

- Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting
  point processes.* Biometrika 58(1), 83–90.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes.* IEEE
  Transactions on Information Theory 27(1), 23–31.
- Daley, D. J. and Vere-Jones, D. (2003). *An Introduction to the Theory of
  Point Processes.* Springer.
