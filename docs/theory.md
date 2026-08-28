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

## Inference

Everything above describes how to *draw* a realisation. Inference is the other
direction: given events, what were the parameters. `hawkes_package.inference`
answers that in blocks, as data arrives.

### The likelihood is available in closed form, up to one integral

A Hawkes process observed in full has no latent state. The intensity at any
moment is a deterministic function of the events already recorded, so the
log-likelihood of a realisation on $[0, T]$ is

$$
\ell(\theta) = \sum_{t_i \le T} \log \lambda_\theta(t_i^- \mid H)
             - \int_0^T \lambda_\theta(s)\,\mathrm{d}s,
$$

and in the spatio-temporal case the same with $\log\lambda_\theta(t_i^-, x_i)$
in the sum and $\int_0^T\!\!\int_D \lambda_\theta$ in the integral. The first
term rewards a high intensity where events happened; the second penalises a high
intensity everywhere else. Both are needed and the second is the one that goes
wrong quietly — a compensator computed too small subtracts too little from the
penalty, so the fitted background and excitation both drift upward while
everything looks converged.

Two details of that formula are load-bearing.

The sum uses $\lambda(t_i^-)$, the intensity *just before* each event. This
package's intensity hooks already filter $t_j < t$ strictly, which is why the
whole observed array can be assigned at once and both terms come out right. It
is also why **tied event times are refused**: two events at one instant would
each vanish from the other's conditional intensity, and one $\log\lambda$ term
would go missing with nothing raised.

The window $T$ is data, not a formality. Observing on $[0, T]$ and stopping at
the $n$-th event are different experiments, and they differ by
$-\int_{t_n}^{T}\lambda$ — the information that *nothing happened* after the
last event. That is why `History.end` has no default.

### The compensator, two ways

For the exponential kernel it is closed-form, as in *Verifying a simulator*
above, and Ozaki's recursion turns the log-sum into a single pass as well.
Writing $B(a) = \sum_{t_i \le a} e^{-\beta(a - t_i)}$, an event at $t$ has
$\lambda(t^-) = \mu + \alpha B(t^-)$ and $B$ advances by one multiplication per
step, so a full evaluation is $O(n)$ and an incremental one is $O(\text{new
events})$.

For any other kernel it is quadrature. $\lambda$ jumps at every event and is
smooth in between, so the panels are cut at the event times and an order-8
Gauss–Legendre rule is applied inside each — spectrally accurate for an analytic
kernel. A kernel with an *interior* kink puts one inside a panel, at a fixed lag
after every event, and there the convergence drops to second order with an error
that has a consistent sign. Comparing order $P$ against order $2P$ is what
detects that: measured on this project's own kernels, an undeclared triangular
kink comes in at $6.7\times10^{-3}$ and a gamma kernel of fractional shape — an
algebraic branch point rather than a kink — at $1.8\times10^{-4}$ or below.

### A bootstrap filter is the wrong algorithm

The obvious approach, a particle filter over the parameter, fails — and fails
silently. For a fully observed process the parameter is **static**: there is no
transition to propagate through. With no transition noise the cloud can never
regain diversity, resampling only ever deletes particles, and after a few
hundred observations the posterior is a single point carrying weight one. Its
variance is zero, so every credible interval it reports is empty, and its
location is wherever the resampling noise left it.

The right algorithm is an SMC sampler over the **data-tempered** sequence
$\pi_k(\theta) \propto p(\theta)\exp\ell_k(\theta)$, where $\ell_k$ is the
log-likelihood of the first $k$ blocks — Chopin's IBIS. Each block's increment
is added to every particle's log weight, and because the parameter is static
those increments telescope: adding block $k$'s increment *is* reweighting from
$\pi_{k-1}$ to $\pi_k$, exactly.

Diversity is restored by **resample–move** (Gilks and Berzuini). When the
effective sample size falls below half the cloud, the particles are resampled
and each takes a few Metropolis–Hastings steps targeting the current posterior.
The move is invariant for $\pi_k$, so it changes the particles without changing
what they represent.

Three notes on the mechanics, each of which is a way to get a plausible answer
from a broken sampler:

- The move happens on the **unconstrained** scale, $\theta = a + e^z$ and its
  relatives, because a Gaussian proposal on a rate that must stay positive is
  rejected at the boundary and stops moving. That transform is not
  measure-preserving, so the target carries $\log|\mathrm{d}\theta/\mathrm{d}z|$;
  omitting it tilts the posterior towards small values by a factor of $\theta$
  per positive coordinate.
- The proposal covariance is taken from the **weighted, pre-resampling** cloud,
  scaled by Roberts and Rosenthal's $2.38^2/d$. After resampling the duplicates
  have collapsed it towards whatever survived.
- Whether the move *worked* is measured, not assumed. The acceptance rate cannot
  tell: a proposal scaled to $10^{-12}$ proposes the point it starts from and is
  accepted essentially always. What distinguishes the two is the distance the
  cloud travelled, in units of its own width — about $0.5$ for a healthy move.

### Drifting parameters, and what the approximation costs

Once $\theta$ is allowed to change over time it becomes a latent state, and the
same loop is a genuine filter: propagate, then reweight. Two transitions ship, a
plain random walk and Liu and West's shrinkage kernel

$$
a = \frac{3\delta - 1}{2\delta}, \qquad h^2 = 1 - a^2, \qquad
z' \sim N\!\left(a z_i + (1-a)\bar z,\; h^2 V\right),
$$

which leaves $E[z'] = \bar z$ and $\mathrm{Var}[z'] = V$ exactly rather than
inflating the cloud at every block. That exactness has a price worth stating: a
jitter proportional to a variance which has already contracted cannot re-expand
the cloud, so Liu–West follows a *drift* well and a *jump* badly. An absolute
random walk is the other trade — it adds variance whether or not the data
supports it, and can therefore recover.

**The block likelihood is then an approximation, and it does not announce
itself.** The increment for block $k$ is computed with $\theta_k$ applied to the
whole intensity, including the excitation contributed by events generated under
earlier parameters. That is a locally-stationary approximation, good when the
drift is slow compared with the kernel's memory and poor when it is not; in
neither case does anything raise, and the symptom is a tracking posterior more
confident than it has earned. Under a static parameter the same expression is
exact, which is the reason the distinction is drawn at all.

Drift and rejuvenation together are refused at construction. An MCMC move
invariant for $\pi_k$ is meaningless when the model says $\pi_k$ changes at every
block, and it is exactly the configuration one arrives at by tuning until the
output looks plausible.

### Verifying an estimator

The same time-rescaling theorem that verifies the simulator verifies the fit.
Mapping the observed events through the compensator *at the fitted parameter*
should give a unit-rate Poisson process, so a Kolmogorov–Smirnov test of the
transformed gaps against $\mathrm{Exp}(1)$ catches a wrong kernel, a missing
background and an under-counted compensator at once — none of which a plot of
the posterior would notice, because a posterior can be tight, stable and wrong.

One caveat, and it generalises: the residuals must be computed with a
compensator that does **not** share the estimator's bug. A fit made with a
compensator 20% too small inflates the intensity by about 25%, and residuals
computed with the same broken compensator come back looking perfect, because the
two errors cancel exactly. A diagnostic that shares a bug with the thing it
checks is not a diagnostic.

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

   Note what that argument does *not* require: it never asks the nodes to fill a
   box, nor the weights to be the flat ones. So a domain that is a proper subset
   of its bounding box may drop the nodes outside it and scale the survivors by
   the measure density $\sqrt{\det g}$, and $M \geq \lambda$ survives untouched.
   That is how `FundamentalDomain` is integrated (§ below).
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

### Fundamental domains

A closed surface is a quotient $\widetilde{X} / \Gamma$ of a model space by a
discrete group acting freely, and it is presented concretely by a *fundamental
domain*: a region $D \subset \widetilde{X}$ containing exactly one point of each
orbit, whose boundary is glued to itself by the side pairings that generate
$\Gamma$. Distance on the quotient is then

$$
d(x, y) \;=\; \min_{g \in \Gamma} \; \tilde{d}\!\left(x, g \cdot y\right),
$$

which a `FundamentalDomain` evaluates over a truncated window of $\Gamma$, after
reducing both points into $D$ — reduce first, or for distant lifts the nearest
image falls outside the window and the kernel decays to zero instead of staying
periodic.

`Circle` is the case $\mathbb{R} / L\mathbb{Z}$ and `Torus2D` the case
$\mathbb{R}^2 / \Lambda$ for a rectangular lattice; both are written out by hand.

#### Which geometry, and why there is no choice

By uniformisation every closed surface carries a metric of constant curvature,
and the *sign* of that curvature is fixed by the topology through Gauss–Bonnet:

$$
\int_S K \, \mathrm{d}A \;=\; 2\pi\chi(S).
$$

A flat surface must have $\chi = 0$, which leaves only the torus and the Klein
bottle. Everything with $\chi > 0$ is spherical, everything with $\chi < 0$ is
hyperbolic, and nothing straddles. So the three model spaces are not three
options but a partition, and each `FundamentalDomain` checks that its polygon,
its curvature and its gluing agree on which cell of that partition it is in.

What makes one implementation serve all three is that each model space has a
**linear** model, in which isometries are $3 \times 3$ matrices and geodesic
half-spaces are linear half-spaces of an ambient $\mathbb{R}^3$: the affine
plane for $E^2$, the sphere with the Euclidean form for $S^2$, the hyperboloid
with the Minkowski form $\mathrm{diag}(1, 1, -1)$ for $H^2$. The group search,
the convexity test, the membership predicate and the Dirichlet reduction are
then the same code; only the bilinear form, the chart and the measure differ.

#### What makes a presentation a surface

A polygon with side pairings glues to *something*. Three conditions decide
whether that something is a closed surface, and none of them is visible
downstream — the thinning loop, the quadrature and the location sampler all run
happily on the alternatives. They are therefore checked at construction:

- **The pairings act freely.** An isometry with a fixed point quotients to an
  *orbifold* — a cone point, or a mirror line — rather than a surface. The case
  that matters is a rotation: it is an isometry, its determinant is $+1$, and
  until 0.4.0 it passed every check the package made.
- **Every side is glued to exactly one other.** Otherwise the quotient has a
  boundary, and is not closed.
- **Poincaré's angle condition.** Walking the corners of the polygon through the
  pairings must return to the starting corner with an interior-angle sum of
  exactly $2\pi$ and a trivial cycle transformation. An angle sum of $2\pi/m$
  gives a cone point of order $m$.

Poincaré's polygon theorem is what turns those three into a guarantee: a convex
polygon satisfying them generates a discrete group acting freely, with the
polygon as a fundamental domain. Gauss–Bonnet is then a fourth, independent
check on the same object, and it is free once the Euler characteristic is known
from the corner cycles.

#### Orientability

Both orientations of pairing are admitted. An orientation-reversing pairing
quotients to a **non-orientable** surface, which is a perfectly good closed
surface and one a Hawkes process is indifferent to: $\lambda$ is a scalar built
from a geodesic distance, and a distance has no orientation. Through 0.3.0 the
package rejected such pairings on the determinant alone. That was a policy, and
it cost the Klein bottle, the projective plane and every $N_k$ — the whole
non-orientable half of the classification — for no correctness gain. What
replaced it is the check that was actually missing: freeness.

The Klein bottle is the clearest illustration. It is the same rectangle as the
torus with the same first pairing; the second is a *glide reflection*
$(x, y) \mapsto (-x, y + L_2)$ rather than a translation. The glide reverses
orientation and has no fixed point, so it is legal. The pure reflection
$(x, y) \mapsto (-x, y)$ reverses orientation too and fixes a whole line, so it
is not.

Three further constraints are worth stating plainly:

- **The polygon is not its bounding box.** Integration masks the quadrature rule
  by $D$, and the rule's summed weights are checked against the domain's declared
  area. A rule too coarse to resolve $\partial D$ mismeasures the area, and
  mismeasuring the area scales the simulated event rate by exactly that factor —
  which is why the check warns rather than trusting the declaration. A
  hyperbolic polygon needs four times the nodes per axis that a flat one does,
  and reports as much through `nodes_per_axis`.
- **The chart is not the surface.** Quadrature nodes and Metropolis proposals
  live in a chart, and on a curved domain the chart measure and the surface
  measure differ by $\sqrt{\det g}$ — $R^2 \sin\theta$ on the sphere,
  $4/(1-|z|^2)^2$ on the Poincaré disc. Both the quadrature weights and the
  density the location sampler is handed carry that factor. Omitting it from
  either samples the wrong law; omitting it from the sampler alone would have
  piled every event at the poles of the sphere.
- **Proposals are rejected, not folded.** Folding is reversible only where
  $\Gamma$ acts by *translations*, which leave the Gaussian proposal invariant.
  That holds for the flat orientable presentations and for nothing else. So
  `FundamentalDomain.periodic` is `False`: correctness first, mixing efficiency
  second.

#### Truncating an infinite group

$\Gamma$ is finite only for the projective plane. For a torus it is a lattice
and for a hyperbolic surface it grows like $e^R$ — the number of deck elements
moving a point by at most $R$ is proportional to the area of a ball of radius
$R$, which is exponential in $H^2$. A truncation is therefore unavoidable, and
the two consumers need different ones.

For `distance` the truncation is **certified**. Both points are reduced into $D$
first; then any $g$ improving on the best distance found so far must satisfy

$$
\tilde{d}(c, g \cdot c) \;\le\; \tilde{d}(c, x) + \text{best} + \tilde{d}(c, y)
$$

by the triangle inequality, so once the window has been searched out to that
displacement the answer is the exact minimum. Word length carries no such
guarantee: it describes how a presentation was written, not how far its elements
move anything.

The radius a certificate asks for grows with the polygon, and the element count
grows like $e^R$ with the radius, so the two compound: a genus-3 surface
certifies a window of 193 elements by visiting tens of thousands, and a genus-4
one cannot be searched at all. Presentations past twelve sides are therefore
refused at construction. Nothing about the geometry forbids them; searching
outward from the *pair of points* rather than from the polygon's centre would
size the work to the answer, and is what reaching further would take.

For an image sum — `make_periodic` — no such certificate exists, because the sum
needs *all* the images and not the nearest. The tail beyond radius $R$ is of
order $e^{R} \sup_{d > R} \kappa_s(d)$, which converges only for a kernel
decaying faster than $e^{-d}$. A Gaussian qualifies; a power law does not, and a
power law says nothing about itself. So the truncation is measured at
construction and warns when the last ring of images is still contributing.

## References

- Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting
  point processes.* Biometrika 58(1), 83–90.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes.* IEEE
  Transactions on Information Theory 27(1), 23–31.
- Daley, D. J. and Vere-Jones, D. (2003). *An Introduction to the Theory of
  Point Processes.* Springer.
- Ozaki, T. (1979). *Maximum likelihood estimation of Hawkes' self-exciting
  point processes.* Annals of the Institute of Statistical Mathematics 31(1),
  145–155.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E. and Frank, L. M. (2002).
  *The time-rescaling theorem and its application to neural spike train data
  analysis.* Neural Computation 14(2), 325–346.
- Chopin, N. (2002). *A sequential particle filter method for static models.*
  Biometrika 89(3), 539–551.
- Gilks, W. R. and Berzuini, C. (2001). *Following a moving target — Monte Carlo
  inference for dynamic Bayesian models.* Journal of the Royal Statistical
  Society B 63(1), 127–146.
- Del Moral, P., Doucet, A. and Jasra, A. (2006). *Sequential Monte Carlo
  samplers.* Journal of the Royal Statistical Society B 68(3), 411–436.
- Liu, J. and West, M. (2001). *Combined parameter and state estimation in
  simulation-based filtering.* In *Sequential Monte Carlo Methods in Practice*,
  Springer, 197–223.
- Douc, R. and Cappé, O. (2005). *Comparison of resampling schemes for particle
  filtering.* Proceedings of the 4th International Symposium on Image and Signal
  Processing and Analysis, 64–69.
- Roberts, G. O. and Rosenthal, J. S. (2001). *Optimal scaling for various
  Metropolis–Hastings algorithms.* Statistical Science 16(4), 351–367.
- Brémaud, P. and Massoulié, L. (1996). *Stability of nonlinear Hawkes
  processes.* Annals of Probability 24(3), 1563–1588.
