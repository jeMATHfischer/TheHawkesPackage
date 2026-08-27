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
