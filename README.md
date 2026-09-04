# the-hawkes-package

[![CI](https://github.com/jeMATHfischer/TheHawkesPackage/actions/workflows/ci.yml/badge.svg)](https://github.com/jeMATHfischer/TheHawkesPackage/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/the-hawkes-package.svg)](https://pypi.org/project/the-hawkes-package/)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://pypi.org/project/the-hawkes-package/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://jeMATHfischer.github.io/TheHawkesPackage/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/jeMATHfischer/TheHawkesPackage/blob/master/LICENSE)

Simulation and Bayesian inference for temporal and spatio-temporal Hawkes processes.

A Hawkes process is a self-exciting point process: every event raises the probability of further
events for a while afterwards. This package simulates them for a range of kernel shapes — including
non-monotone ("bell-shaped") kernels and nonlinear intensities, where the naive thinning bound is
wrong — extends the construction to **any closed surface**, and **fits them to observed events**
by sequential Monte Carlo.

```
λ(t | H_t) = φ( μ + Σ_{t_i < t} κ(t − t_i) )
```

## Installation

```bash
pip install the-hawkes-package
```

Requires Python 3.10+, NumPy and SciPy. To draw the intensity as an animated 3-D surface, add the
optional renderer: `pip install "the-hawkes-package[viz]"`.

## Quickstart

```python
import numpy as np
import hawkes_package as hp

# Linear Hawkes with an exponential kernel: param = [mu, alpha, beta]
process = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
process.simulate(100)

times, intensity = process.intensity_over_interval(np.linspace(0, process.events[-1], 1000))
```

The constructor enforces the stationarity condition `alpha / beta < 1` and raises `ValueError`
otherwise — an unstable process would not terminate.

### Available processes

| Class | Use when |
|---|---|
| `ExponentialHawkes` | Linear intensity, exponential kernel. The classic case. |
| `MonotoneKernelHawkes` | Any monotone-decreasing kernel, with a monotone-increasing nonlinearity `φ`. |
| `BellShapeHawkes` | Kernels with a single interior maximum, where the excitation ramps up before decaying. |
| `SpatioTemporalHawkesProcess` | Events carry a location on a surface: `Circle`, `Sphere`, `Torus2D` or any `FundamentalDomain`. |

`simulate(k)` stops after `k` events; `simulate_until(T)` stops at a horizon, and is what a
forecast needs — a fixed-count simulation cannot express "no events at all in the window". Its
`start=` may sit *later* than the last recorded event, which conditions on the observed fact that
nothing happened in between; that is how a forecast continues from the end of a window rather than
from the last thing that happened.

### Spatial domains

```python
import hawkes_package as hp

process = hp.SpatioTemporalHawkesProcess(
    base=lambda x: 0.5,
    spatial=lambda d: max(0.0, 1 - d / np.pi),
    temporal=lambda dt: 0.9 * np.exp(-5 * dt),
    domain=hp.Circle(),
    monotone_temporal_kernel=True,
    rng=42,
)
process.simulate(50)
```

The background may vary in space — `base=lambda x: 0.5 + 0.2*np.cos(x[0])` — and generalises
unchanged to two dimensions, since a callable always receives a shape-`(ndim,)` point.

`SpatialDomain` is an ABC — implement `distance`, `wrap`, `sample_uniform`, `volume` and `bounds`
to simulate on your own geometry. A domain need not fill its bounding box: override the optional
`contains` and `volume_element` hooks, and `volume` must then equal the integral of
`volume_element` over the part of `bounds` that `contains` admits. For a domain that does fill its
box — the default — that reduces to `volume == prod(bounds widths)`.

`Circle`, `Torus2D` and `Sphere` are written out by hand. `FundamentalDomain` is the general
construction the first two are instances of — a convex geodesic polygon plus the side-pairing
isometries that identify its boundary — and between them they reach **every closed surface**:

```python
hp.Sphere()  # the sphere
hp.FundamentalDomain.hexagon(1.0)  # the hexagonal torus, which no rectangle expresses
hp.FundamentalDomain.klein_bottle(3, 5)  # non-orientable, and still flat
hp.FundamentalDomain.projective_plane()  # a hemisphere, antipodes identified
hp.FundamentalDomain.genus(2)  # two handles, so hyperbolic — necessarily
hp.FundamentalDomain.crosscaps(3)  # three crosscaps, hyperbolic too
```

Which geometry a surface gets is not a setting: Gauss–Bonnet ties the sign of the curvature to the
Euler characteristic, so the torus and the Klein bottle are flat, the sphere and the projective
plane are spherical, and everything else is hyperbolic. Each domain reports what it built, through
`domain.topology`, and refuses at construction to build something that is not a closed surface.

`make_periodic` wraps an isotropic kernel so it sums correctly over the domain's image points, and
can be passed straight in as `spatial=`:

```python
kernel = hp.make_periodic(lambda d: np.exp(-(d**2)), hp.Circle())
process = hp.SpatioTemporalHawkesProcess(base, kernel, temporal, domain=hp.Circle())
```

## Inference

`hawkes_package.inference` estimates parameters from observed events, in blocks as they arrive,
by sequential Monte Carlo with MCMC rejuvenation. The likelihood is computed from the simulator's
own intensity hooks, so what is fitted is exactly what would be drawn.

```python
from hawkes_package.inference import (
    ConstrainedPrior,
    ExponentialLogLikelihood,
    History,
    IndependentPrior,
    LogNormal,
    SMCSampler,
    exponential_model,
    ks_exponential,
    residuals,
)

history = History.from_simulation(process)  # events *and* their window
model = exponential_model()  # theta -> process, and where it is defined
prior = ConstrainedPrior(
    IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
    model.support,  # excludes alpha >= beta
)

smc = SMCSampler(ExponentialLogLikelihood(model), prior, n_particles=256, rng=0)
cloud = smc.run(history, blocks=8)  # or smc.update(...) as data arrives

print(cloud.summary())  # marginals and 90% intervals
print(smc.diagnostics.summary())  # ESS, acceptance, move size, evidence
print(ks_exponential(residuals(smc.likelihood, cloud.mean(), history)))  # does it fit?
```

The same fit is also available behind a scikit-learn-shaped surface, for when one object beats
four:

```python
from hawkes_package.inference import HawkesEstimator

est = HawkesEstimator(model, prior, n_particles=256, rng=0).fit(history)

est.theta_  # posterior mean
est.predict(times)  # intensity, averaged over the particles
est.partial_fit(new_events, end=T)  # the filter is genuinely online
est.score(held_out, end=T2)  # log posterior-predictive density
```

`predict` averages the intensity **over the posterior** rather than evaluating it at `theta_`; the
two differ, and the plug-in is the one biased low. It refuses times past `history.end`, where the
intensity computed from the observed record is the intensity given that nothing has happened
since — `forecast` answers that question properly, by simulating forward. scikit-learn is not a
dependency and is never imported.

Spatio-temporal fits work the same way through `spatio_temporal_model` and
`SpatioTemporalLogLikelihood`, on any `SpatialDomain`. `posterior_predictive` forecasts from the
posterior by simulating forward, carrying the parameter uncertainty rather than plugging in a
point estimate.

`History.end` is keyword-required and has no default: "observed on `[0, T]`" and "stopped at the
n-th event" are different experiments, and defaulting it would switch between them silently.

## Visualization

`hawkes_package.viz` draws the surface a spatio-temporal process lives on, colours it by
`λ(t, x | H_t)`, and animates it over time into one self-contained interactive page — play
button, frame slider, orbitable camera.

```python
from hawkes_package.viz import animate_intensity

frames = animate_intensity(process, np.linspace(0, horizon, 40), "surface.html")
print(frames.summary())
```

Four surfaces: the sphere and the projective plane are drawn **exactly**, the flat torus and the
Klein bottle as immersions — neither admits an isometric embedding in three-space, so the
distances on screen are not the geodesic distances driving the intensity, and each figure's
caption says so. Hyperbolic surfaces are refused rather than drawn misleadingly.

The renderer is an optional extra; the field itself is computed with numpy alone, through the same
intensity hooks the simulator thins against.

## Reproducibility

Every process takes `rng=`, accepting `None`, an `int` seed, or an existing `numpy.random.Generator`:

```python
hp.ExponentialHawkes(param, rng=42)  # reproducible
hp.ExponentialHawkes(param, rng=my_generator)  # share one stream
```

`np.random.seed(...)` does **not** control simulations. See [CHANGELOG.md](https://github.com/jeMATHfischer/TheHawkesPackage/blob/master/CHANGELOG.md).

## Migrating

The import name is `hawkes_package`. The `TheHawkesPackage` shim was **removed in 0.4.0**, together
with `propagate_by_amount`, `propagate_by_k_events`, the `propogate_by_amount` typo,
`Spatio_Temporal_Hawkes_Process` and `LegacySpatioTemporalHawkesProcess`. `simulate(k)` is the
method on every process class.

0.4.0 renamed the frozen public names, keeping each old spelling as a deprecated alias:
`Events` → `events`, `Sim_num` → `n_simulated`, and `L1`/`L2` → `width`/`height`. **0.5.0 removed
those aliases**, along with the `n_images` argument of `FundamentalDomain` and the internal
`_deprecation` module that served them — the package now carries no deprecations at all.

One case is worth knowing because Python cannot refuse it: `process.Events = history` now binds a
plain attribute instead of seeding the realisation. Assign to `process.events`.

[docs/migration.md](https://jeMATHfischer.github.io/TheHawkesPackage/migration.html) has the rest,
including the changes that alter previously produced numbers.

## Documentation

<https://jeMATHfischer.github.io/TheHawkesPackage/>

Five executed notebooks, run on every documentation build:
[temporal processes](https://jeMATHfischer.github.io/TheHawkesPackage/examples/temporal_processes.html),
[spatio-temporal](https://jeMATHfischer.github.io/TheHawkesPackage/examples/spatio_temporal.html),
[compact surfaces](https://jeMATHfischer.github.io/TheHawkesPackage/examples/surfaces.html),
[intensity surfaces](https://jeMATHfischer.github.io/TheHawkesPackage/examples/intensity_surfaces.html) and
[fitting to data](https://jeMATHfischer.github.io/TheHawkesPackage/examples/online_inference.html).

## License

MIT — see [LICENSE](https://github.com/jeMATHfischer/TheHawkesPackage/blob/master/LICENSE).
