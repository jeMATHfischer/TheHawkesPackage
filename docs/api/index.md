# API reference

```{eval-rst}
.. currentmodule:: hawkes_package
```

## Temporal processes

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   ExponentialHawkes
   MonotoneKernelHawkes
   BellShapeHawkes
```

## Spatio-temporal processes

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   SpatioTemporalHawkesProcess
```

## Spatial domains

Implement {class}`SpatialDomain` to simulate on your own geometry: the
simulator works against that interface alone.

{class}`Circle` and {class}`Torus2D` are written out by hand.
{class}`FundamentalDomain` is the general construction they are instances of —
a convex geodesic polygon plus the side-pairing isometries that identify its
boundary. {class}`Sphere` is the one closed surface that is *not* a quotient:
it is simply connected, so a deck group is the wrong tool for it.

Between them these reach **every closed surface**. Which geometry a surface
needs is decided by the sign of its Euler characteristic, not by preference:

| χ | Surface | Geometry | Built by |
|---|---|---|---|
| `2` | sphere | spherical | {class}`Sphere` |
| `1` | projective plane | spherical | `FundamentalDomain.projective_plane()` |
| `0` | torus | flat | `FundamentalDomain.rectangle()`, `.hexagon()` |
| `0` | Klein bottle | flat | `FundamentalDomain.klein_bottle()` |
| `2 − 2g` | genus `g ≥ 2` | hyperbolic | `FundamentalDomain.genus(g)` |
| `2 − k` | `k ≥ 3` crosscaps | hyperbolic | `FundamentalDomain.crosscaps(k)` |

None of these fills its bounding box, and none but the flat ones carries the
flat chart measure — which is what the `contains` and `volume_element` hooks on
the base class are for.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   SpatialDomain
   Circle
   Torus2D
   Sphere
   FundamentalDomain
```

## Base classes

Shared machinery. `HawkesProcess` owns the random stream, `simulate` and the
deprecated aliases; `TemporalHawkesProcess` adds the Ogata loop and the
intensity accessor, driven by two hooks that concrete classes supply.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   HawkesProcess
   TemporalHawkesProcess
```

## Functions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   make_periodic
   mcmc_sampler
```

## Inference

Fitting a model to observed events, in blocks as they arrive. The narrative
guide is [Fitting a process to data](../inference.md); what follows is the
surface.

```{eval-rst}
.. currentmodule:: hawkes_package.inference
```

The data, the model, and the map between them. A `History` carries the events
**and the window they were observed on**; a `ProcessModel` is the map from a
parameter vector to a process, and the set of parameters that map is defined on.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   History
   ProcessModel
   Parameter
   ParameterSpec
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   exponential_model
   monotone_model
   bell_shape_model
   spatio_temporal_model
```

### Likelihoods

`ExponentialLogLikelihood` is the closed form and is `O(n)`;
`TemporalLogLikelihood` works for any temporal model through its intensity hook
and is `O(n²P)`; `SpatioTemporalLogLikelihood` carries two backends that compute
the same number, and always records which one ran.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   ExponentialLogLikelihood
   TemporalLogLikelihood
   SpatioTemporalLogLikelihood
   LikelihoodState
```

### Priors

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   IndependentPrior
   ConstrainedPrior
   LogNormal
   Gamma
   Normal
   Uniform
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   stationarity
```

### Families

Parameterised kernels, backgrounds and nonlinearities, each carrying the
analytic peak and mass that a numerical search would otherwise have to find once
per particle per move.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   ExponentialKernel
   GammaKernel
   GaussianSpatial
   ConstantBase
   LinearNonlinearity
   SoftPlusNonlinearity
```

### Fitting

{class}`~hawkes_package.inference.HawkesEstimator` is the same fit behind a
scikit-learn-shaped surface — `fit`, `partial_fit`, `predict`, `score` — for callers who
would rather hold one object than four. It does not import scikit-learn and does not
require it.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   HawkesEstimator
   SMCSampler
   ParticleCloud
   SMCDiagnostics
   StepRecord
   Static
   RandomWalkDrift
   LiuWest
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   fit_smc
   block_boundaries
   metropolis_chain
   batch_posterior
   effective_sample_size
   systematic
   multinomial
```

### Checking and forecasting

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   residuals
   ks_exponential
   posterior_report
   posterior_predictive
   predictive_counts
   predictive_interval
```
