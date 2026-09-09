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

## Multivariate processes

Several event types exciting one another, through one shared kernel shape and
a non-negative matrix of scales.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   MultivariateHawkes
   MultivariateExponentialHawkes
   MultivariateTemporalHawkesProcess
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

{class}`Rectangle` and {class}`Polygon` are the odd ones out: they are **not**
closed surfaces but bounded regions with a real edge, and they set
`has_boundary`. On those the process renormalises each event's spatial kernel by
its own in-domain mass, because an event near the edge otherwise produces fewer
offspring than the model says — omitting that over-estimates the excitation by
64% on a 4×3 rectangle. `Rectangle(4, 3)` and `FundamentalDomain.rectangle(4, 3)`
share a bounding box and nothing else: the second glues its sides up and is a
torus.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   SpatialDomain
   Circle
   Torus2D
   Rectangle
   Polygon
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
   ExcitationMatrix
   MultivariateBase
   UnitExponentialKernel
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   exponential_model
   monotone_model
   bell_shape_model
   multivariate_model
   spatio_temporal_model
```

### Validation

`hawkes_package.inference.validation` answers a harder question than the
diagnostics above: not "do the two implementations of this model agree" but "is
the model right at all", judged without reusing the arithmetic that produced it.

That distinction is one specific cancellation. A fit made with a compensator 20%
too small inflates the intensity, and rescaling the events through that *same*
broken integral gives unit-rate gaps — so the goodness-of-fit test passes, and
the worse the compensator the more exactly the fit compensates for it. Measured
on 400 events: the broken compensator gives KS `p = 0.46`, an honest one
`p = 3.3e-05`. `independent_compensator` integrates the simulator's own hook on a
dense uniform grid, sharing only the parameter vector with the likelihood it
checks.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   validation.independent_compensator
   validation.compensator_agreement
   validation.cell_residuals
   validation.compare_with_baseline
   validation.homogeneous_log_likelihood
   validation.rolling_origin
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   validation.CellResiduals
   validation.BaselineComparison
   validation.Backtest
   validation.OriginScore
```

### Likelihoods

`ExponentialLogLikelihood` is the closed form and is `O(n)`;
`TemporalLogLikelihood` works for any temporal model through its intensity hook
and is `O(n²P)`; `SpatioTemporalLogLikelihood` carries two backends that compute
the same number, and always records which one ran.

For a multivariate model the pair repeats: `MultivariateLogLikelihood` goes
through the hooks, `MultivariateExponentialLogLikelihood` is the `O(n·d)`
closed form. Both sum the intensity of the type each event **carries** while
integrating the total across types -- using the total in both places inflates
the log-sum by 265 nats over 400 events, which is why `TemporalLogLikelihood`
refuses a multivariate model rather than reading its scalar hook.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   ExponentialLogLikelihood
   TemporalLogLikelihood
   MultivariateLogLikelihood
   MultivariateExponentialLogLikelihood
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

## Visualization

An optional extra; see {doc}`../visualization`. `intensity_frames` needs nothing
beyond numpy, and only `animate_intensity` reaches for the rendering backend.

```{eval-rst}
.. currentmodule:: hawkes_package.viz

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   animate_intensity
   build_figure
   intensity_frames
   embed
   event_opacities
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   IntensityFrames
   SurfaceEmbedding
```
