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
   LegacySpatioTemporalHawkesProcess
```

## Spatial domains

Implement {class}`SpatialDomain` to simulate on your own geometry: the
simulator works against that interface alone.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   SpatialDomain
   Circle
   Torus2D
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
