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

{class}`Circle` and {class}`Torus2D` are written out by hand.
{class}`FundamentalDomain` is the general construction they are instances of —
a convex polygon plus the side-pairing isometries that identify its boundary —
and reaches quotients no rectangle expresses, the hexagonal torus first among
them. Unlike the other two it does not fill its bounding box, which is what the
`contains` and `volume_element` hooks on the base class are for.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   SpatialDomain
   Circle
   Torus2D
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
