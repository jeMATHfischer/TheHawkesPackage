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
