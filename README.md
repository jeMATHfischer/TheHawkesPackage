# the-hawkes-package

Simulation of temporal and spatio-temporal Hawkes processes via Ogata's thinning algorithm.

A Hawkes process is a self-exciting point process: every event raises the probability of further
events for a while afterwards. This package simulates them for a range of kernel shapes — including
non-monotone ("bell-shaped") kernels and nonlinear intensities, where the naive thinning bound is
wrong — and extends the construction to a spatial domain with periodic boundaries.

```
λ(t | H_t) = φ( μ + Σ_{t_i < t} κ(t − t_i) )
```

## Installation

```bash
pip install the-hawkes-package
```

Requires Python 3.10+, NumPy and SciPy.

## Quickstart

```python
import numpy as np
import hawkes_package as hp

# Linear Hawkes with an exponential kernel: param = [mu, alpha, beta]
process = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
process.simulate(100)

times, intensity = process.intensity_over_interval(np.linspace(0, process.Events[-1], 1000))
```

The constructor enforces the stationarity condition `alpha / beta < 1` and raises `ValueError`
otherwise — an unstable process would not terminate.

### Available processes

| Class | Use when |
|---|---|
| `ExponentialHawkes` | Linear intensity, exponential kernel. The classic case. |
| `MonotoneKernelHawkes` | Any monotone-decreasing kernel, with a monotone-increasing nonlinearity `φ`. |
| `BellShapeHawkes` | Kernels with a single interior maximum, where the excitation ramps up before decaying. |
| `SpatioTemporalHawkesProcess` | Events carry a location on a `Circle` or `Torus2D`. |

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
to simulate on your own geometry, subject to `volume == prod(bounds widths)`.

`make_periodic` wraps an isotropic kernel so it sums correctly over the domain's image points, and
can be passed straight in as `spatial=`:

```python
kernel = hp.make_periodic(lambda d: np.exp(-(d**2)), hp.Circle())
process = hp.SpatioTemporalHawkesProcess(base, kernel, temporal, domain=hp.Circle())
```

## Reproducibility

Every process takes `rng=`, accepting `None`, an `int` seed, or an existing `numpy.random.Generator`:

```python
hp.ExponentialHawkes(param, rng=42)  # reproducible
hp.ExponentialHawkes(param, rng=my_generator)  # share one stream
```

`np.random.seed(...)` does **not** control simulations. See [CHANGELOG.md](CHANGELOG.md).

## Migrating from `TheHawkesPackage`

The import name changed to `hawkes_package`. `import TheHawkesPackage` still works but emits a
`DeprecationWarning`; the shim is removed in 0.4.0. `simulate(k)` is now the method on every
process class — `propagate_by_amount`, `propagate_by_k_events` and the `propogate_by_amount` typo
remain as deprecated aliases.

## Documentation

<https://jeMATHfischer.github.io/TheHawkesPackage/>

## License

MIT — see [LICENSE](LICENSE).
