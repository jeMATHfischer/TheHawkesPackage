# Quickstart

## A temporal process

```python
import numpy as np
import hawkes_package as hp

# param = [mu, alpha, beta]: background rate, excitation size, decay rate
process = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
process.simulate(100)

print(process.Events[:5])  # the first five event times
```

`simulate(k)` appends `k` further events; calling it again continues the same
realisation rather than restarting it.

The constructor enforces the stationarity condition $\alpha/\beta < 1$ and
raises `ValueError` otherwise — an unstable process would never terminate:

```python
hp.ExponentialHawkes(np.array([1.0, 5.0, 1.0]))
# ValueError: Stability condition violated: alpha/beta = 5.0000 >= 1.
```

## The intensity

```python
times, intensity = process.intensity_over_interval(np.linspace(0, 20, 500))
```

The realised event times are merged into the grid, so `times` is generally
longer than what you pass in. The value reported *at* an event time is the
left limit — the pre-jump value.

```python
import matplotlib.pyplot as plt

plt.plot(times, intensity)
plt.xlabel("time")
plt.ylabel(r"$\lambda(t \mid H_t)$")
plt.show()
```

## Reproducibility

Every process takes `rng=`, accepting `None`, an integer seed, or an existing
{class}`numpy.random.Generator`:

```python
hp.ExponentialHawkes(param, rng=42)  # reproducible
hp.ExponentialHawkes(param, rng=my_generator)  # share one stream
```

:::{warning}
As of 0.2.0, `np.random.seed(...)` does **not** control simulations. Scripts
written against 0.0.1 that relied on it still run, but produce different
numbers. See [migration](migration.md).
:::

## Other kernel shapes

```python
# Any monotone-decreasing kernel, with an optional monotone nonlinearity
H = hp.MonotoneKernelHawkes(lambda t: 0.9 * np.exp(-2 * t), nonlinearity=lambda x: x + 2, rng=0)


# A kernel that rises to a peak before decaying
def triangular(t):
    return 2 * t * ((t > 0) & (t < 0.5)) + (-2 * t + 2) * ((t >= 0.5) & (t < 1))


K = hp.BellShapeHawkes(triangular, rng=0)
K.simulate(50)
print(f"kernel peaks at lag {K.ext:.2f}")
```

## A spatio-temporal process

Events carry a location as well as a time:

```python
P = hp.SpatioTemporalHawkesProcess(
    base=lambda x: 0.5,  # background intensity mu(x)
    spatial=lambda d: max(0.0, 1 - d / np.pi),  # kernel of the distance
    temporal=lambda dt: 0.9 * np.exp(-5 * dt),
    domain=hp.Circle(),
    monotone_temporal_kernel=True,  # enables a tighter bound
    rng=42,
)
P.simulate(50)

P.Events.shape  # (2, 50): row 0 is times, row 1 the coordinate
P.intensity(1.5, [0.0])  # lambda at one time and place
```

Swap `hp.Circle()` for `hp.Torus2D()` to simulate in two spatial dimensions;
nothing else changes. For the whole field on a grid:

```python
times, points, field = P.intensity_over_interval(np.linspace(0, 10, 200))
# field is (n_points, n_times) — rows index space, columns index time
plt.contourf(times, points[:, 0], field, 50)
```

`points` defaults to 200 equispaced points across a one-dimensional domain, and
is required in two or more dimensions.

## Which surface

`domain=` is the only thing that changes, and between them the built-in domains
reach every closed surface:

```python
hp.Circle()  # the circle
hp.Sphere()  # the sphere
hp.Torus2D()  # the flat torus, written out by hand
hp.FundamentalDomain.rectangle(3, 5)  # the same torus, as a glued polygon
hp.FundamentalDomain.hexagon(1.0)  # the hexagonal torus
hp.FundamentalDomain.klein_bottle(3, 5)  # non-orientable, and still flat
hp.FundamentalDomain.projective_plane()  # a hemisphere, antipodes identified
hp.FundamentalDomain.genus(2)  # two handles; hyperbolic, necessarily
hp.FundamentalDomain.crosscaps(3)  # three crosscaps; hyperbolic too
```

The geometry is not a setting. A surface's Euler characteristic fixes the sign
of its curvature through Gauss–Bonnet, so the torus and the Klein bottle are
flat, the sphere and the projective plane are spherical, and everything else is
hyperbolic. Each domain reports what it built:

```pycon
>>> hp.FundamentalDomain.klein_bottle().topology
Topology(orientable=False, euler_characteristic=0, genus=2, name='Klein bottle')
```

A hyperbolic domain is expensive — its deck group is infinite, and a distance
searches a window of it — so keep runs on one short, and expect the quadrature
to want four times the nodes per axis a flat polygon does.

## Your own domain

Subclass {class}`~hawkes_package.spatio_temporal.domains.SpatialDomain` and
implement `distance`, `wrap`, `sample_uniform`, `volume` and `bounds`. The
simulator works against that interface alone.
