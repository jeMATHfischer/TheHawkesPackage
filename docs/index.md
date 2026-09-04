---
sd_hide_title: true
---

# the-hawkes-package

Simulation of temporal and spatio-temporal Hawkes processes via Ogata's thinning
algorithm.

A Hawkes process is a self-exciting point process: every event raises the
probability of further events for a while afterwards. Its conditional intensity is

$$
\lambda(t \mid H_t) = \varphi\!\left( \mu + \sum_{t_i < t} \kappa(t - t_i) \right),
$$

where $\kappa$ is the excitation kernel and $\varphi$ an optional monotone
nonlinearity.

This package simulates them for a range of kernel shapes — including non-monotone
"bell-shaped" kernels, where the naive thinning bound is invalid — and extends
the construction to a spatial domain with periodic boundaries. Since 0.5.0 it
also **fits** them: `hawkes_package.inference` estimates parameters from observed
events by sequential Monte Carlo, in blocks as the data arrives, using a
likelihood computed from the simulator's own intensity hooks.

```{code-block} python
import numpy as np
import hawkes_package as hp

process = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=42)
process.simulate(100)
times, intensity = process.intensity_over_interval(np.linspace(0, 20, 500))
```

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install from PyPI, and what you need to build from source.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Simulate a process and plot its intensity in ten lines.
:::

:::{grid-item-card} Theory
:link: theory
:link-type: doc

Ogata's thinning, stability conditions, and the spatio-temporal intensity.
:::

:::{grid-item-card} Inference
:link: inference
:link-type: doc

Fit a process to data, read the diagnostics, and forecast from the posterior.
:::

:::{grid-item-card} Visualization
:link: visualization
:link-type: doc

Render the intensity as an animated 3-D surface: sphere, torus, Klein bottle or
projective plane.
:::

:::{grid-item-card} Migration
:link: migration
:link-type: doc

What 0.5.0 removed, what 0.4.0 renamed, and what they ask of a custom domain.
:::

::::

## Process classes

| Class | Use when |
|---|---|
| {class}`~hawkes_package.exponential.ExponentialHawkes` | Linear intensity, exponential kernel. The classic case. |
| {class}`~hawkes_package.monotone.MonotoneKernelHawkes` | Any monotone-decreasing kernel, with a monotone-increasing nonlinearity. |
| {class}`~hawkes_package.bell_shape.BellShapeHawkes` | Kernels with a single interior maximum: excitation ramps up before decaying. |
| {class}`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess` | Events carry a location on a closed surface, or on your own domain. |

## Fitting

| Class | Use when |
|---|---|
| {class}`~hawkes_package.inference.smc.SMCSampler` | Estimate parameters from observed events, in blocks as they arrive. |
| {class}`~hawkes_package.inference.likelihood.ExponentialLogLikelihood` | The exponential kernel, whose likelihood is closed-form and `O(n)`. |
| {class}`~hawkes_package.inference.likelihood.TemporalLogLikelihood` | Any other temporal kernel, through the hook the simulator thins against. |
| {class}`~hawkes_package.inference.likelihood.SpatioTemporalLogLikelihood` | Events with locations, on any {class}`~hawkes_package.SpatialDomain`. |

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
theory
inference
visualization
```

```{toctree}
:hidden:
:caption: Examples

examples/temporal_processes
examples/spatio_temporal
examples/surfaces
examples/intensity_surfaces
examples/online_inference
```

```{toctree}
:hidden:
:caption: Reference

api/index
migration
changelog
contributing
```
