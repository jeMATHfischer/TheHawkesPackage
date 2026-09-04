# Visualization

`hawkes_package.viz` draws the surface a spatio-temporal process lives on, colours it by
$\lambda(t, x \mid H_t)$, and animates it over time into one self-contained interactive
page — a play button, a frame slider, and a camera that stays orbitable while it plays.

For the four surfaces drawn and animated live, see
{doc}`Seeing the intensity on a surface <examples/intensity_surfaces>`. This page is the
reference: what each picture claims, what it costs, and how to work with the numbers
behind it.

```{eval-rst}
.. currentmodule:: hawkes_package.viz
```

## Installing

The renderer is plotly, and it is an optional extra:

```bash
pip install "the-hawkes-package[viz]"
```

It is **not** a runtime dependency. {func}`intensity_frames` builds the field from the
simulator's own intensity hooks with nothing but numpy, so the numbers can be computed,
inspected and tested without the extra; only writing the page needs it. Attempting a
render without it raises an `ImportError` that says exactly this.

## One call

```python
import numpy as np
import hawkes_package as hp
from hawkes_package.spatio_temporal import FundamentalDomain
from hawkes_package.viz import animate_intensity

bottle = FundamentalDomain.klein_bottle(3.0, 5.0)
process = hp.SpatioTemporalHawkesProcess(
    base=lambda x: 0.3,
    spatial=lambda d: 0.8 * np.exp(-2.0 * d),
    temporal=lambda s: 1.5 * np.exp(-1.0 * s),
    domain=bottle,
    monotone_temporal_kernel=True,
    rng=0,
)
process.simulate(25)

frames = animate_intensity(
    process,
    np.linspace(0.0, process.events[0, -1], 40),
    "klein.html",
)
print(frames.summary())
# 40 frames on a 48x48 grid, lambda in [0, 1.67], factorised
```

Open `klein.html`: a figure-8 Klein bottle you can orbit, brightening where events land
and decaying between them, with each event marked and fading at its own kernel's rate.

{func}`animate_intensity` **returns** the frames rather than discarding them, because the
realised colour range is not otherwise recoverable and two runs can only be compared once
they are on one scale — pass one run's `vmax` to the other.

## What each picture claims

Four surfaces are supported, and they are not equally honest. The distinction is recorded
on {attr}`SurfaceEmbedding.isometric` and spelled out in
{attr}`SurfaceEmbedding.note`, which the page's caption carries.

| Domain | Drawn as | Distances |
|---|---|---|
| {class}`~hawkes_package.spatio_temporal.Sphere` | itself | **exact** |
| `FundamentalDomain.projective_plane()` | its double cover, the sphere | **exact** — the covering map is a local isometry |
| {class}`~hawkes_package.spatio_temporal.Torus2D`, `FundamentalDomain.rectangle()` | a donut | distorted |
| `FundamentalDomain.klein_bottle()` | the figure-8 immersion | distorted, and self-intersecting |

The flat torus admits no isometric $C^2$ embedding in $\mathbb{R}^3$ and the Klein bottle
admits no embedding at all, so for those two **the geodesic distances driving the
intensity are not the distances you measure on screen**. The outer rim of the donut is
stretched and the inner rim compressed; the Klein bottle's colouring appears mirrored
after one turn around its base circle, which is the surface and not an artefact.

The projective plane is worth a sentence of its own. Its fundamental polygon is a
hemisphere, but the picture is the whole sphere: the deck group is the antipodal map, so
the colouring comes out antipodally symmetric, and that symmetry *is* the identification
made visible. No Boy or Roman surface is attempted — every one of them distorts the
distances the colour encodes.

Hyperbolic surfaces — genus 2 and above, three or more crosscaps — are **refused** at
{func}`embed` rather than approximated. By Hilbert's theorem no complete surface of
constant negative curvature embeds isometrically in three-space, and a solid that
pretended otherwise would be making a false claim about the geometry.

## Cost, and the size of the page

Every value is $\lambda$ evaluated through the hooks the simulator thins against, so the
cost is `domain.distance` — 7 µs on a torus, 25 µs on a sphere, 142 µs on a Klein bottle
and 327 µs on the projective plane, per call.

{func}`intensity_frames` hoists the time-independent factors out of the frame loop, which
turns `n_grid × n_frames × n_events` distance evaluations into `n_grid × n_events`. At
64×64 over 40 frames with 25 events that is the difference between 22 minutes and 33
seconds on the projective plane. The hoist is a rearrangement of the same sum and is
**bit-exact** against `process.intensity`; it checks itself against the hook on a sample
of finished values and raises rather than degrading. Pass `fast=False` to evaluate
through the hook directly.

The page carries a colour per vertex per frame, roughly `nu × nv × n_frames × 150` bytes,
plus about 5 MB for the embedded plotly library. A 48×48 grid over 40 frames is about
6 MB. Pass `inline_js=False` to link a CDN instead and drop it to 1.4 MB, at the cost of
a page that is blank offline.

## Working with the frames directly

{func}`intensity_frames` is the piece worth reaching for if you want the numbers rather
than the picture — to check a value, to feed a different renderer, or to put two runs on
one scale.

```python
from hawkes_package.viz import embed, intensity_frames

frames = intensity_frames(process, np.linspace(0, 10, 40), resolution=(64, 64))
frames.values  # (n_t, nu, nv), lambda, floored at zero
frames.chart  # (nu, nv, 2), the chart points it was evaluated at
frames.normalised(0)  # that frame rescaled to [0, 1] through the *global* range
```

Two details that matter for reading a value back:

- `frames.chart` holds the grid **after** {meth}`~hawkes_package.SpatialDomain.wrap`, so
  checking a value against `process.intensity` must use those points and not the raw
  grid. On the projective plane the difference is not cosmetic: `distance` folds its own
  arguments, but a chart-dependent `base` would be sampled outside the polygon it was
  written for and the picture would tear along the equator.
- The colour range is **global across every frame**. A per-frame rescale would make a
  quiet frame look as hot as a burst, which destroys the one thing an animation of a
  self-exciting process is for.

## What is not here

The 1-D {class}`~hawkes_package.spatio_temporal.Circle` has no 3-D picture worth drawing;
its intensity field is already a 2-D contour plot, shown in the
{doc}`spatio-temporal example <examples/spatio_temporal>`. Hyperbolic surfaces are
refused, as above. There is no video export: the page *is* the artefact, and it is
interactive in a way a video is not.

## Where to go next

- {doc}`Seeing the intensity on a surface <examples/intensity_surfaces>` -- all four
  surfaces, drawn and animated, with the flip and the antipodal symmetry checked in code
  rather than asserted in prose.
- {doc}`Compact surfaces from fundamental domains <examples/surfaces>` -- where these
  surfaces come from, including the hyperbolic ones no picture reaches.
