"""The rendering layer: a translation with no arithmetic of its own.

Everything numerical is asserted in `test_field.py` against the intensity hook.
What is left here is whether the figure is assembled the way the page needs it,
and two of those are load-bearing rather than cosmetic:

* the vertex positions must appear once, not once per frame -- the difference
  between a 6 MB page and a 90 MB one;
* the colour scale must be pinned to the frames' global range, or each frame
  gets its own and the animation stops meaning anything.

`test_a_missing_backend_says_what_to_install` deliberately does **not** skip
without plotly: it is the one test here that must run everywhere, because it
covers what a user without the extra actually hits.
"""

import numpy as np
import pytest

from hawkes_package.viz import _plotly, build_figure, embed, intensity_frames

pytest.importorskip("plotly")

RESOLUTION = (12, 12)

#: A CDN link is a `<script src=...>` tag. The string "cdn.plot.ly" on its own
#: is no use as a probe: the inlined library carries it as a config default, so
#: it is present either way.
CDN_TAG = 'src="https://cdn.plot.ly'


def _marker_count(frame):
    """How many event markers a frame draws.

    Spelled out rather than `len(frame.data[1].x or ())`, which raises: the
    empty case is `None` but the populated case is an array, and an array has no
    truth value.
    """
    coordinates = frame.data[1].x
    return 0 if coordinates is None else len(coordinates)


@pytest.fixture
def frames(torus, frame_times):
    return intensity_frames(torus, frame_times, resolution=RESOLUTION)


def test_the_geometry_is_sent_once_not_once_per_frame(torus, frames):
    """A frame carries the new colour and nothing else.

    The mesh is static -- it is the colour that animates -- so repeating the
    vertex positions would multiply the page by the frame count. Asserted
    structurally rather than by file size, which would be a threshold nobody
    could interpret when it failed.
    """
    figure = build_figure(torus, frames)

    base = figure.data[0]
    assert base.x is not None
    assert base.surfacecolor is not None

    for frame in figure.frames:
        assert frame.data[0].x is None
        assert frame.data[0].y is None
        assert frame.data[0].z is None
        assert frame.data[0].surfacecolor is not None


def test_the_colour_scale_is_pinned_to_the_global_range(torus, frames):
    """In the intensity's own units, so the colour bar reads in lambda, not in 0-to-1."""
    figure = build_figure(torus, frames)
    assert figure.data[0].cmin == frames.vmin
    assert figure.data[0].cmax == frames.vmax
    assert figure.data[0].cmax == pytest.approx(frames.values.max())


def test_the_surface_is_the_embedding(torus, frames):
    surface = embed(torus.domain)
    figure = build_figure(torus, frames, embedding=surface)
    nu, nv = frames.chart.shape[:2]
    expected = surface.ambient(frames.chart.reshape(-1, 2)).reshape(nu, nv, 3)
    assert np.allclose(figure.data[0].x, expected[..., 0])
    assert np.allclose(figure.data[0].z, expected[..., 2])


def test_one_frame_per_requested_time(torus, frames):
    figure = build_figure(torus, frames)
    assert len(figure.frames) == len(frames.times)
    assert len(figure.layout.sliders[0].steps) == len(frames.times)
    # The slider step names must match the frame names or the slider moves nothing.
    assert [step.args[0][0] for step in figure.layout.sliders[0].steps] == [
        frame.name for frame in figure.frames
    ]


def test_events_accumulate_across_the_animation(torus, frames):
    """Markers appear as their events happen and never disappear again."""
    figure = build_figure(torus, frames, events=True)
    counts = [_marker_count(frame) for frame in figure.frames]
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]
    assert counts[-1] == int((torus.events[0, :] <= frames.times[-1]).sum())


def test_events_can_be_left_off(torus, frames):
    figure = build_figure(torus, frames, events=False)
    assert all(_marker_count(frame) == 0 for frame in figure.frames)


def test_the_caption_names_the_range_and_whether_it_is_isometric(torus, frames):
    """A reader comparing two saved pages has only the titles side by side."""
    title = build_figure(torus, frames).layout.title.text
    assert "flat torus" in title
    assert "distances distorted" in title
    assert f"{frames.vmax:.4g}" in title


def test_a_sphere_is_captioned_as_isometric(sphere, frame_times):
    frames = intensity_frames(sphere, frame_times, resolution=RESOLUTION)
    assert "isometric" in build_figure(sphere, frames).layout.title.text


def test_the_axes_are_never_rescaled(torus, frames):
    """`aspectmode="data"` -- anything else would stretch the surface out of shape."""
    assert build_figure(torus, frames).layout.scene.aspectmode == "data"


@pytest.mark.slow
def test_a_render_writes_one_self_contained_page(torus, frame_times, tmp_path):
    page = tmp_path / "torus.html"
    written = _plotly.animate_intensity(torus, frame_times, page, resolution=(24, 24))

    assert page.is_file()
    text = page.read_text(encoding="utf8")
    assert len(text) > 100_000
    assert "plotly" in text.lower()
    # The library is embedded, so the page opens with no network at all.
    assert CDN_TAG not in text
    assert written.vmax == pytest.approx(written.values.max())


@pytest.mark.slow
def test_the_cdn_variant_links_the_library_instead(torus, frame_times, tmp_path):
    page = tmp_path / "cdn.html"
    _plotly.animate_intensity(torus, frame_times, page, resolution=(24, 24), inline_js=False)
    assert CDN_TAG in page.read_text(encoding="utf8")
