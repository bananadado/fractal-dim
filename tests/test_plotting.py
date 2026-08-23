"""Rendering stills and animations."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from fractaldim import library
from fractaldim.plotting import DEFAULT_DPI, animate_segments, save_segments
from fractaldim.turtle import trace


@pytest.fixture
def koch():
    return trace(library.get("koch").system, 3)


def test_still_resolution_follows_size_and_dpi(tmp_path, koch):
    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    save_segments(koch, small, dpi=50, size=4.0, title=None)
    save_segments(koch, large, dpi=100, size=4.0, title=None)
    assert Image.open(large).size[0] > Image.open(small).size[0]


def test_default_still_is_high_resolution(tmp_path, koch):
    """The defaults should comfortably clear 1500px on the long edge.

    Only the long edge is checked: bbox_inches="tight" crops away the unused
    margin, so a wide, short curve like Koch legitimately comes out short in
    the other direction.
    """
    wide = tmp_path / "wide.png"
    save_segments(koch, wide)
    assert max(Image.open(wide).size) > 1500

    # A square curve should clear it in both directions.
    square = tmp_path / "square.png"
    save_segments(trace(library.get("hilbert").system, 4), square)
    assert min(Image.open(square).size) > 1500
    assert DEFAULT_DPI >= 300


def test_solid_colour_is_accepted(tmp_path, koch):
    path = tmp_path / "solid.png"
    save_segments(koch, path, color="black", dpi=50, size=3.0)
    assert path.exists()


def test_animation_frame_count_and_hold(tmp_path, koch):
    path = tmp_path / "anim.gif"
    animate_segments(koch, path, frames=20, hold=5, dpi=40, size=3.0, fps=10)
    image = Image.open(path)
    # Pillow folds the identical hold frames into the last frame's duration
    # rather than storing them, so count distinct reveal steps instead.
    assert image.n_frames == 20
    durations = []
    for index in range(image.n_frames):
        image.seek(index)
        durations.append(image.info["duration"])
    assert durations[-1] > durations[0], "the finished curve should hold"


def test_animation_never_asks_for_more_frames_than_segments(tmp_path):
    segments = trace(library.get("koch").system, 1)
    path = tmp_path / "short.gif"
    animate_segments(segments, path, frames=500, hold=0, dpi=40, size=3.0)
    assert Image.open(path).n_frames == len(segments)


def test_animating_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no segments"):
        animate_segments(np.empty((0, 2, 2)), tmp_path / "empty.gif")


def test_progress_callback_reports_completion(tmp_path, koch):
    seen = []
    animate_segments(koch, tmp_path / "p.gif", frames=8, hold=0, dpi=40,
                     size=3.0, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1][0] == seen[-1][1]


def test_animated_webp_is_refused(tmp_path, koch):
    """Pillow drops WebP frame durations, so the timing would be silently wrong."""
    with pytest.raises(ValueError, match="WebP"):
        animate_segments(koch, tmp_path / "anim.webp", frames=4, dpi=40, size=3.0)


def test_apng_output_keeps_frame_timing(tmp_path, koch):
    path = tmp_path / "anim.png"
    animate_segments(koch, path, frames=10, hold=6, fps=10, dpi=40, size=3.0)
    image = Image.open(path)
    assert image.n_frames == 10
    durations = []
    for index in range(image.n_frames):
        image.seek(index)
        durations.append(image.info["duration"])
    assert durations[-1] > durations[0]
