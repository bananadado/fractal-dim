"""Rendering of traced curves, as stills and as animations."""

from __future__ import annotations

import numpy as np

#: Figure edge in inches and dots per inch. Their product is roughly the pixel
#: width of the result, before ``bbox_inches="tight"`` trims the margins.
DEFAULT_SIZE = 8.0
DEFAULT_DPI = 300


def _linewidth_for(count: int, scale: float = 1.0) -> float:
    """Thin the stroke as the curve gets denser, or deep levels blot out."""
    return float(np.clip(60.0 / np.sqrt(max(count, 1)), 0.12, 1.6)) * scale


def _frame_axes(segments: np.ndarray, ax, background: str, title: str | None):
    """Fix the view to the whole curve and strip the axes."""
    if segments.size:
        points = segments.reshape(-1, 2)
        lo, hi = points.min(axis=0), points.max(axis=0)
        pad = 0.02 * max(float((hi - lo).max()), 1e-9)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_facecolor(background)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def plot_segments(
    segments: np.ndarray,
    *,
    ax=None,
    color: str = "index",
    cmap: str = "viridis",
    linewidth: float | None = None,
    linewidth_scale: float = 1.0,
    background: str = "white",
    title: str | None = None,
    size: float = DEFAULT_SIZE,
):
    """Draw traced segments and return the axes.

    ``color="index"`` shades each segment by its position along the word, which
    shows the order the turtle visits the plane -- worth seeing for a
    space-filling curve, where the drawn figure alone hides the traversal.
    ``color`` may also be any matplotlib colour, used as a solid stroke.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(size, size))

    if segments.size:
        if linewidth is None:
            linewidth = _linewidth_for(len(segments), linewidth_scale)
        collection = LineCollection(segments, linewidths=linewidth)
        if color == "index":
            collection.set_array(np.linspace(0.0, 1.0, len(segments)))
            collection.set_cmap(cmap)
            collection.set_clim(0.0, 1.0)
        else:
            collection.set_color(color)
        ax.add_collection(collection)

    return _frame_axes(segments, ax, background, title)


def save_segments(segments: np.ndarray, path, *, dpi: int = DEFAULT_DPI, **kwargs) -> None:
    """Render ``segments`` to an image file."""
    import matplotlib.pyplot as plt

    background = kwargs.get("background", "white")
    ax = plot_segments(segments, **kwargs)
    figure = ax.get_figure()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=background)
    plt.close(figure)


def animate_segments(
    segments: np.ndarray,
    path,
    *,
    frames: int = 150,
    fps: int = 20,
    hold: int = 12,
    dpi: int = 150,
    size: float = DEFAULT_SIZE,
    color: str = "index",
    cmap: str = "viridis",
    linewidth: float | None = None,
    linewidth_scale: float = 1.0,
    background: str = "white",
    title: str | None = None,
    head: bool = True,
    progress=None,
) -> None:
    """Animate the turtle drawing ``segments``, writing a GIF.

    The curve is revealed a slice at a time, with a marker at the turtle's
    current position.  The view is fixed to the finished figure from the first
    frame, so the drawing grows into a stable frame rather than rescaling.

    ``frames`` is the number of steps the reveal is divided into, and ``hold``
    adds still frames at the end so the completed curve lingers before the loop
    restarts.  Uses Pillow, so it needs no ffmpeg.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    if segments.size == 0:
        raise ValueError("nothing to animate: the trace produced no segments")

    # Pillow accepts a .webp animation but silently discards the frame
    # durations, so the result plays at whatever rate the viewer picks and the
    # hold disappears. Refuse it rather than write a file that looks fine here
    # and plays wrong everywhere else.
    if str(path).lower().endswith(".webp"):
        raise ValueError(
            "animated WebP loses frame timing in Pillow; use .gif for "
            "compatibility, or .png for an APNG (much smaller, but not every "
            "viewer animates it)"
        )

    total = len(segments)
    frames = max(1, min(frames, total))
    if linewidth is None:
        linewidth = _linewidth_for(total, linewidth_scale)

    figure, ax = plt.subplots(figsize=(size, size))
    collection = LineCollection(np.empty((0, 2, 2)), linewidths=linewidth)
    shades = np.linspace(0.0, 1.0, total)
    if color == "index":
        collection.set_cmap(cmap)
        collection.set_clim(0.0, 1.0)
    else:
        collection.set_color(color)
    ax.add_collection(collection)
    _frame_axes(segments, ax, background, title)

    marker, = ax.plot(
        [], [], "o",
        markersize=5.0 if head else 0.0,
        color="crimson", zorder=5, visible=head,
    )

    # Reveal counts, then a run of repeats so the finished curve holds.
    counts = np.unique(np.linspace(1, total, frames).astype(int))
    counts = np.concatenate([counts, np.repeat(counts[-1], max(hold, 0))])

    def update(index):
        drawn = int(counts[index])
        collection.set_segments(segments[:drawn])
        if color == "index":
            collection.set_array(shades[:drawn])
        if head:
            tip = segments[drawn - 1, 1]
            marker.set_data([tip[0]], [tip[1]])
            # Hide the marker once the curve is complete and merely holding.
            marker.set_visible(drawn < total)
        if progress is not None:
            progress(index + 1, len(counts))
        return collection, marker

    animation = FuncAnimation(
        figure, update, frames=len(counts), interval=1000 / fps, blit=False
    )
    animation.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi,
                   savefig_kwargs={"facecolor": background})
    plt.close(figure)
