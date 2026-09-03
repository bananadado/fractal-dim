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


def plot_boxcount(result, *, reference: float | None = None, size: float = DEFAULT_SIZE,
                  background: str = "white", title: str | None = None):
    """Two panels: the log-log counts, and the local slope that justifies them.

    The second panel is the one that matters.  A fitted line through box counts
    always produces a number; only a plateau in the local slope says that the
    number is a dimension rather than an average of two roll-offs.
    """
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(size, size * 0.9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    figure.patch.set_facecolor(background)

    low, high = result.window
    used = result.used
    for axes in (top, bottom):
        axes.set_facecolor(background)
        axes.axvspan(low, high, color="0.92", zorder=0)
        axes.set_xscale("log")

    top.plot(result.eps[~used], result.counts[~used], "o", ms=4.5,
             mfc="none", color="0.55", label="outside the window")
    top.plot(result.eps[used], result.counts[used], "o", ms=5,
             color="#2b6cb0", label="fitted")
    line = np.exp(result.intercept) * result.eps ** (-result.dimension)
    top.plot(result.eps, line, "-", lw=1.2, color="#c53030",
             label=f"slope {result.dimension:.4f}")
    top.set_yscale("log")
    top.set_ylabel("N(eps)")
    top.legend(frameon=False, fontsize=9)
    if title:
        top.set_title(title, fontsize=11)

    bottom.plot(result.slope_centres, result.slopes, "o-", ms=4, lw=1.0,
                color="#2b6cb0")
    bottom.axhline(result.dimension, color="#c53030", lw=1.2,
                   label=f"fit {result.dimension:.4f}")
    if reference is not None:
        bottom.axhline(reference, color="#2f855a", lw=1.2, ls="--",
                       label=f"exact {reference:.4f}")
    bottom.axhline(1.0, color="0.6", lw=0.8, ls=":")
    bottom.set_ylabel("local slope")
    bottom.set_xlabel("box size eps")
    bottom.legend(frameon=False, fontsize=9)

    figure.tight_layout()
    return figure


def plot_grid(segments: np.ndarray, eps: float, *, origin=(0.0, 0.0),
              size: float = DEFAULT_SIZE, background: str = "white",
              ax=None, title: str | None = None, show_lines: bool = True):
    """The curve with every occupied box shaded: box counting, drawn.

    This is the picture the log-log plot is an abstraction of -- one point on
    that plot is one of these grids, and its N is the number of shaded cells.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Rectangle

    from .boxcount import occupied_cells

    if ax is None:
        _, ax = plt.subplots(figsize=(size, size))

    cells = occupied_cells(segments, eps, origin)
    boxes = [
        Rectangle((origin[0] + i * eps, origin[1] + j * eps), eps, eps)
        for i, j in cells
    ]
    ax.add_collection(PatchCollection(
        boxes, facecolor="#90cdf4", edgecolor="#2b6cb0",
        linewidth=0.3 if show_lines else 0.0, alpha=0.55, zorder=1,
    ))
    ax.add_collection(LineCollection(
        segments, colors="#1a202c",
        linewidths=_linewidth_for(len(segments), 1.4), zorder=2,
    ))
    label = title if title is not None else f"eps = {eps:g},  N = {len(cells)}"
    return _frame_axes(segments, ax, background, label)


def plot_grid_series(segments: np.ndarray, sizes, *, size: float = DEFAULT_SIZE,
                     background: str = "white", title: str | None = None):
    """A panel per box size, laid out to suit the curve's own proportions.

    A wide, short curve like Koch's wants its panels stacked; a square one like
    Hilbert's wants a grid.  Choosing by aspect ratio keeps the drawn figure
    filling its panel instead of floating in whitespace.
    """
    import matplotlib.pyplot as plt

    sizes = list(sizes)
    points = segments.reshape(-1, 2)
    extent = points.max(axis=0) - points.min(axis=0)
    aspect = float(extent[1] / extent[0]) if extent[0] > 0 else 1.0

    if aspect < 0.5:            # wide and short: stack them
        rows, columns = len(sizes), 1
    elif aspect > 2.0:          # tall and narrow: lay them out in a row
        rows, columns = 1, len(sizes)
    else:
        columns = 1 if len(sizes) == 1 else 2
        rows = int(np.ceil(len(sizes) / columns))

    panel = size / columns
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(size, max(1.2, panel * np.clip(aspect, 0.25, 3.0)) * rows + 0.3),
        squeeze=False,
    )
    figure.patch.set_facecolor(background)
    flat = axes.ravel()
    for eps, axis in zip(sizes, flat):
        plot_grid(segments, eps, ax=axis, background=background)
    for spare in flat[len(sizes):]:
        spare.axis("off")
    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    return figure
