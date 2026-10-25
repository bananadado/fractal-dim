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


#: Blue ordinal ramp, light to dark, for series keyed by L-system level.  Level
#: is an ordered quantity, so it takes a single-hue ramp rather than
#: categorical hues: darker is deeper, and the reader needs no legend lookup to
#: see which way the family is moving.  Steps 250-700 of the sequential blue;
#: the light end is held at 250 so the palest line still clears the surface.
_LEVEL_RAMP = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: Ink and reference colours, matching the rest of the module.
_EXACT = "#2f855a"
_RULE = "#9b9b96"
_TEXT = "#52514e"

#: Smallest error the roll-off panel draws.  A dyadic curve on a dyadic grid
#: can land on its exact dimension, and zero has no place on a log axis; such
#: points are drawn at this floor and called what they are.
_ERROR_FLOOR = 1e-5


def _level_colors(count: int) -> list[str]:
    """``count`` steps spread over the ramp, darkest last."""
    if count <= 1:
        return [_LEVEL_RAMP[-3]]
    picks = np.linspace(0, len(_LEVEL_RAMP) - 1, count).round().astype(int)
    return [_LEVEL_RAMP[index] for index in picks]


def _hairline_grid(axes) -> None:
    """A solid, recessive grid: one shade off the surface, never dashed."""
    axes.grid(True, which="major", color="0.90", lw=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("0.75")
        axes.spines[side].set_linewidth(0.6)


def plot_rolloff(sweeps, *, size: float = DEFAULT_SIZE, background: str = "white",
                 title: str | None = None):
    """The V: what counting at ever finer boxes does to the estimate.

    Top panel, the mechanism -- the local slope at each box size, which holds
    near the dimension over the scaling window and falls to 1 once the boxes
    are smaller than a step, where the figure is just line segments.

    Bottom panel, the consequence -- the error of a fit taken from that box
    size up to a fixed coarse end.  It is V-shaped: letting finer boxes into
    the fit makes it worse, and the floor sits a few step lengths up, not at
    the smallest box affordable.  Deeper levels move the whole V down and left,
    which is the only thing that actually improves the estimate.

    Both panels stop at the coarse end of the fits.  Past it the counts are a
    handful of boxes around the outline and the slope is meaningless; leaving
    that tail in stretches the axis and buries the part being argued about.
    """
    import matplotlib.pyplot as plt

    sweeps = list(sweeps)
    if not sweeps:
        raise ValueError("nothing to plot: no sweeps given")
    exact = sweeps[0].exact
    if exact is None:
        raise ValueError("the roll-off plot needs a reference dimension to "
                         "measure error against")

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(size, size * 0.95), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    figure.patch.set_facecolor(background)
    colors = _level_colors(len(sweeps))

    # Everything is measured in step lengths, which is what aligns the roll-off
    # across levels: each figure leaves the fractal at its own step, so plotting
    # against a raw eps would scatter the one feature the panels are about.
    left = min(float(s.eps[0] / s.step) for s in sweeps) / 1.4
    right = max(float(s.top / s.step) for s in sweeps) * 1.4

    for axes in (top, bottom):
        axes.set_facecolor(background)
        axes.set_xscale("log")
        _hairline_grid(axes)
        # The step length is where the approximation stops resembling the
        # fractal, so it is the one landmark both panels are read against.
        axes.axvline(1.0, color=_RULE, lw=0.9, zorder=1)

    shown = []
    for index, (sweep, color) in enumerate(zip(sweeps, colors)):
        inside = sweep.slope_centres <= sweep.top * (1 + 1e-9)
        shown.append(sweep.slopes[inside])
        # Shallow levels are drawn over deep ones.  On a curve whose scaling
        # factor matches the grid the slopes coincide exactly, and painting
        # deepest-last would hide every level but the longest.
        top.plot(sweep.slope_centres[inside] / sweep.step, sweep.slopes[inside],
                 "-", lw=1.4, color=color, zorder=3 + len(sweeps) - index,
                 label=f"level {sweep.level}")
        bottom.plot(sweep.cutoffs / sweep.step,
                    np.maximum(np.abs(sweep.errors), _ERROR_FLOOR), "-o",
                    lw=1.4, ms=4.5, mfc=background, mew=1.2, color=color,
                    zorder=3, label=f"level {sweep.level}")

    top.axhline(exact, color=_EXACT, lw=1.2, ls="--", zorder=2,
                label=f"exact {exact:.4f}")
    top.axhline(1.0, color=_RULE, lw=1.0, ls=":", zorder=2,
                label="line segments (1)")

    span = np.concatenate(shown)
    top.set_xlim(left, right)
    # Floored well under 1 so the "line segments" rule and the step-length
    # label below it both sit clear of the axis and of each other, and headroom
    # taken as a fraction of the range so the legend never lands on the
    # reference line -- which it would on a space-filling curve, where that
    # line is also the top of the data.
    floor = min(0.92, float(span.min()) - 0.06)
    ceiling = max(exact, float(span.max()))
    rows = -(-(len(sweeps) + 2) // 2)       # legend entries over two columns
    top.set_ylim(floor, ceiling + (0.12 + 0.07 * rows) * (ceiling - floor))
    top.set_ylabel("local slope")
    top.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    top.annotate("step length", xy=(1.0, 0.015), xycoords=("data", "axes fraction"),
                 ha="left", va="bottom", fontsize=8, color=_TEXT,
                 xytext=(4, 0), textcoords="offset points")
    if title:
        top.set_title(title, fontsize=11)

    # Direct-label the floor of the deepest V: the number the panel exists to
    # make, rather than a value beside every point.
    deepest = sweeps[-1]
    best = deepest.best
    at = (deepest.cutoffs[best] / deepest.step,
          max(abs(deepest.errors[best]), _ERROR_FLOOR))
    if any(np.abs(s.errors).min() < _ERROR_FLOOR for s in sweeps):
        bottom.axhline(_ERROR_FLOOR, color=_RULE, lw=0.9, ls=":", zorder=2)
        bottom.annotate(
            f"exact (error below {_ERROR_FLOOR:g})",
            xy=(0.01, _ERROR_FLOOR), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points",
            fontsize=8, color=_TEXT, ha="left", va="bottom",
        )
    bottom.plot(*at, "o", ms=9, mfc="none", mec=colors[-1], mew=1.4, zorder=5)
    bottom.set_yscale("log")
    # Open the floor of the axis first, so the label below the ring has room
    # that no series is using.
    low, high = bottom.get_ylim()
    bottom.set_ylim(low / 3.2, high)
    bottom.annotate(
        f"best at eps = {at[0]:g} steps, error {deepest.errors[best]:+.4f}",
        xy=at, xytext=(10, -8), textcoords="offset points",
        fontsize=8.5, color=_TEXT, ha="left", va="top",
    )
    bottom.set_ylabel("|fitted dimension - exact|")
    bottom.set_xlabel("finest box let into the fit, in step lengths")
    bottom.legend(frameon=False, fontsize=9, loc="upper right")

    figure.tight_layout()
    return figure
