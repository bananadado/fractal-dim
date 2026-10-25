"""Why counting at ever smaller boxes makes the estimate worse, not better.

The box-counting dimension is a limit as ``eps -> 0``, which reads as an
instruction: count at the smallest boxes you can afford.  Followed literally on
a drawn L-system it gives the wrong answer, and gives it confidently.

The reason is that the limit is taken on the fractal and the counting is done
on a level-``n`` approximation to it.  Those two sets agree down to the step
length and not below: under the step length the figure is a finite union of
straight segments, and a straight segment is one-dimensional.  So the local
slope of the log-log plot leaves the dimension and falls towards 1, and every
fine scale added to the fit drags the fitted number down with it.

That is the left arm.  There is a right arm too, from the other end: once the
boxes are a sizeable fraction of the figure there are only a handful of them,
the count reflects the outline rather than the curve, and the few points left
are too short a lever to fit through.  Plotting the error of the fit against
where the fit was cut off therefore gives a **V**: bad at coarse scales, bad at
fine scales, least bad in a window a few step lengths wide.

The V is the point.  It says the estimator has an optimum rather than a limit,
that the optimum is set by the level drawn and not by the compute available,
and that the way to a better estimate is a deeper curve -- which moves the
whole V down and to the left -- rather than a smaller box on the same one.

There are two views of this and they answer different questions.  ``sweep``
here holds the top of the fit fixed and walks the bottom down, which is the
experiment someone reading the definition has in mind: *what happens to my
number if I count finer?*  The local slopes already carried by
:class:`~fractaldim.estimate.Estimate` answer the pointwise version: *what is
being measured at this scale?*  A least-squares slope is a weighted average of
the local slopes under it, which is why the arms of the V are the smoother of
the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boxcount import count_over_offsets
from .estimate import local_slopes, scaling_window
from .geometry import diameter

#: Octaves to count below the step length.  Each one doubles the number of
#: occupied cells on a curve that is locally a line, so the cost of the left
#: arm is linear in this and the default is a compromise, not a limit.
BELOW_STEP = 32

#: Shortest fit worth reporting.  Two points always fit a line exactly and say
#: nothing about whether the plot is straight.
MIN_POINTS = 3


@dataclass(frozen=True)
class CutoffSweep:
    """Fitted dimension as a function of the finest box let into the fit."""

    level: int
    step: float
    span: float
    eps: np.ndarray             # full ladder of box sizes, increasing
    counts: np.ndarray
    top: float                  # fixed upper end of every fit
    cutoffs: np.ndarray         # lower end of each fit, increasing
    dimensions: np.ndarray      # fitted slope for each cutoff
    used: np.ndarray            # points each fit was made from
    slope_centres: np.ndarray
    slopes: np.ndarray
    exact: float | None

    @property
    def errors(self) -> np.ndarray:
        """Signed error of each fit, or NaN throughout when nothing is known."""
        if self.exact is None:
            return np.full_like(self.dimensions, math.nan)
        return self.dimensions - self.exact

    @property
    def best(self) -> int:
        """Index of the cutoff with the smallest error: the floor of the V."""
        if self.exact is None:
            raise ValueError("no reference dimension to measure error against")
        return int(np.abs(self.errors).argmin())


def sweep(
    segments: np.ndarray,
    *,
    level: int,
    step: float = 1.0,
    exact: float | None = None,
    top: float | None = None,
    offsets: int = 4,
    below_step: int = BELOW_STEP,
    min_points: int = MIN_POINTS,
) -> CutoffSweep:
    """Fit the log-log plot from each box size up to ``top``, and report both.

    Every fit shares an upper end, so the only thing that varies between them
    is how far down the ladder they reach.  Any change in the fitted dimension
    is then attributable to the fine scales alone, which is what makes the
    resulting curve readable as a statement about ``eps -> 0``.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if len(segments) == 0:
        raise ValueError("nothing to count: the trace produced no segments")
    if below_step < 1:
        raise ValueError("below_step must be at least 1")

    span = diameter(segments)
    if top is None:
        top = scaling_window(step, span)[1]

    finest = step / below_step
    scales = max(2, int(math.ceil(math.log2(span / finest))) + 1)
    counted, _ = count_over_offsets(
        segments, finest=finest, scales=scales, offsets=offsets
    )
    eps, counts = counted.eps, counted.counts

    cutoffs, dimensions, used = [], [], []
    for index, low in enumerate(eps):
        inside = (eps >= low * (1 - 1e-9)) & (eps <= top * (1 + 1e-9))
        if inside.sum() < min_points:
            continue
        x = -np.log(eps[inside])
        y = np.log(counts[inside])
        cutoffs.append(low)
        dimensions.append(float(np.polyfit(x, y, 1)[0]))
        used.append(int(inside.sum()))

    if not cutoffs:
        raise ValueError(
            f"no fit of {min_points} points fits under eps <= {top:g}; "
            "draw a deeper level or raise below_step"
        )

    centres, slopes = local_slopes(eps, counts)
    return CutoffSweep(
        level=level, step=step, span=span, eps=eps, counts=counts, top=top,
        cutoffs=np.array(cutoffs), dimensions=np.array(dimensions),
        used=np.array(used), slope_centres=centres, slopes=slopes, exact=exact,
    )


def sweep_levels(fractal, levels, *, step: float = 1.0, **kwargs) -> list[CutoffSweep]:
    """Run :func:`sweep` at each level, tracing once per level.

    Deeper levels push the roll-off further below the step length *relative to
    the figure*, so the family of sweeps shows the V descending rather than a
    single curve that could be mistaken for noise.
    """
    from .turtle import trace

    results = []
    for level in sorted(levels):
        segments = trace(
            fractal.system, level, step=step,
            commands=fractal.commands, start_heading=fractal.start_heading,
        )
        results.append(sweep(
            segments, level=level, step=step,
            exact=fractal.dimension, **kwargs,
        ))
    return results
