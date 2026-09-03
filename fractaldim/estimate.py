"""Turning box counts into a dimension, and knowing which counts to trust.

The box-counting dimension is the limit of ``log N(eps) / log(1/eps)`` as
``eps -> 0``, but the level-n figure of an L-system is a finite union of line
segments, and a finite union of line segments has dimension one.  Take eps far
enough below the step length and the measured slope duly falls to 1: what is
being measured there is the polyline, not the fractal it approximates.

So the log-log plot is not a straight line, and fitting all of it is wrong.  It
has three regions:

* ``eps`` below the step length -- the segments look one-dimensional, slope 1;
* ``eps`` above roughly a quarter of the figure -- too few boxes to say
  anything, and the count is dominated by the shape of the outline;
* between the two, the scaling window, where the slope estimates the dimension.

The window is where the answer lives, so this module reports the local slope at
every scale rather than only the fitted number.  A plateau in that curve is the
evidence that a dimension was measured at all; a fit through data with no
plateau is a number with nothing behind it.

One bias survives any choice of window, and it is worth naming rather than
tuning away: the level-n figure is an approximation to the fractal, so at fine
box sizes it occupies fewer cells than the limit set does, and the fitted slope
comes out **low**.  Measured against the validation family the error is
consistently negative and shrinks with level -- the Koch curve reads 1.023 at
level 4, 1.224 at level 6 and 1.238 at level 8, against an exact 1.2619.  So an
estimate is only as good as the level it was taken at, and quoting one without
the level is quoting half the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boxcount import BoxCount, count_over_offsets
from .geometry import diameter

#: Window bounds, chosen by sweeping the validation family rather than by
#: taste.  The lower one is the step length, below which the segments look
#: one-dimensional.  The upper one asks for at least this many boxes across the
#: figure: expressing it as a fraction of the diameter lets in scales with a
#: dozen boxes total, where the count says more about the outline than the
#: curve, and those points visibly drag the fit.
LOWER_MULTIPLE = 1.0
BOXES_ACROSS = 32

#: How far below the step length to count, so the roll-off to slope 1 is
#: visible rather than merely asserted.
_BELOW_STEP = 8


def local_slopes(eps: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slope between consecutive scales, and the scale it belongs to.

    Returned against the geometric mean of the two box sizes, which is where
    the finite difference is actually centred on a log axis.
    """
    log_eps = np.log(eps)
    log_counts = np.log(counts)
    slopes = -np.diff(log_counts) / np.diff(log_eps)
    centres = np.exp(0.5 * (log_eps[:-1] + log_eps[1:]))
    return centres, slopes


@dataclass(frozen=True)
class Estimate:
    """A dimension, the data behind it, and the window it was fitted over."""

    eps: np.ndarray
    counts: np.ndarray
    window: tuple[float, float]
    used: np.ndarray            # boolean mask over eps
    dimension: float
    stderr: float
    intercept: float
    offset_spread: np.ndarray   # counts at every grid placement tried
    slope_centres: np.ndarray
    slopes: np.ndarray

    @property
    def points(self) -> int:
        return int(self.used.sum())

    @property
    def plateau(self) -> tuple[float, float]:
        """Spread of the local slope inside the window: the honest error bar.

        The fit's standard error assumes independent noise about a straight
        line, which is not what the deviations here are -- they are systematic
        curvature from the two roll-offs.  The range of local slopes says more
        about how straight the plot really is.
        """
        inside = self.used[:-1] & self.used[1:]
        if not inside.any():
            return (math.nan, math.nan)
        chosen = self.slopes[inside]
        return (float(chosen.min()), float(chosen.max()))


def scaling_window(
    step: float,
    span: float,
    lower_multiple: float = LOWER_MULTIPLE,
    boxes_across: int = BOXES_ACROSS,
) -> tuple[float, float]:
    """Box sizes over which a slope is worth fitting.

    Needs about six octaves between the step length and the figure's diameter
    before a window exists at all, which is why shallow levels cannot be
    box-counted however carefully the fit is done.
    """
    return (lower_multiple * step, span / boxes_across)


def estimate(
    segments: np.ndarray,
    *,
    step: float = 1.0,
    offsets: int = 4,
    window: tuple[float, float] | None = None,
    below_step: int = _BELOW_STEP,
) -> Estimate:
    """Box-count ``segments`` across scales and fit the scaling window.

    Counting starts below the step length on purpose: the roll-off towards
    slope 1 is the check that the counter is measuring what it should, and it
    cannot be seen from inside the window alone.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if len(segments) == 0:
        raise ValueError("nothing to count: the trace produced no segments")

    span = diameter(segments)
    finest = step / below_step
    scales = max(2, int(math.ceil(math.log2(span / finest))) + 1)

    counted, spread = count_over_offsets(
        segments, finest=finest, scales=scales, offsets=offsets
    )
    eps, counts = counted.eps, counted.counts

    low, high = scaling_window(step, span) if window is None else window
    used = (eps >= low * (1 - 1e-9)) & (eps <= high * (1 + 1e-9))
    if used.sum() < 2:
        # Too shallow a curve for a window; fall back to everything above the
        # step length so the caller still gets a number, with points to judge.
        used = eps >= step * (1 - 1e-9)
    if used.sum() < 2:
        used = np.ones_like(eps, dtype=bool)

    x = -np.log(eps[used])
    y = np.log(counts[used])
    slope, intercept = np.polyfit(x, y, 1)

    residual = y - (slope * x + intercept)
    degrees = max(len(x) - 2, 1)
    variance = float((residual ** 2).sum() / degrees)
    spread_x = float(((x - x.mean()) ** 2).sum())
    stderr = math.sqrt(variance / spread_x) if spread_x > 0 else math.nan

    centres, slopes = local_slopes(eps, counts)
    return Estimate(
        eps=eps, counts=counts, window=(low, high), used=used,
        dimension=float(slope), stderr=stderr, intercept=float(intercept),
        offset_spread=spread, slope_centres=centres, slopes=slopes,
    )
