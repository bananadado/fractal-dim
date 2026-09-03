"""Box counting on line segments, without rasterising.

The usual way to box-count a curve is to draw it into a bitmap and count
non-empty pixels.  That makes the image resolution a hard floor on the box size
and quietly turns a geometry question into a rendering question.  Here the
curve is already an exact set of segments, so the cells it meets are computed
exactly instead.

For one segment, the cells it passes through are found by parametrising it as
``p(t) = p0 + t(p1 - p0)`` and collecting every ``t`` at which it crosses a
gridline.  Between consecutive crossings the segment lies wholly inside one
cell, so evaluating at the midpoint of each interval and taking the floor names
that cell -- exactly, with no sampling and no chance of stepping over a corner.

Counting at many scales does not need many traversals.  The cells are found
once on the finest dyadic grid; a cell at twice the size is reached by shifting
its indices right by one bit, since the grids are nested.  Coarsening is then a
shift and a deduplication rather than a fresh pass over the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Segments per chunk.  Each one expands to several candidate cells, so this
#: bounds peak memory independently of how deep the curve is.
_CHUNK = 200_000


def _unique_cells(cells: np.ndarray) -> np.ndarray:
    """Deduplicate integer cell indices of shape ``(n, 2)``."""
    if len(cells) == 0:
        return cells.reshape(0, 2)
    lo = cells.min(axis=0)
    span = cells.max(axis=0) - lo + 1
    # Pack the pair into one integer when it cannot overflow, which makes the
    # deduplication a 1-D sort rather than a structured one.
    if span[0] * span[1] < (1 << 62):
        keys = (cells[:, 0] - lo[0]) * span[1] + (cells[:, 1] - lo[1])
        keys = np.unique(keys)
        return np.stack([keys // span[1] + lo[0], keys % span[1] + lo[1]], axis=1)
    return np.unique(cells, axis=0)


def _crossings(a0: np.ndarray, a1: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Parameters at which segments cross gridlines along one axis.

    Padded to a rectangular array with ``1.0``, which is already an endpoint of
    the parameter range and so adds no cell that is not covered anyway.
    """
    lo = np.floor(np.minimum(a0, a1))
    hi = np.floor(np.maximum(a0, a1))
    width = int((hi - lo).max()) if len(lo) else 0
    if width <= 0:
        return np.empty((len(a0), 0))

    lines = lo[:, None] + 1.0 + np.arange(width)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (lines - a0[:, None]) / delta[:, None]
    inside = (lines <= hi[:, None]) & np.isfinite(t)
    return np.clip(np.where(inside, t, 1.0), 0.0, 1.0)


def occupied_cells(
    segments: np.ndarray,
    eps: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Integer indices of every grid cell of side ``eps`` the curve meets.

    Cells are half-open, ``[i*eps, (i+1)*eps) x [j*eps, (j+1)*eps)``, offset by
    ``origin``.  The result is sorted and free of duplicates.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    if len(segments) == 0:
        return np.empty((0, 2), dtype=np.int64)
    if eps <= 0:
        raise ValueError(f"box size must be positive, got {eps}")

    shift = np.asarray(origin, dtype=np.float64)
    found = []
    for start in range(0, len(segments), _CHUNK):
        block = segments[start:start + _CHUNK]
        p0 = (block[:, 0] - shift) / eps
        p1 = (block[:, 1] - shift) / eps
        delta = p1 - p0

        ends = np.zeros((len(block), 2))
        parameters = [ends, np.ones((len(block), 2))]
        for axis in (0, 1):
            crossed = _crossings(p0[:, axis], p1[:, axis], delta[:, axis])
            if crossed.size:
                parameters.append(crossed)

        times = np.concatenate(parameters, axis=1)
        times.sort(axis=1)
        midpoints = 0.5 * (times[:, :-1] + times[:, 1:])

        points = p0[:, None, :] + midpoints[:, :, None] * delta[:, None, :]
        found.append(_unique_cells(
            np.floor(points).astype(np.int64).reshape(-1, 2)
        ))

    return _unique_cells(np.concatenate(found, axis=0))


@dataclass(frozen=True)
class BoxCount:
    """Occupied-cell counts across a dyadic ladder of box sizes."""

    eps: np.ndarray        # box side, increasing
    counts: np.ndarray     # cells occupied at that box side
    origin: tuple[float, float]

    def __len__(self) -> int:
        return len(self.eps)


def count(
    segments: np.ndarray,
    *,
    finest: float,
    scales: int,
    origin: tuple[float, float] = (0.0, 0.0),
) -> BoxCount:
    """Count occupied cells at ``finest``, then at each doubling above it.

    One exact traversal of the geometry, then ``scales - 1`` bit shifts.
    """
    cells = occupied_cells(segments, finest, origin)
    sizes, counts = [], []
    for level in range(scales):
        coarse = cells >> level if level else cells
        # Arithmetic shift rounds toward minus infinity, which is the floor the
        # coarser grid wants, so negative coordinates need no special case.
        counts.append(len(_unique_cells(coarse)))
        sizes.append(finest * (1 << level))
    return BoxCount(np.array(sizes), np.array(counts, dtype=np.int64), origin)


def count_over_offsets(
    segments: np.ndarray,
    *,
    finest: float,
    scales: int,
    offsets: int = 1,
) -> tuple[BoxCount, np.ndarray]:
    """Repeat the count with the grid shifted, and report the spread.

    Where the grid falls is arbitrary but the count is not, so a single
    placement carries a systematic error that no amount of averaging over
    scales removes.  Returns the counts at the tightest placement together with
    the full ``(offsets, scales)`` table behind them.
    """
    if offsets < 1:
        raise ValueError("offsets must be at least 1")

    table = []
    results = []
    for index in range(offsets):
        fraction = index / offsets
        shift = (fraction * finest, fraction * finest)
        result = count(segments, finest=finest, scales=scales, origin=shift)
        results.append(result)
        table.append(result.counts)

    stacked = np.array(table)
    best = int(stacked.sum(axis=1).argmin())
    return results[best], stacked
