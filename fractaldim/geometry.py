"""Point-set geometry: convex hull and true diameter.

The bounding box is a cheap stand-in for a figure's size, but it is not
rotation invariant, and that matters when measuring how a self-similar figure
grows.  The Heighway dragon turns 45 degrees per level, so its bounding box
oscillates while the figure itself scales cleanly: successive box ratios
alternate around sqrt(2) instead of converging to it.  The diameter -- the
greatest distance between two points of the figure -- has no such artefact,
because it does not care which way the figure is pointing.
"""

from __future__ import annotations

import numpy as np


def _column_extremes(points: np.ndarray) -> np.ndarray:
    """Discard points that cannot be hull vertices, keeping the order sorted.

    For a given x, any point strictly between the lowest and highest lies on
    the segment joining them and so is interior to the hull.  Dropping those
    first turns a million-point lattice curve into a few thousand candidates,
    which is what keeps the hull affordable at deep levels.
    """
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]
    xs = ordered[:, 0]
    first = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    last = np.r_[first[1:] - 1, len(ordered) - 1]
    return ordered[np.union1d(first, last)]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Vertices of the convex hull of ``points``, counter-clockwise.

    Andrew's monotone chain.  Collinear points are dropped, so the result is
    the vertex set proper rather than every boundary point.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) <= 2:
        return points.copy()

    candidates = _column_extremes(points)
    if len(candidates) <= 2:
        return candidates

    def chain(sequence):
        out: list[np.ndarray] = []
        for point in sequence:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
                if cross > 0.0:
                    break
                out.pop()
            out.append(point)
        return out

    lower = chain(candidates)
    upper = chain(candidates[::-1])
    # Each chain repeats the other's endpoint, so drop the last of both.
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def diameter(points: np.ndarray) -> float:
    """Greatest distance between any two of ``points``.

    Only hull vertices can realise it, so the search runs over those.  Accepts
    either a point array or a ``(m, 2, 2)`` segment array.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    hull = convex_hull(points)
    if len(hull) < 2:
        return 0.0

    # Chunked so the pairwise distances of a large hull never materialise at
    # once; hulls here are small, but coastlines later will not be.
    best = 0.0
    for start in range(0, len(hull), 512):
        block = hull[start:start + 512]
        delta = block[:, None, :] - hull[None, :, :]
        best = max(best, float(np.sqrt((delta ** 2).sum(-1)).max()))
    return best
