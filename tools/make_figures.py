#!/usr/bin/env python3
"""Regenerate the figures used by the write-up and the README.

Every image under ``writeup/images`` that is not a borrowed one, and every
image under ``images/``, is built here, so a change to the estimator or to the
palette is one command away from reaching both documents.

    python3 tools/make_figures.py                 # everything
    python3 tools/make_figures.py grids rolloff   # only these

Formats follow the destination: the write-up takes PDF, so the plots stay
sharp under LaTeX, except where a figure carries tens of thousands of segments
and a vector file would be enormous.  The README takes PNG and GIF.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fractaldim import library
from fractaldim.estimate import estimate
from fractaldim.plotting import (
    animate_segments, plot_boxcount, plot_grid_series, plot_hull, plot_rolloff,
    plot_strip,
)
from fractaldim.rolloff import sweep_levels
from fractaldim.turtle import trace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WRITEUP = os.path.join(ROOT, "writeup", "images")
README = os.path.join(ROOT, "images")

#: Grid placement is not swept here: the figures all show the unshifted grid,
#: which is the reproducible one, and the extra placements would only be
#: averaged away.
OFFSETS = 1


def _traced(name: str, level: int) -> np.ndarray:
    fractal = library.get(name)
    return trace(fractal.system, level, commands=fractal.commands,
                 start_heading=fractal.start_heading)


def _save(figure, *paths: str, dpi: int = 300) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  wrote {os.path.relpath(path, ROOT)} "
              f"({os.path.getsize(path) / 1e6:.2f} MB)")
    plt.close(figure)


# ── write-up ────────────────────────────────────────────────────────────────

def boxcount() -> None:
    """9.5, the three regions: where the slope is a dimension and where it is not."""
    segments = _traced("koch", 7)
    result = estimate(segments, offsets=OFFSETS)
    figure = plot_boxcount(result, reference=library.get("koch").dimension,
                           size=7.0, title=None)
    _save(figure, os.path.join(WRITEUP, "koch-boxcount.pdf"))


def grids() -> None:
    """9.4, exact occupancy: what N(eps) is actually counting.

    Level 4 rather than the level the table quotes, and box sizes chosen by
    hand: at level 7 the cells at the fine end of the window are a pixel wide
    and the picture shows nothing that the log-log plot does not.
    """
    segments = _traced("koch", 4)
    figure = plot_grid_series(segments, [1.0, 2.0, 4.0, 8.0], size=7.0,
                              title=None)
    _save(figure, os.path.join(WRITEUP, "koch-grids.pdf"),
          os.path.join(README, "koch-grids.png"))


def hull() -> None:
    """9.2, the convex hull: why the bounding box is the wrong size to measure.

    Two consecutive levels, because the dragon turns 45 degrees per level: the
    box stands square to the axes at one level and sits at an angle at the
    next, which is what makes Table 2's box column alternate.  The hull and the
    chord it realises do not care.
    """
    from fractaldim.geometry import diameter

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    figure.patch.set_facecolor("white")
    for level, axis in zip((10, 11), axes):
        segments = _traced("dragon", level)
        points = segments.reshape(-1, 2)
        box = float(np.hypot(*(points.max(axis=0) - points.min(axis=0))))
        plot_hull(segments, ax=axis,
                  title=f"level {level}:  box diagonal {box:.1f},  "
                        f"diameter {diameter(segments):.1f}")
    figure.tight_layout()
    _save(figure, os.path.join(WRITEUP, "dragon-hull.pdf"))


def _cantor_stack(levels: int, spacing: float = 0.16) -> np.ndarray:
    """The Cantor construction: each level normalised to unit width, stacked.

    One traced level on its own is a row of dots with nothing to give it scale
    -- at level 5 it is 32 segments across a span of 243 -- whereas the stack
    is the picture the set is usually recognised by.
    """
    rows = []
    for level in range(levels + 1):
        segments = _traced("cantor", level) / 3.0 ** level
        segments[:, :, 1] -= spacing * level
        rows.append(segments)
    return np.vstack(rows)


#: The appendix gallery, split the way the library is: the six systems of the
#: validation table first, then the extras.  Levels are chosen to show each
#: system's character and still trace and render in a second or two.
GALLERY_FAMILY = (
    ("Cantor, levels 0-5", "cantor", 5),
    ("Koch, level 6", "koch", 6),
    ("Minkowski, 5", "minkowski", 5),
    ("Arrowhead, 9", "arrowhead", 9),
    ("Peano, 5", "peano", 5),
    ("Hilbert, 7", "hilbert", 7),
)

GALLERY_EXTRAS = (
    ("Koch snowflake, 5", "koch_snowflake", 5),
    ("Sierpinski gasket, 7", "sierpinski_gasket", 7),
    ("Heighway dragon, 15", "dragon", 15),
    ("Levy C, 15", "levy_c", 15),
    ("Gosper flowsnake, 5", "gosper", 5),
    ("Plant, 7", "plant", 7),
)


def gallery() -> None:
    """The appendix: the whole catalogue at depth.

    PNG rather than PDF: between them these panels carry a few hundred thousand
    segments, which no reader needs as vector art.
    """
    for name, panels in (("family", GALLERY_FAMILY), ("extras", GALLERY_EXTRAS)):
        drawn = [(label,
                  _cantor_stack(level) if system == "cantor"
                  else _traced(system, level))
                 for label, system, level in panels]
        # Capped at square: one lanky panel -- the plant, the snowflake --
        # would otherwise set the height of its row, and with it the height of
        # a figure that has to share its page with the other half of the
        # catalogue.
        figure = plot_strip(drawn, columns=3, size=7.5, linewidth_scale=1.3,
                            max_aspect=1.0)
        _save(figure, os.path.join(WRITEUP, f"gallery-{name}.png"), dpi=200)


def rolloff() -> None:
    """9.6, the V: counting finer makes the estimate worse."""
    for name in ("koch", "hilbert"):
        sweeps = sweep_levels(library.get(name), [5, 6, 7], offsets=OFFSETS)
        figure = plot_rolloff(sweeps, size=7.0, title=None)
        paths = [os.path.join(WRITEUP, f"{name}-rolloff.pdf")]
        if name == "koch":
            paths.append(os.path.join(README, "koch-rolloff.png"))
        _save(figure, *paths)


# ── README ──────────────────────────────────────────────────────────────────

#: The strip at the top: four systems that look nothing alike, which is the
#: point of driving all of them from one tracer.
HERO = (
    ("dragon", "dragon", 14),
    ("gosper", "gosper", 5),
    ("plant", "plant", 7),
    ("hilbert", "hilbert", 6),
)


def hero() -> None:
    """The README's opening strip."""
    panels = [(None, _traced(name, level)) for _, name, level in HERO]
    figure = plot_strip(panels, size=12.0, linewidth_scale=1.4)
    _save(figure, os.path.join(README, "hero.png"), dpi=120)


def gif() -> None:
    """The turtle drawing a Hilbert curve, for the README."""
    path = os.path.join(README, "hilbert.gif")
    os.makedirs(README, exist_ok=True)
    animate_segments(_traced("hilbert", 4), path, frames=120, fps=20, hold=20,
                     dpi=120, size=4.0, title=None)
    print(f"  wrote {os.path.relpath(path, ROOT)} "
          f"({os.path.getsize(path) / 1e6:.2f} MB)")


FIGURES = {
    "boxcount": boxcount,
    "grids": grids,
    "hull": hull,
    "gallery": gallery,
    "rolloff": rolloff,
    "hero": hero,
    "gif": gif,
}


def main(argv: list[str]) -> int:
    chosen = argv or list(FIGURES)
    unknown = [name for name in chosen if name not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}\n"
              f"available: {', '.join(FIGURES)}", file=sys.stderr)
        return 1
    for name in chosen:
        print(f"{name}:")
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
