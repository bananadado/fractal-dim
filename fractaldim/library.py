"""Catalogue of L-systems with known dimension.

The systems in the write-up's validation table appear here, together with a few
extras that widen the range of behaviour (bracketed branching, non-dyadic
scaling factors, curves whose scaling factor is irrational).

The ``lam``, ``k`` and ``dimension`` fields are *reference* values taken from
Moran's theorem, recorded so that step 2 has something to check itself against.
Nothing in the drawing path reads them: they are the answers, not the inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .lsystem import LSystem
from .turtle import Command

_DRAW_AB: Mapping[str, Command] = {"A": Command.DRAW, "B": Command.DRAW}


@dataclass(frozen=True)
class Fractal:
    """An L-system plus the geometric data needed to draw it."""

    system: LSystem
    start_heading: float = 0.0
    commands: Mapping[str, Command] = field(default_factory=dict)

    # Reference values -- reproduced, not used, by the estimator.
    lam: float | None = None          # spectral radius of the substitution matrix
    k: float | None = None            # geometric scaling factor per rewriting step
    dimension: float | None = None    # log(lam) / log(k)
    note: str = ""

    @property
    def name(self) -> str:
        return self.system.name

    @property
    def angle(self) -> float:
        return self.system.angle

    def __str__(self) -> str:
        dim = "unknown" if self.dimension is None else f"{self.dimension:.6f}"
        return f"{self.name}: delta={self.angle:g}, dim={dim}"


def _fractal(name, axiom, rules, angle, **kwargs) -> Fractal:
    return Fractal(LSystem(axiom, rules, angle, name), **kwargs)


LIBRARY: dict[str, Fractal] = {
    # -- the write-up's validation family ------------------------------------
    "cantor": _fractal(
        "cantor", "F", {"F": "FfF", "f": "fff"}, 90.0,
        lam=2, k=3, dimension=math.log(2) / math.log(3),
        note="Cantor dust; 'f' leaves the removed middle third as a gap. "
             "The gap must triple, not double: with 'f->ff' the gaps shrink "
             "relative to the segments and the figure spans 2.29^n, not 3^n. "
             "Note rho(M) = 3 here but the drawn symbols grow as 2^n -- the "
             "dominant eigenvalue lives in the undrawn 'f' block.",
    ),
    "koch": _fractal(
        "koch", "F", {"F": "F+F--F+F"}, 60.0,
        lam=4, k=3, dimension=math.log(4) / math.log(3),
        note="Koch curve; a single side of the snowflake.",
    ),
    "koch_snowflake": _fractal(
        "koch_snowflake", "F--F--F", {"F": "F+F--F+F"}, 60.0,
        lam=4, k=3, dimension=math.log(4) / math.log(3),
        note="Three Koch curves closed into a snowflake.",
    ),
    "minkowski": _fractal(
        "minkowski", "F", {"F": "F+F-F-FF+F+F-F"}, 90.0,
        lam=8, k=4, dimension=1.5,
        note="Minkowski sausage; eight steps of a quarter the length.",
    ),
    "arrowhead": _fractal(
        "arrowhead", "A", {"A": "B-A-B", "B": "A+B+A"}, 60.0,
        commands=_DRAW_AB,
        lam=3, k=2, dimension=math.log(3) / math.log(2),
        note="Sierpinski arrowhead; A and B both draw.",
    ),
    "peano": _fractal(
        "peano", "F", {"F": "F+F-F-F-F+F+F+F-F"}, 90.0,
        lam=9, k=3, dimension=2.0,
        note="Peano curve; space-filling, edge-rewriting.",
    ),
    "hilbert": _fractal(
        "hilbert", "A", {"A": "+BF-AFA-FB+", "B": "-AF+BFB+FA-", "F": "F"}, 90.0,
        lam=4, k=2, dimension=2.0,
        note="Hilbert curve; space-filling, node-rewriting, so the naive "
             "count of F symbols (three) is not the growth rate (four).",
    ),

    # -- extras --------------------------------------------------------------
    "sierpinski_gasket": _fractal(
        "sierpinski_gasket", "F-G-G", {"F": "F-G+F+G-F", "G": "GG"}, 120.0,
        lam=3, k=2, dimension=math.log(3) / math.log(2),
        note="Sierpinski gasket outline; same dimension as the arrowhead.",
    ),
    "dragon": _fractal(
        "dragon", "FX", {"X": "X+YF+", "Y": "-FX-Y"}, 90.0,
        lam=2, k=math.sqrt(2), dimension=2.0,
        note="Heighway dragon; irrational scaling factor sqrt(2).",
    ),
    "levy_c": _fractal(
        "levy_c", "F", {"F": "+F--F+"}, 45.0,
        lam=2, k=math.sqrt(2), dimension=2.0,
        note="Levy C curve; two half-steps at 45 degrees.",
    ),
    "gosper": _fractal(
        "gosper", "A", {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"}, 60.0,
        commands=_DRAW_AB,
        lam=7, k=math.sqrt(7), dimension=2.0,
        note="Gosper flowsnake; tiles the plane, scaling factor sqrt(7).",
    ),
    "plant": _fractal(
        "plant", "X", {"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"}, 25.0,
        start_heading=90.0,
        note="Branching plant (ABOP fig. 1.24f); exercises '[' and ']'. "
             "Not self-similar in the Moran sense, so no exact dimension.",
    ),
}

#: Systems appearing in the write-up's validation table, in table order.
VALIDATION_FAMILY = (
    "cantor", "koch", "minkowski", "arrowhead", "peano", "hilbert",
)


def get(name: str) -> Fractal:
    """Look up a catalogued fractal by name."""
    try:
        return LIBRARY[name]
    except KeyError:
        options = ", ".join(sorted(LIBRARY))
        raise KeyError(f"unknown fractal {name!r}; available: {options}") from None


def names() -> list[str]:
    """Every catalogued name, validation family first."""
    rest = sorted(set(LIBRARY) - set(VALIDATION_FAMILY))
    return list(VALIDATION_FAMILY) + rest
