"""Turtle interpretation of an L-system word.

The turtle carries a state ``(x, y, heading)`` and reads a word left to right.
Which symbol means what is a separate layer from the grammar -- a symbol table
mapping characters to :class:`Command` -- so that recursion-carrying symbols
like Hilbert's ``A`` and ``B`` can be invisible to the turtle, while a system
like the Sierpinski arrowhead can have those very symbols draw.

Headings are held as an integer multiple of the turning angle rather than as a
float.  When ``360 / angle`` is a whole number there are only finitely many
directions, so they are computed once and reused; for a right angle this makes
the step vectors exactly ``(+-1, 0)`` and ``(0, +-1)`` with no floating-point
drift, which later lets the box counter work in integer arithmetic.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, Iterator, Mapping

import numpy as np


class Command(Enum):
    """What the turtle does when it reads a symbol."""

    DRAW = "draw"      # forward one step, laying down a segment
    MOVE = "move"      # forward one step without drawing
    LEFT = "left"      # turn left by delta
    RIGHT = "right"    # turn right by delta
    REVERSE = "rev"    # turn by 180 degrees
    PUSH = "push"      # save position and heading
    POP = "pop"        # restore the most recently saved state
    NOP = "nop"        # no geometric meaning


DEFAULT_COMMANDS: Mapping[str, Command] = {
    "F": Command.DRAW,
    "G": Command.DRAW,
    "f": Command.MOVE,
    "g": Command.MOVE,
    "+": Command.LEFT,
    "-": Command.RIGHT,
    "|": Command.REVERSE,
    "[": Command.PUSH,
    "]": Command.POP,
}

_SNAP = 1e-12


def _snap(value: float) -> float:
    """Round values that are numerically indistinguishable from 0 or +-1."""
    for exact in (0.0, 1.0, -1.0):
        if abs(value - exact) < _SNAP:
            return exact
    return value


class _Directions:
    """Unit step vectors indexed by an integer number of turns."""

    def __init__(self, angle: float, start_heading: float) -> None:
        self.angle = angle
        self.start_heading = start_heading
        self.period: int | None = None
        if angle:
            turns = 360.0 / angle
            if abs(turns - round(turns)) < 1e-9 and 0 < round(turns) < 100_000:
                self.period = int(round(turns))
        self._cache: dict[int, tuple[float, float]] = {}

    def __getitem__(self, turn: int) -> tuple[float, float]:
        if self.period is not None:
            turn %= self.period
        vector = self._cache.get(turn)
        if vector is None:
            radians = math.radians(self.start_heading + turn * self.angle)
            vector = (_snap(math.cos(radians)), _snap(math.sin(radians)))
            self._cache[turn] = vector
        return vector


def walk(
    symbols: Iterable[str],
    *,
    angle: float = 90.0,
    step: float = 1.0,
    start: tuple[float, float] = (0.0, 0.0),
    start_heading: float = 0.0,
    commands: Mapping[str, Command] | None = None,
    chunk: int = 1 << 16,
) -> Iterator[np.ndarray]:
    """Trace ``symbols``, yielding drawn segments in chunks.

    Each yielded array has shape ``(m, 4)`` holding ``(x0, y0, x1, y1)`` rows.
    Chunking keeps memory bounded, so this consumes a streamed word from
    :meth:`~fractaldim.lsystem.LSystem.expand_iter` at levels where the whole
    curve would not fit in memory.
    """
    table = dict(DEFAULT_COMMANDS)
    if commands:
        table.update(commands)
    directions = _Directions(angle, start_heading)

    x, y = start
    turn = 0
    stack: list[tuple[float, float, int]] = []
    buffer: list[tuple[float, float, float, float]] = []

    for symbol in symbols:
        command = table.get(symbol, Command.NOP)
        if command is Command.DRAW or command is Command.MOVE:
            dx, dy = directions[turn]
            nx, ny = x + step * dx, y + step * dy
            if command is Command.DRAW:
                buffer.append((x, y, nx, ny))
                if len(buffer) >= chunk:
                    yield np.array(buffer, dtype=np.float64)
                    buffer = []
            x, y = nx, ny
        elif command is Command.LEFT:
            turn += 1
        elif command is Command.RIGHT:
            turn -= 1
        elif command is Command.REVERSE:
            if directions.period is None or directions.period % 2:
                raise ValueError(
                    f"'|' needs a turning angle dividing 180 degrees, got {angle:g}"
                )
            turn += directions.period // 2
        elif command is Command.PUSH:
            stack.append((x, y, turn))
        elif command is Command.POP:
            if not stack:
                raise ValueError("']' with no matching '['")
            x, y, turn = stack.pop()

    if buffer:
        yield np.array(buffer, dtype=np.float64)


def trace(system, n: int, *, step: float = 1.0, **kwargs) -> np.ndarray:
    """Draw level ``n`` of ``system`` and return every segment.

    Returns an array of shape ``(m, 2, 2)``: ``segments[i, 0]`` is the start
    point of segment ``i`` and ``segments[i, 1]`` its end point.  Materialises
    the whole curve; use :func:`trace_iter` for deep levels.
    """
    chunks = list(trace_iter(system, n, step=step, **kwargs))
    if not chunks:
        return np.empty((0, 2, 2), dtype=np.float64)
    flat = np.concatenate(chunks, axis=0)
    return flat.reshape(-1, 2, 2)


def trace_iter(
    system,
    n: int,
    *,
    step: float = 1.0,
    angle: float | None = None,
    commands: Mapping[str, Command] | None = None,
    **kwargs,
) -> Iterator[np.ndarray]:
    """Stream the segments of level ``n`` of ``system`` as ``(m, 4)`` chunks."""
    return walk(
        system.expand_iter(n),
        angle=system.angle if angle is None else angle,
        step=step,
        commands=commands,
        **kwargs,
    )


def bounds(segments: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box ``(xmin, ymin, xmax, ymax)`` of ``segments``."""
    if segments.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    points = segments.reshape(-1, 2)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def diameter(segments: np.ndarray) -> float:
    """Diagonal of the bounding box.

    Used as a cheap, monotone stand-in for the true diameter.  Step 2 takes
    ratios of this across successive levels to recover the geometric scaling
    factor ``k``, where only the growth rate matters, not the constant.
    """
    xmin, ymin, xmax, ymax = bounds(segments)
    return math.hypot(xmax - xmin, ymax - ymin)
