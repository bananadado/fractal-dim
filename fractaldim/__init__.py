"""Fractal dimension: L-systems, exact box counting, and estimation."""

from .lsystem import LSystem
from .turtle import Command, bounds, diameter, trace, trace_iter, walk
from .library import Fractal, LIBRARY, VALIDATION_FAMILY, get, names

__version__ = "0.1.0"

__all__ = [
    "LSystem",
    "Command",
    "Fractal",
    "LIBRARY",
    "VALIDATION_FAMILY",
    "bounds",
    "diameter",
    "get",
    "names",
    "trace",
    "trace_iter",
    "walk",
]
