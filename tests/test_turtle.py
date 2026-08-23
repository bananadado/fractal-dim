"""Turtle interpretation."""

import math

import numpy as np
import pytest

from fractaldim import library
from fractaldim.lsystem import LSystem
from fractaldim.turtle import Command, bounds, diameter, trace, trace_iter, walk


def test_right_angle_steps_are_exact():
    """No floating-point drift at 90 degrees, which the box counter relies on."""
    segments = trace(library.get("hilbert").system, 6)
    assert np.array_equal(segments, np.round(segments))


def test_forward_and_turns():
    system = LSystem("F+F", {}, angle=90.0)
    segments = trace(system, 0)
    assert segments.shape == (2, 2, 2)
    np.testing.assert_allclose(segments[0], [[0, 0], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(segments[1], [[1, 0], [1, 1]], atol=1e-12)


def test_lowercase_f_moves_without_drawing():
    system = LSystem("FfF", {}, angle=90.0)
    segments = trace(system, 0)
    assert len(segments) == 2
    np.testing.assert_allclose(segments[1], [[2, 0], [3, 0]], atol=1e-12)


def test_brackets_restore_position_and_heading():
    system = LSystem("F[+F]F", {}, angle=90.0)
    segments = trace(system, 0)
    assert len(segments) == 3
    # The branch starts at the tip of the first segment and turns left...
    np.testing.assert_allclose(segments[1], [[1, 0], [1, 1]], atol=1e-12)
    # ...and the trunk resumes from that tip, still heading along +x.
    np.testing.assert_allclose(segments[2], [[1, 0], [2, 0]], atol=1e-12)


def test_unmatched_close_bracket_is_an_error():
    with pytest.raises(ValueError, match=r"no matching"):
        trace(LSystem("F]", {}, angle=90.0), 0)


def test_reverse_needs_an_angle_dividing_half_a_turn():
    np.testing.assert_allclose(
        trace(LSystem("F|F", {}, angle=90.0), 0)[1], [[1, 0], [0, 0]], atol=1e-12
    )
    with pytest.raises(ValueError, match=r"180 degrees"):
        trace(LSystem("F|F", {}, angle=50.0), 0)


def test_command_table_overrides_let_other_symbols_draw():
    """The arrowhead's recursion symbols are the ones that draw."""
    fractal = library.get("arrowhead")
    assert len(trace(fractal.system, 3, commands=fractal.commands)) == 27
    # Without the override, A and B are ignored and nothing is drawn.
    assert len(trace(fractal.system, 3)) == 0


def test_start_heading_rotates_the_figure():
    system = LSystem("F", {}, angle=90.0)
    np.testing.assert_allclose(
        trace(system, 0, start_heading=90.0)[0], [[0, 0], [0, 1]], atol=1e-12
    )


def test_step_length_scales_the_figure():
    system = library.get("koch").system
    unit = trace(system, 3)
    scaled = trace(system, 3, step=0.25)
    np.testing.assert_allclose(scaled, unit * 0.25, atol=1e-12)


def test_streaming_matches_materialised_trace():
    fractal = library.get("gosper")
    segments = trace(fractal.system, 3, commands=fractal.commands)
    chunks = list(
        trace_iter(fractal.system, 3, commands=fractal.commands, chunk=16)
    )
    assert max(len(chunk) for chunk in chunks) <= 16
    streamed = np.concatenate(chunks, axis=0).reshape(-1, 2, 2)
    np.testing.assert_array_equal(streamed, segments)


def test_walk_consumes_any_iterable_of_symbols():
    segments = np.concatenate(list(walk(iter("F+F"), angle=90.0)), axis=0)
    assert segments.shape == (2, 4)


@pytest.mark.parametrize(
    "name,level,expected",
    [("koch", 6, 4), ("peano", 4, 9), ("hilbert", 6, 4), ("gosper", 4, 7)],
)
def test_segment_count_grows_at_the_expected_rate(name, level, expected):
    """Drawn segments multiply by lambda per rewriting step.

    Hilbert needs a deeper level than the rest: its count is 4^n - 1, so the
    ratio only settles once the subdominant eigenvalue has washed out.
    """
    fractal = library.get(name)
    counts = [
        len(trace(fractal.system, n, commands=fractal.commands))
        for n in (level, level + 1)
    ]
    assert counts[1] / counts[0] == pytest.approx(expected, rel=0.01)


#: Levels deep enough for the boundary correction to have decayed, while
#: keeping the segment count manageable. High-lambda systems need low levels.
_K_LEVELS = {
    "cantor": (8, 10), "koch": (6, 8), "minkowski": (3, 5), "arrowhead": (7, 9),
    "peano": (3, 5), "hilbert": (6, 8), "sierpinski_gasket": (7, 9),
    "gosper": (3, 5), "dragon": (8, 10), "levy_c": (8, 10),
    "koch_snowflake": (5, 7),
}


@pytest.mark.parametrize("name", sorted(_K_LEVELS))
def test_measured_scaling_factor_matches_the_catalogue(name):
    """k recovered from geometry alone, as sqrt(diam(gamma_{n+2}) / diam(gamma_n)).

    Taking the ratio two levels apart rather than one is deliberate: the dragon
    and Levy curves have a parity in their bounding box, so consecutive levels
    alternate either side of k. This is what step 2 builds on to reproduce the
    catalogue's k rather than take it on trust.
    """
    fractal = library.get(name)
    lo_level, hi_level = _K_LEVELS[name]
    kwargs = dict(commands=fractal.commands, start_heading=fractal.start_heading)
    lo = diameter(trace(fractal.system, lo_level, **kwargs))
    hi = diameter(trace(fractal.system, hi_level, **kwargs))
    assert math.sqrt(hi / lo) == pytest.approx(fractal.k, rel=0.03)


def test_bounds_and_diameter_of_an_empty_trace():
    empty = np.empty((0, 2, 2))
    assert bounds(empty) == (0.0, 0.0, 0.0, 0.0)
    assert diameter(empty) == 0.0


def test_hilbert_fills_a_square_of_side_two_to_the_n_minus_one():
    for n in range(1, 7):
        xmin, ymin, xmax, ymax = bounds(trace(library.get("hilbert").system, n))
        assert (xmax - xmin, ymax - ymin) == (2**n - 1, 2**n - 1)
