"""Exact box counting on segments."""

import numpy as np
import pytest

from fractaldim import library
from fractaldim.boxcount import count, count_over_offsets, occupied_cells
from fractaldim.turtle import trace


def _cells(x0, y0, x1, y1, eps=1.0, origin=(0.0, 0.0)):
    segment = np.array([[[x0, y0], [x1, y1]]], dtype=float)
    return {tuple(int(v) for v in cell) for cell in occupied_cells(segment, eps, origin)}


def test_segment_inside_one_cell():
    assert _cells(0.1, 0.1, 0.9, 0.9) == {(0, 0)}


def test_horizontal_segment_crossing_gridlines():
    assert _cells(0.5, 0.5, 2.5, 0.5) == {(0, 0), (1, 0), (2, 0)}


def test_diagonal_through_corners_does_not_pick_up_neighbours():
    """Passing exactly through a lattice corner touches one cell, not four.

    Half-open cells make this unambiguous: the point (1, 1) belongs to cell
    (1, 1) alone, so a rasterising counter's corner ambiguity never arises.
    """
    assert _cells(0.5, 0.5, 2.5, 2.5) == {(0, 0), (1, 1), (2, 2)}


def test_shallow_diagonal_visits_every_cell_it_passes_through():
    """The supercover case: a shallow slope must not skip cells.

    Crossing x=1 at y=0.833 keeps it in (1, 0); it enters (1, 1) at x=1.5.
    The final cell is (3, 1) because the segment ends exactly on the gridline
    x=3, and a half-open cell owns its lower edge.
    """
    assert _cells(0.0, 0.5, 3.0, 1.5) == {(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)}


def test_an_endpoint_on_a_gridline_belongs_to_the_cell_above_it():
    """Half-open cells make the convention explicit rather than accidental,
    which is what keeps the count reproducible for lattice curves where every
    endpoint sits on a gridline."""
    assert _cells(0.5, 0.5, 1.0, 0.5) == {(0, 0), (1, 0)}


def test_negative_coordinates():
    assert _cells(-0.5, -0.5, 0.5, 0.5) == {(-1, -1), (0, 0)}


def test_origin_shifts_the_grid():
    assert _cells(0.5, 0.5, 0.5, 0.5, origin=(0.25, 0.25)) == {(0, 0)}
    assert _cells(0.1, 0.1, 0.1, 0.1, origin=(0.25, 0.25)) == {(-1, -1)}


def test_box_size_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        occupied_cells(np.zeros((1, 2, 2)), 0.0)


def test_empty_input_counts_nothing():
    assert len(occupied_cells(np.empty((0, 2, 2)), 1.0)) == 0


def test_hilbert_fills_its_grid_exactly():
    """Level n visits every cell of a 2^n x 2^n grid: N(eps) = 4^n exactly.

    The space-filling case is the one where the right answer is known cell by
    cell, not just asymptotically, so it pins the counter down completely.
    """
    for level in range(1, 7):
        segments = trace(library.get("hilbert").system, level)
        cells = occupied_cells(segments, 1.0)
        assert len(cells) == 4 ** level


def test_coarsening_agrees_with_counting_from_scratch():
    """Shifting indices right must give the same answer as recounting.

    This is what lets one traversal serve every scale, so it has to be exact
    rather than nearly right.
    """
    fractal = library.get("koch")
    segments = trace(fractal.system, 5)
    laddered = count(segments, finest=0.5, scales=6)
    for eps, expected in zip(laddered.eps, laddered.counts):
        assert len(occupied_cells(segments, eps)) == expected


def test_counts_fall_as_boxes_grow():
    segments = trace(library.get("koch").system, 5)
    result = count(segments, finest=0.25, scales=8)
    assert np.all(np.diff(result.counts) <= 0)
    assert len(result) == 8


def test_grid_placement_changes_the_count():
    """Offsetting the grid is a real systematic effect, not noise."""
    segments = trace(library.get("koch").system, 5)
    _, table = count_over_offsets(segments, finest=1.0, scales=6, offsets=4)
    assert table.shape == (4, 6)
    assert table.sum(axis=1).min() < table.sum(axis=1).max()


def test_offsets_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        count_over_offsets(np.zeros((1, 2, 2)), finest=1.0, scales=2, offsets=0)
