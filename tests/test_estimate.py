"""Fitting a dimension, and knowing which scales to fit."""

import numpy as np
import pytest

from fractaldim import library
from fractaldim.estimate import estimate, local_slopes, scaling_window
from fractaldim.geometry import diameter
from fractaldim.turtle import trace


def _traced(name, level):
    fractal = library.get(name)
    return fractal, trace(fractal.system, level, commands=fractal.commands,
                          start_heading=fractal.start_heading)


def test_local_slopes_recover_a_pure_power_law():
    eps = 2.0 ** -np.arange(10)
    counts = eps ** -1.5
    centres, slopes = local_slopes(eps, counts)
    np.testing.assert_allclose(slopes, 1.5)
    assert len(centres) == len(eps) - 1


def test_window_needs_headroom_between_step_and_span():
    low, high = scaling_window(step=1.0, span=1024.0)
    assert low == 1.0 and high == 32.0
    # A shallow figure has no window at all, which is the honest answer.
    low, high = scaling_window(step=1.0, span=16.0)
    assert high < low


def test_hilbert_is_exact_because_it_tiles_the_grid():
    """Every dyadic scale sees a perfectly full square, so there is no
    roll-off inside the window and no residual to fit."""
    _, segments = _traced("hilbert", 8)
    result = estimate(segments)
    assert result.dimension == pytest.approx(2.0, abs=1e-9)
    assert result.plateau == pytest.approx((2.0, 2.0), abs=1e-9)


@pytest.mark.parametrize("name,level,tolerance", [
    ("koch", 8, 0.03), ("minkowski", 6, 0.04), ("cantor", 12, 0.02),
    ("peano", 5, 0.03), ("sierpinski_gasket", 10, 0.02),
    ("arrowhead", 11, 0.06), ("gosper", 5, 0.06),
])
def test_box_counting_recovers_the_known_dimension(name, level, tolerance):
    """An independent measurement: nothing here knows lambda or k exist."""
    fractal, segments = _traced(name, level)
    result = estimate(segments)
    assert result.dimension == pytest.approx(fractal.dimension, abs=tolerance)


@pytest.mark.parametrize("name,level", [("koch", 8), ("arrowhead", 11),
                                        ("minkowski", 6), ("cantor", 12)])
def test_the_estimate_is_biased_low(name, level):
    """The level-n figure is not the fractal: at fine box sizes it occupies
    fewer cells than the limit set, so the fitted slope falls short.  The bias
    is systematic, not noise, and pretending otherwise would overstate what
    box counting on a finite approximation can deliver."""
    fractal, segments = _traced(name, level)
    assert estimate(segments).dimension < fractal.dimension


def test_the_bias_shrinks_as_the_level_deepens():
    fractal, _ = _traced("koch", 4)
    errors = []
    for level in (4, 6, 8):
        _, segments = _traced("koch", level)
        errors.append(abs(estimate(segments).dimension - fractal.dimension))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] > 0.2 and errors[2] < 0.03


def test_roll_off_to_slope_one_below_the_step_length():
    """Below the step length the figure is a polyline, and a polyline has
    dimension 1.  Seeing that happen is the check that the counter measures
    geometry rather than the grid."""
    _, segments = _traced("koch", 7)
    result = estimate(segments)
    below = result.slope_centres < 1.0
    assert below.any()
    assert result.slopes[below].max() < 1.15


def test_scales_outside_the_window_are_reported_but_not_fitted():
    _, segments = _traced("koch", 7)
    result = estimate(segments)
    assert result.points < len(result.eps)
    assert result.used.sum() >= 3


def test_counting_nothing_is_an_error():
    with pytest.raises(ValueError, match="no segments"):
        estimate(np.empty((0, 2, 2)))


def test_offsets_are_reported_for_an_error_bar():
    _, segments = _traced("koch", 6)
    result = estimate(segments, offsets=3)
    assert result.offset_spread.shape[0] == 3
    assert np.ptp(result.offset_spread.sum(axis=1)) > 0
