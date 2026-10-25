"""The cost of counting below the step length."""

import numpy as np
import pytest

from fractaldim import library
from fractaldim.rolloff import sweep, sweep_levels
from fractaldim.turtle import trace


def _traced(name, level):
    fractal = library.get(name)
    return fractal, trace(fractal.system, level, commands=fractal.commands,
                          start_heading=fractal.start_heading)


def test_every_fit_shares_its_coarse_end():
    """Only the bottom of the fit varies, which is what makes the curve
    readable as a statement about eps -> 0 rather than about window width."""
    _, segments = _traced("koch", 5)
    result = sweep(segments, level=5, exact=library.get("koch").dimension)
    assert np.all(np.diff(result.cutoffs) > 0)
    # One point drops out of the fit for each step up the ladder.
    assert np.all(np.diff(result.used) == -1)
    assert result.used.min() >= 3


def test_counting_finer_than_a_step_always_costs_accuracy():
    """The left arm of the V.  Below the step the figure is line segments, so
    every scale added there pulls the fitted slope towards 1."""
    fractal, segments = _traced("koch", 6)
    result = sweep(segments, level=6, exact=fractal.dimension)
    errors = result.errors[result.cutoffs <= result.step]
    assert np.all(errors < 0)
    # Monotone: each finer scale let in is strictly worse than the last.
    assert np.all(np.diff(errors) > 0)


def test_the_floor_sits_above_the_finest_box_counted():
    """The headline.  The best estimate is not the one taken at the smallest
    box affordable, so 'eps -> 0' is not an instruction the estimator obeys."""
    fractal, segments = _traced("koch", 7)
    result = sweep(segments, level=7, exact=fractal.dimension)
    assert result.cutoffs[result.best] > result.cutoffs[0]
    assert abs(result.errors[result.best]) < abs(result.errors[0]) / 10


def test_deeper_levels_push_the_whole_curve_down():
    """A deeper curve, not a smaller box, is what buys accuracy."""
    fractal = library.get("koch")
    results = sweep_levels(fractal, [5, 6, 7], below_step=8)
    finest = [abs(result.errors[0]) for result in results]
    assert finest == sorted(finest, reverse=True)
    best = [abs(result.errors[result.best]) for result in results]
    assert best == sorted(best, reverse=True)


def test_a_dyadic_curve_can_sit_exactly_on_its_dimension():
    """Hilbert's scaling factor matches the grid, so at the step length the
    fit has no residual at all -- and still collapses below it."""
    fractal, segments = _traced("hilbert", 7)
    result = sweep(segments, level=7, exact=fractal.dimension)
    assert result.errors[result.best] == pytest.approx(0.0, abs=1e-9)
    assert result.errors[0] < -0.5


def test_too_shallow_a_curve_has_no_fit_to_report():
    _, segments = _traced("koch", 2)
    with pytest.raises(ValueError, match="deeper level"):
        sweep(segments, level=2, exact=1.26, below_step=1)


def test_error_is_unknown_without_a_reference():
    _, segments = _traced("koch", 4)
    result = sweep(segments, level=4, below_step=4)
    assert np.all(np.isnan(result.errors))
    with pytest.raises(ValueError, match="reference dimension"):
        result.best
