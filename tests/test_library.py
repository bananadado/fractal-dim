"""The catalogue."""

import math

import pytest

from fractaldim import library
from fractaldim.turtle import trace


def test_unknown_name_lists_the_alternatives():
    with pytest.raises(KeyError, match=r"hilbert"):
        library.get("nonesuch")


def test_validation_family_comes_first():
    assert library.names()[: len(library.VALIDATION_FAMILY)] == list(
        library.VALIDATION_FAMILY
    )


@pytest.mark.parametrize("name", library.names())
def test_every_entry_draws_something(name):
    fractal = library.get(name)
    segments = trace(
        fractal.system, 3,
        commands=fractal.commands, start_heading=fractal.start_heading,
    )
    assert len(segments) > 0


@pytest.mark.parametrize("name", library.names())
def test_reference_dimension_is_consistent_with_lambda_and_k(name):
    """The recorded dimension must equal log(lambda) / log(k)."""
    fractal = library.get(name)
    if fractal.dimension is None:
        assert fractal.lam is None and fractal.k is None
        return
    assert math.log(fractal.lam) / math.log(fractal.k) == pytest.approx(
        fractal.dimension, rel=1e-12
    )


def test_recorded_dimensions_match_the_writeup_table():
    expected = {
        "cantor": 0.630930,
        "koch": 1.261860,
        "minkowski": 1.5,
        "arrowhead": 1.584963,
        "peano": 2.0,
        "hilbert": 2.0,
    }
    for name, value in expected.items():
        assert library.get(name).dimension == pytest.approx(value, abs=5e-7)
