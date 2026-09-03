"""Recovering lambda, k and the dimension from a system."""

import math

import numpy as np
import pytest

from fractaldim import library
from fractaldim.growth import (
    analyse,
    deepest_levels,
    drawn_counts,
    growth_rate,
    is_irreducible,
    level_counts,
    scaling_factor_exact,
    scaling_factor_measured,
    spectral_radius,
    substitution_matrix,
)
from fractaldim.lsystem import LSystem
from fractaldim.turtle import trace


def test_hilbert_matrix_matches_the_writeup():
    """Example 6.5, verbatim: M over {A, B, F} with eigenvalues {4, 1, 0}."""
    alphabet, matrix = substitution_matrix(library.get("hilbert").system)
    assert alphabet == "ABF"
    np.testing.assert_array_equal(matrix, [[2, 2, 3], [2, 2, 3], [0, 0, 1]])
    eigenvalues = sorted(round(abs(v), 9) for v in np.linalg.eigvals(matrix))
    assert eigenvalues == [0.0, 1.0, 4.0]


def test_orientation_symbols_are_left_out_of_the_matrix():
    """'+' and '-' carry no production and no length, so they pad M with
    dead rows; excluding them is what makes the Hilbert matrix 3x3."""
    alphabet, _ = substitution_matrix(library.get("koch").system)
    assert alphabet == "F"


def test_drawn_counts_agree_with_the_traced_figure():
    for name in ("koch", "hilbert", "gosper", "cantor"):
        fractal = library.get(name)
        counts = drawn_counts(fractal.system, range(5), fractal.commands)
        for level, expected in counts.items():
            segments = trace(fractal.system, level, commands=fractal.commands)
            assert len(segments) == expected


def test_level_counts_agree_with_the_expanded_word():
    system = library.get("hilbert").system
    for level, length, drawn in level_counts(system):
        if level > 5:
            break
        word = system.expand(level)
        assert length == len(word)
        assert drawn == word.count("F")


def test_cantor_growth_rate_is_not_the_spectral_radius():
    """The write-up's lambda is the *drawn* growth rate.

    With 'f->fff' the dominant eigenvalue of M is 3, but it belongs to the
    undrawn gaps; the drawn symbols only double. Using rho(M) in Proposition
    6.9 would give dimension 1, not log(2)/log(3).
    """
    system = library.get("cantor").system
    _, matrix = substitution_matrix(system)
    assert spectral_radius(matrix) == pytest.approx(3.0)
    assert growth_rate(system) == pytest.approx(2.0)

    result = analyse(system)
    assert result.dominant_is_undrawn
    assert result.dimension == pytest.approx(math.log(2) / math.log(3))


def test_reducibility_is_detected():
    # Hilbert's F is a sink: reachable from A and B, reaching neither.
    _, hilbert = substitution_matrix(library.get("hilbert").system)
    assert not is_irreducible(hilbert)
    # Koch has the single symbol F, which reproduces itself.
    _, koch = substitution_matrix(library.get("koch").system)
    assert is_irreducible(koch)


@pytest.mark.parametrize("name,expected", [
    ("cantor", 3.0), ("koch", 3.0), ("minkowski", 4.0), ("arrowhead", 2.0),
    ("peano", 3.0), ("sierpinski_gasket", 2.0), ("koch_snowflake", 3.0),
    ("levy_c", math.sqrt(2)), ("gosper", math.sqrt(7)),
])
def test_exact_scaling_factor_is_the_production_displacement(name, expected):
    fractal = library.get(name)
    k = scaling_factor_exact(fractal.system, fractal.commands)
    assert k == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name", ["hilbert", "dragon"])
def test_node_rewriting_systems_have_no_exact_scaling_factor(name):
    """The recursion is carried by symbols that draw nothing, so there is no
    drawn production to walk and k has to be measured instead."""
    fractal = library.get(name)
    assert scaling_factor_exact(fractal.system, fractal.commands) is None
    assert analyse(fractal.system, fractal.commands,
                   fractal.start_heading).k_method == "measured"


def test_hilbert_measured_k_is_exact_despite_the_boundary_term():
    """The figure spans 2^n - 1; differencing cancels the -1 identically."""
    fractal = library.get("hilbert")
    k, _ = scaling_factor_measured(fractal.system, fractal.commands)
    assert k == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("name", sorted(library.LIBRARY))
def test_exact_and_measured_scaling_factors_agree(name):
    """The algebra has to describe the picture: where both routes apply, the
    displacement of one production must match how the drawn figure grows."""
    fractal = library.get(name)
    result = analyse(fractal.system, fractal.commands, fractal.start_heading)
    if result.k_method != "exact":
        pytest.skip("no exact route for this system")
    assert result.k_agrees, f"{result.k} vs measured {result.k_measured}"


@pytest.mark.parametrize("name", library.VALIDATION_FAMILY)
def test_validation_family_dimensions_are_reproduced(name):
    """The write-up's Table 1, recomputed from the grammar rather than quoted."""
    fractal = library.get(name)
    result = analyse(fractal.system, fractal.commands, fractal.start_heading)
    assert result.dimension == pytest.approx(fractal.dimension, abs=1e-9)


def test_dragon_dimension_is_close_but_only_measured():
    """The one entry with no exact route and slow convergence; 0.2% is what
    the diameter ratio achieves at the deepest affordable level."""
    fractal = library.get("dragon")
    result = analyse(fractal.system, fractal.commands, fractal.start_heading)
    assert result.k_method == "measured"
    assert result.dimension == pytest.approx(2.0, rel=2e-3)


def test_branching_systems_report_no_dimension():
    """The plant scales by 2 per level and its drawn symbols quadruple, but it
    branches, so the figure is not a union of congruent scaled copies and
    log(lambda)/log(k) is not its dimension."""
    fractal = library.get("plant")
    result = analyse(fractal.system, fractal.commands, fractal.start_heading)
    assert result.branching
    assert result.k == pytest.approx(2.0)
    assert result.dimension is None


def test_deepest_levels_respects_the_symbol_budget():
    """Cantor's word grows as 3^n while only 2^n of it draws, so the budget
    has to count symbols, not segments."""
    fractal = library.get("cantor")
    levels = deepest_levels(fractal.system, fractal.commands, budget=1000)
    assert levels == [4, 5, 6]
    assert len(fractal.system.expand(6)) <= 1000 < len(fractal.system.expand(7))


def test_a_system_that_draws_nothing_has_no_growth_rate():
    assert growth_rate(LSystem("X", {"X": "XX"}, 90.0)) is None
