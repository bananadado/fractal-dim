"""Expansion of D0L-systems."""

import pytest

from fractaldim import library
from fractaldim.lsystem import LSystem


def test_missing_production_is_identity():
    system = LSystem("F+X", {"F": "FF"})
    assert system.expand(1) == "FF+X"
    assert system.production("+") == "+"
    assert system.is_constant("+")
    assert not system.is_constant("F")


def test_rewriting_is_parallel_not_sequential():
    # A single step rewrites every symbol once. If the output were rescanned,
    # 'A' would keep expanding within the same step and never terminate.
    system = LSystem("A", {"A": "AB", "B": "A"})
    assert system.expand(1) == "AB"
    assert system.expand(2) == "ABA"
    assert system.expand(3) == "ABAAB"
    # Fibonacci word lengths, the standard check on parallel rewriting.
    assert [len(system.expand(n)) for n in range(8)] == [1, 2, 3, 5, 8, 13, 21, 34]


def test_axiom_must_be_non_empty():
    with pytest.raises(ValueError):
        LSystem("", {"F": "FF"})


def test_productions_are_keyed_by_single_symbols():
    with pytest.raises(ValueError):
        LSystem("F", {"FF": "F"})


@pytest.mark.parametrize("name", library.names())
def test_streamed_expansion_matches_materialised(name):
    """expand_iter must yield exactly the symbols of expand, in order."""
    system = library.get(name).system
    for n in range(5):
        assert "".join(system.expand_iter(n)) == system.expand(n)


def test_hilbert_draws_four_to_the_n_minus_one_segments():
    """The write-up's Example 6.5: F-count is 4^n - 1, not 3^n."""
    system = library.get("hilbert").system
    for n in range(9):
        assert system.expand(n).count("F") == 4**n - 1


def test_koch_draws_four_to_the_n_segments():
    system = library.get("koch").system
    for n in range(8):
        assert system.expand(n).count("F") == 4**n


def test_cantor_gaps_keep_pace_with_segments():
    """Every symbol must expand to three units, or the figure is not Cantor's.

    'F->FfF' spans three units, so 'f' must too. The write-up's table gives
    'f->ff', under which the gaps fall behind and the level-n figure spans
    neither 3^n nor any other power.
    """
    system = library.get("cantor").system
    for n in range(7):
        word = system.expand(n)
        assert len(word) == 3**n
        assert word.count("F") == 2**n


def test_symbol_counts_agree_with_expansion():
    system = library.get("peano").system
    counts = system.symbol_counts(4)
    word = system.expand(4)
    assert counts == {symbol: word.count(symbol) for symbol in set(word)}


def test_alphabet_collects_axiom_and_productions():
    assert library.get("hilbert").system.alphabet == "+-ABF"


def test_negative_level_rejected():
    system = library.get("koch").system
    with pytest.raises(ValueError):
        system.expand(-1)
    with pytest.raises(ValueError):
        list(system.expand_iter(-1))
