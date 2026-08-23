"""Deterministic context-free L-systems (D0L-systems).

A D0L-system is a triple ``G = (V, omega, P)``: a finite alphabet ``V``, an
axiom ``omega``, and a production ``P`` assigning a word to each symbol.  A
rewriting step replaces *every* symbol of the current word simultaneously by its
production.  The parallelism matters: it is what makes every part of the word
grow at the same rate, which is what gives the limit figure its self-similarity.

Symbols with no production rewrite to themselves.  That is the usual convention
and it means constants (``+``, ``-``, ``[``, ``]``) need no declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping


@dataclass(frozen=True)
class LSystem:
    """A D0L-system.

    The grammar alone does not determine a curve.  The turning angle is bundled
    here because it is a property of the symbol alphabet's intended reading, but
    the step length -- and hence the scale of the drawn figure -- is supplied
    separately by the turtle.
    """

    axiom: str
    rules: Mapping[str, str]
    angle: float = 90.0
    name: str = ""

    _table: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.axiom:
            raise ValueError("axiom must be a non-empty word")
        for symbol in self.rules:
            if len(symbol) != 1:
                raise ValueError(
                    f"productions are keyed by single symbols, got {symbol!r}"
                )
        # str.maketrans accepts one-to-many mappings, so a whole parallel
        # rewriting step is a single C-level pass over the word.
        object.__setattr__(self, "_table", str.maketrans(dict(self.rules)))

    @property
    def alphabet(self) -> str:
        """Every symbol appearing in the axiom or in any production."""
        seen = set(self.axiom)
        for symbol, production in self.rules.items():
            seen.add(symbol)
            seen.update(production)
        return "".join(sorted(seen))

    def production(self, symbol: str) -> str:
        """The production for ``symbol``; the symbol itself if it has none."""
        return self.rules.get(symbol, symbol)

    def is_constant(self, symbol: str) -> bool:
        """True if rewriting ``symbol`` can never change it."""
        return self.production(symbol) == symbol

    def expand(self, n: int) -> str:
        """Return the word after ``n`` rewriting steps.

        Materialises the whole word, so it is the right choice for drawing and
        inspection but not for deep levels -- level 12 of the Hilbert system is
        roughly sixty million symbols.  Use :meth:`expand_iter` there.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        word = self.axiom
        for _ in range(n):
            word = word.translate(self._table)
        return word

    def expand_iter(self, n: int) -> Iterator[str]:
        """Yield the symbols of the level-``n`` word one at a time.

        Walks the derivation tree depth-first with an explicit stack instead of
        building the word, so memory is O(n * max production length) rather than
        O(lambda^n).  The symbols come out in the same order as ``expand(n)``.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        rules = self.rules
        # Each frame is [word, next index into word, depth of that word].
        stack: list[list] = [[self.axiom, 0, 0]]
        while stack:
            frame = stack[-1]
            word, i, depth = frame
            if i >= len(word):
                stack.pop()
                continue
            frame[1] = i + 1
            symbol = word[i]
            production = rules.get(symbol, symbol)
            if depth >= n or production == symbol:
                # At the target depth, or a constant that further rewriting
                # would leave untouched -- either way, emit it now rather than
                # pushing frames for the remaining levels.
                yield symbol
            elif production:
                stack.append([production, 0, depth + 1])

    def symbol_counts(self, n: int) -> dict[str, int]:
        """Symbol multiplicities of the level-``n`` word.

        Computed by streaming, so it stays usable at levels where the word
        itself would not fit in memory.
        """
        counts: dict[str, int] = {}
        for symbol in self.expand_iter(n):
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    def __str__(self) -> str:
        rules = ", ".join(f"{s}->{p}" for s, p in sorted(self.rules.items()))
        label = self.name or "L-system"
        return f"{label}: axiom={self.axiom}, {rules}, delta={self.angle:g}"
