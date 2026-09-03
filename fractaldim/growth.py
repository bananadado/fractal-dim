"""Recovering lambda and k, and hence the dimension, from a D0L-system.

Proposition 6.9 of the write-up bounds the box dimension of the limit curve by
``log(lambda) / log(k)``, where lambda counts how fast the drawn segments
multiply and k says how much the step length must shrink to keep the figure a
fixed size.  This module computes both from the system itself, so the table of
dimensions becomes an output rather than an input.

Two things need more care than the proposition's statement suggests.

**lambda is not always the spectral radius.**  Perron-Frobenius gives the growth
of the whole symbol count, but box counting sees only drawn symbols.  If the
dominant eigenvalue belongs to symbols that draw nothing, the two part company:
the Cantor system's matrix has ``rho(M) = 3`` while its drawn symbols grow as
``2^n``, because the third unit of every production is an undrawn gap.  What
Proposition 6.9 needs is the drawn growth rate, computed here in exact integer
arithmetic.

**k has an exact route and a measured one.**  Where a drawn symbol has a
production, the net displacement of that production *is* k -- one walk of one
word, no limit.  Where the recursion lives in symbols that draw nothing, as in
the Hilbert and dragon systems, there is no such word and k has to be measured
from the drawn figure instead.  Both are implemented, and where both apply they
are expected to agree; that agreement is the check that the algebra describes
the picture.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Mapping, Sequence

import numpy as np

from .geometry import diameter
from .lsystem import LSystem
from .turtle import Command, DEFAULT_COMMANDS, endpoint, trace

#: Commands that only reorient the turtle or manage its stack.  Symbols
#: carrying them are excluded from the substitution matrix: they have no
#: production, contribute no length, and would pad M with dead rows.  This is
#: what makes the Hilbert matrix 3x3 over {A, B, F}, as in Example 6.5.
_ORIENTATION = frozenset(
    {Command.LEFT, Command.RIGHT, Command.REVERSE, Command.PUSH, Command.POP}
)

#: Levels used when k has to be measured.  Deep enough for the transient to
#: decay, shallow enough to trace in a moment.
_MAX_LEVEL = 32
_SYMBOL_BUDGET = 2_000_000


def command_table(commands: Mapping[str, Command] | None = None) -> dict[str, Command]:
    """The default symbol->command table, with ``commands`` layered on top."""
    table = dict(DEFAULT_COMMANDS)
    if commands:
        table.update(commands)
    return table


def matrix_alphabet(system: LSystem, table: Mapping[str, Command]) -> str:
    """Symbols the substitution matrix is indexed by, in sorted order."""
    return "".join(
        symbol for symbol in system.alphabet
        if table.get(symbol, Command.NOP) not in _ORIENTATION
    )


def substitution_matrix(
    system: LSystem, commands: Mapping[str, Command] | None = None
) -> tuple[str, np.ndarray]:
    """The matrix ``M[i, j] = #occurrences of a_j in P(a_i)``.

    Returns the alphabet it is indexed by alongside it.  Rows are productions,
    matching the write-up's convention that ``v_{n+1} = v_n M``.
    """
    table = command_table(commands)
    alphabet = matrix_alphabet(system, table)
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    matrix = np.zeros((len(alphabet), len(alphabet)), dtype=np.int64)
    for symbol, row in zip(alphabet, matrix):
        for produced in system.production(symbol):
            j = index.get(produced)
            if j is not None:
                row[j] += 1
    return alphabet, matrix


def drawn_symbols(system: LSystem, commands: Mapping[str, Command] | None = None) -> str:
    """Symbols of the alphabet that lay down a segment."""
    table = command_table(commands)
    return "".join(
        symbol for symbol in system.alphabet
        if table.get(symbol, Command.NOP) is Command.DRAW
    )


def drawn_counts(
    system: LSystem,
    levels: Sequence[int],
    commands: Mapping[str, Command] | None = None,
) -> dict[int, int]:
    """Exact number of drawn symbols at each requested level.

    Iterates ``v_{n+1} = v_n M`` in Python integers rather than raising M to a
    power in floating point, so the counts are exact however deep they go --
    which is what lets the growth rate be read off cleanly at a level where the
    subdominant terms have decayed below double precision.
    """
    table = command_table(commands)
    alphabet, matrix = substitution_matrix(system, commands)
    index = {symbol: i for i, symbol in enumerate(alphabet)}
    rows = matrix.tolist()
    is_drawn = [table.get(symbol, Command.NOP) is Command.DRAW for symbol in alphabet]

    vector = [0] * len(alphabet)
    for symbol in system.axiom:
        i = index.get(symbol)
        if i is not None:
            vector[i] += 1

    wanted = set(levels)
    counts: dict[int, int] = {}
    for level in range(max(levels, default=0) + 1):
        if level in wanted:
            counts[level] = sum(v for v, d in zip(vector, is_drawn) if d)
        vector = [
            sum(v * row[j] for v, row in zip(vector, rows))
            for j in range(len(alphabet))
        ]
    return counts


def growth_rate(
    system: LSystem, commands: Mapping[str, Command] | None = None, level: int = 40
) -> float | None:
    """The rate at which drawn symbols multiply per rewriting step.

    This is the ``lambda`` of Proposition 6.9.  Usually it equals ``rho(M)``,
    but not when the dominant eigenvalue belongs to undrawn symbols; see
    :func:`spectral_radius` and the module docstring.
    """
    counts = drawn_counts(system, (level, level + 1), commands)
    previous, current = counts[level], counts[level + 1]
    if not previous:
        return None
    return float(Fraction(current, previous))


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest eigenvalue modulus of the substitution matrix."""
    if matrix.size == 0:
        return 0.0
    return float(np.abs(np.linalg.eigvals(matrix.astype(np.float64))).max())


def is_irreducible(matrix: np.ndarray) -> bool:
    """True if M's digraph is strongly connected.

    Remark 6.7: when it is not, Perron-Frobenius applies only componentwise and
    the growth rate is that of the dominant strongly connected component, which
    need not be the component containing the drawn symbols.
    """
    size = len(matrix)
    if size <= 1:
        return True
    reach = matrix.astype(bool)
    closure = reach | np.eye(size, dtype=bool)
    for _ in range(size):
        closure = closure | (closure @ closure)
    return bool(closure.all())


def scaling_factor_exact(
    system: LSystem, commands: Mapping[str, Command] | None = None
) -> float | None:
    """``k`` from the grammar: the net displacement of a drawn production.

    A drawn symbol spans one step.  If its production carries the turtle ``k``
    steps end to end, then the level-n figure is ``k`` times the size of the
    level-(n-1) one, so the step length must shrink by ``k`` per level to hold
    the figure still -- which is the definition of the scaling factor.

    Returns ``None`` when no drawn symbol has a production (the Hilbert and
    dragon systems, where the recursion is carried by symbols that draw
    nothing), or when two drawn symbols disagree, since a single scaling factor
    then does not describe the system.
    """
    table = command_table(commands)
    factors = []
    for symbol in drawn_symbols(system, commands):
        production = system.production(symbol)
        if production == symbol:
            continue
        x, y = endpoint(production, angle=system.angle, commands=table)
        factors.append(math.hypot(x, y))
    if not factors:
        return None
    if max(factors) - min(factors) > 1e-9 * max(factors):
        return None
    return factors[0]


def level_counts(
    system: LSystem, commands: Mapping[str, Command] | None = None
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(level, word length, drawn symbols)`` exactly, level by level.

    Both counts come from the production multiplicities alone, so the cost of
    picking a level to trace does not depend on how big that level is.  The two
    can grow at quite different rates -- the Cantor word grows as ``3^n`` while
    only ``2^n`` of it draws -- and it is the word length that bounds the work.
    """
    productions = {
        symbol: Counter(system.production(symbol)) for symbol in system.alphabet
    }
    table = command_table(commands)
    counts = Counter(system.axiom)
    level = 0
    while True:
        drawn = sum(
            value for symbol, value in counts.items()
            if table.get(symbol, Command.NOP) is Command.DRAW
        )
        yield level, sum(counts.values()), drawn
        following: Counter[str] = Counter()
        for symbol, value in counts.items():
            if value:
                for produced, multiplicity in productions[symbol].items():
                    following[produced] += value * multiplicity
        counts = following
        level += 1


def deepest_levels(
    system: LSystem,
    commands: Mapping[str, Command] | None = None,
    budget: int = _SYMBOL_BUDGET,
    count: int = 3,
) -> list[int]:
    """The deepest ``count`` levels whose words stay inside ``budget`` symbols."""
    affordable = []
    for level, length, drawn in level_counts(system, commands):
        if level > _MAX_LEVEL or length > budget:
            break
        if drawn >= 2:
            affordable.append(level)
    return affordable[-count:]


def scaling_factor_measured(
    system: LSystem,
    commands: Mapping[str, Command] | None = None,
    start_heading: float = 0.0,
    budget: int = _SYMBOL_BUDGET,
) -> tuple[float | None, int]:
    """``k`` measured from the drawn figure, with the deepest level used.

    Uses the true diameter rather than the bounding box, because the box is not
    rotation invariant and the dragon turns as it grows.  Then takes the ratio
    of successive *differences* rather than of the diameters themselves: the
    Hilbert figure spans ``2^n - 1``, and differencing cancels that additive
    ``-1`` identically instead of waiting for it to decay.
    """
    levels = deepest_levels(system, commands, budget)
    if len(levels) < 3:
        return None, max(levels, default=0)

    sizes = [
        diameter(trace(system, level, commands=commands,
                       start_heading=start_heading))
        for level in levels
    ]
    denominator = sizes[-2] - sizes[-3]
    if abs(denominator) < 1e-12:
        return None, levels[-1]
    return (sizes[-1] - sizes[-2]) / denominator, levels[-1]


@dataclass(frozen=True)
class Growth:
    """Everything Proposition 6.9 needs, recovered from the system."""

    name: str
    alphabet: str
    matrix: np.ndarray
    drawn: str
    lam: float | None            # growth rate of the drawn symbols
    rho: float | None            # spectral radius of M
    k: float | None
    k_method: str                # "exact", "measured", or "none"
    k_measured: float | None     # always populated, for cross-checking
    measured_level: int
    irreducible: bool
    branching: bool
    dimension: float | None

    @property
    def dominant_is_undrawn(self) -> bool:
        """True when rho(M) overstates the drawn growth rate.

        The Cantor case: the dominant eigenvalue lives in the undrawn block, so
        using rho(M) in Proposition 6.9 would give the wrong dimension.
        """
        if self.lam is None or self.rho is None:
            return False
        return self.rho > self.lam * (1.0 + 1e-6)

    @property
    def k_agrees(self) -> bool | None:
        """Whether the exact and measured scaling factors match, if both exist."""
        if self.k_method != "exact" or self.k_measured is None:
            return None
        return abs(self.k_measured - self.k) <= 0.02 * self.k


def analyse(
    system: LSystem,
    commands: Mapping[str, Command] | None = None,
    start_heading: float = 0.0,
) -> Growth:
    """Recover lambda, k and the dimension of ``system``."""
    table = command_table(commands)
    alphabet, matrix = substitution_matrix(system, commands)
    lam = growth_rate(system, commands)
    rho = spectral_radius(matrix)

    exact = scaling_factor_exact(system, commands)
    measured, level = scaling_factor_measured(
        system, commands, start_heading=start_heading
    )
    if exact is not None:
        k, method = exact, "exact"
    elif measured is not None:
        k, method = measured, "measured"
    else:
        k, method = None, "none"

    # Bracketed systems branch, so the level-n figure is not a union of
    # congruent scaled copies and Moran's theorem does not apply -- k and
    # lambda are still meaningful, their ratio is not a dimension.
    branching = any(
        table.get(symbol) in (Command.PUSH, Command.POP) for symbol in system.alphabet
    )

    dimension = None
    if not branching and lam and lam > 1 and k and k > 1:
        dimension = math.log(lam) / math.log(k)

    return Growth(
        name=system.name,
        alphabet=alphabet,
        matrix=matrix,
        drawn=drawn_symbols(system, commands),
        lam=lam,
        rho=rho,
        k=k,
        k_method=method,
        k_measured=measured,
        measured_level=level,
        irreducible=is_irreducible(matrix),
        branching=branching,
        dimension=dimension,
    )
