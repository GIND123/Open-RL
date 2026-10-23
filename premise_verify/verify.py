"""Label a premise set with SymPy, independently of how the generator built it.

Each premise becomes one linear equation, and a vague number ("a few", "between 3
and 6") becomes a parameter. The label follows from the solution set:

- inconsistent: no assignment satisfies every premise.
- solvable: the query has one value and every determined count is a non-negative
  integer.
- unrealistic: the query has one value, but some determined count is negative or
  fractional.
- underdetermined: two valid completions give the query different values. The two
  witnesses are found by search, so underdetermination is shown rather than assumed.
  With one free unknown (every single-edit variant) the search is exact.
- unresolved: none of the above could be shown, which can only happen with several
  free unknowns. The generator discards these.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from premise_verify.dag import apply

LABELS = ("solvable", "underdetermined", "inconsistent", "unrealistic")


@dataclass(frozen=True)
class Premise:
    node: str  # the person this premise states a count for
    kind: str  # given | add_c | sub_c | mul_c | div_c | add2 | sub2
    inputs: tuple[str, ...] = ()
    const: int | None = None
    vague: tuple[int, int] | str | None = None  # replaces `const`: a range or a word
    role: str = "core"  # core | distractor | flaw
    phrase: int = 0


@dataclass(frozen=True)
class Verdict:
    label: str
    value: Fraction | None = None  # the query's value when it is determined
    reason: str = ""


def _equation(p: Premise, sym, param):
    c = param if param is not None else p.const
    x = sym(p.node)
    ins = [sym(i) for i in p.inputs]
    if p.kind == "div_c":
        return c * x - ins[0]
    return x - apply(p.kind, ins, c)


def _fraction(r) -> Fraction:
    r = sp.Rational(r)
    return Fraction(int(r.p), int(r.q))


def _valid(v: Fraction) -> bool:
    return v.denominator == 1 and v >= 0


def verify(
    premises: list[Premise] | tuple[Premise, ...],
    query: str,
    hint: dict[str, int] | None = None,
    window: int = 30,
) -> Verdict:
    """Label `premises` with respect to `query`. `hint` (counts from a solvable
    parent) centres the witness search for unbounded unknowns."""
    node_syms: dict[str, sp.Symbol] = {}

    def sym(name: str) -> sp.Symbol:
        return node_syms.setdefault(name, sp.Symbol(name))

    ranges: dict[sp.Symbol, tuple[int, int] | None] = {}
    equations = []
    for i, p in enumerate(premises):
        param = None
        if p.vague is not None:
            param = sp.Symbol(f"_k{i}")
            ranges[param] = p.vague if isinstance(p.vague, tuple) else None
        equations.append(_equation(p, sym, param))
    sym(query)
    # Parameters go last so that elimination leaves them free whenever it can.
    unknowns = [*node_syms.values(), *ranges]
    solutions = sp.linsolve(equations, unknowns)
    if not solutions:
        return Verdict("inconsistent", reason="no assignment satisfies every premise")
    (solution,) = solutions
    free = sorted(set().union(*(e.free_symbols for e in solution)), key=str)
    forms = {
        name: (
            _fraction(expr.subs({s: 0 for s in free})),
            {s: _fraction(expr.coeff(s)) for s in free if expr.coeff(s) != 0},
        )
        for name, expr in zip(node_syms, solution)
    }

    q_const, q_coeffs = forms[query]
    if not q_coeffs:
        for name, (c0, coeffs) in forms.items():
            if not coeffs and not _valid(c0):
                return Verdict("unrealistic", q_const, reason=f"{name} = {c0}")
        return Verdict("solvable", q_const)

    # The query depends on free unknowns: search for valid completions. Whether a
    # completion keeps every count whole repeats with this period.
    period = math.lcm(*(c.denominator for _, coeffs in forms.values() for c in coeffs.values()))
    domains: dict[sp.Symbol, range | list[int]] = {}
    exhaustive = True
    if len(free) == 1:
        # Exact: each count c0 + c*s >= 0 bounds s from one side, and two periods
        # above the lower bound hold every residue twice, so two witnesses exist
        # within them whenever any valid completion does.
        (s,) = free
        lo, hi = ranges[s] if ranges.get(s) is not None else (1 if s in ranges else 0, None)
        for c0, coeffs in forms.values():
            if c := coeffs.get(s):
                if c > 0:
                    lo = max(lo, math.ceil(-c0 / c))
                else:
                    hi = math.floor(-c0 / c) if hi is None else min(hi, math.floor(-c0 / c))
        top = lo + 2 * period - 1
        domains[s] = range(lo, (top if hi is None else min(hi, top)) + 1)
    else:
        hint = hint or {}
        names = {s: name for name, s in node_syms.items()}
        for s in free:
            if ranges.get(s) is not None:
                domains[s] = range(ranges[s][0], ranges[s][1] + 1)
                continue
            exhaustive = False
            floor = 1 if s in ranges else 0
            centre = hint.get(names.get(s, ""), floor)
            around = {centre + k * period for k in range(-window, window + 1)}
            low = set(range(floor, floor + 2 * period + 2 * window))
            domains[s] = sorted(v for v in around | low if v >= floor)

    seen: set[Fraction] = set()
    for combo in itertools.islice(itertools.product(*domains.values()), 200_000):
        assignment = dict(zip(domains, combo))
        counts = {
            name: c0 + sum(c * assignment[s] for s, c in coeffs.items())
            for name, (c0, coeffs) in forms.items()
        }
        if all(_valid(v) for v in counts.values()):
            seen.add(counts[query])
            if len(seen) >= 2:
                break
    if len(seen) >= 2:
        low, high = sorted(seen)[:2]
        return Verdict("underdetermined", reason=f"{query} can be {low} or {high}")
    if exhaustive and len(seen) == 1:
        return Verdict("solvable", next(iter(seen)), reason="only one completion keeps every count a non-negative integer")
    if exhaustive:
        return Verdict("unrealistic", reason="no completion keeps every count a non-negative integer")
    return Verdict("unresolved", reason="no second valid completion found")
