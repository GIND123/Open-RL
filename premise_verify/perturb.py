"""Flaw operators. Each edits one premise of a solvable parent, so a parent and its
variants form minimal pairs.

- delete: remove a premise the query depends on (missing premise).
- contradict: restate a given count with a different value (`restate`), or state a
  wrong value for a derived count (`derived`, which only computation reveals).
- unrealistic: change one stated number so that some count turns negative or
  fractional.
- ambiguate: replace one stated number with a range or a vague word.
- drop_distractor: a control that deletes a premise the query does not depend on,
  so the text loses a sentence but the answer is unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from premise_verify.dag import Graph, Values
from premise_verify.themes import DIVISORS, VAGUE_PHRASES, VAGUE_WORDS
from premise_verify.verify import Premise

FLAWS = ("delete", "contradict", "unrealistic", "ambiguate")


@dataclass(frozen=True)
class Variant:
    tag: str
    premises: tuple[Premise, ...]
    expected: str  # the label the edit should produce
    flaw: str | None = None
    mode: str | None = None
    control: str | None = None
    target: str | None = None  # node whose premise was edited
    original: Premise | None = None  # premise removed or rewritten
    edited: Premise | None = None  # premise inserted or rewritten


def parent_premises(graph: Graph, values: Values, order: list[str]) -> tuple[Premise, ...]:
    return tuple(
        Premise(
            node=n,
            kind=graph.nodes[n].kind,
            inputs=graph.nodes[n].inputs,
            const=values.consts.get(n),
            role=graph.nodes[n].role,
            phrase=graph.nodes[n].phrase,
        )
        for n in order
    )


def _index(premises: tuple[Premise, ...], node: str) -> int:
    return next(i for i, p in enumerate(premises) if p.node == node and p.role != "flaw")


def _replace(premises, i, new):
    return premises[:i] + (new,) + premises[i + 1 :]


def deletions(
    premises: tuple[Premise, ...],
    graph: Graph,
    rng: random.Random,
    k: int | None,
    include_query: bool = False,
) -> list[Variant]:
    candidates = [n for n in graph.ancestors() if include_query or n != graph.query]
    chosen = candidates if k is None else rng.sample(candidates, min(k, len(candidates)))
    out = []
    for node in sorted(chosen, key=candidates.index):
        i = _index(premises, node)
        out.append(
            Variant(
                f"del-{node}",
                premises[:i] + premises[i + 1 :],
                "underdetermined",
                flaw="delete",
                target=node,
                original=premises[i],
            )
        )
    return out


def contradiction(
    premises: tuple[Premise, ...],
    graph: Graph,
    values: Values,
    rng: random.Random,
    avoid: int,
) -> Variant | None:
    core = [n for n in graph.ancestors() if n != graph.query]
    modes = {
        "restate": [n for n in core if graph.nodes[n].kind == "given"],
        "derived": [n for n in core if graph.nodes[n].kind != "given"],
    }
    modes = {m: nodes for m, nodes in modes.items() if nodes}
    if not modes:
        return None
    mode = rng.choice(sorted(modes))
    node = rng.choice(modes[mode])
    true = values.values[node]
    stated = true
    while stated in (true, avoid) or stated < 2:
        stated = true + rng.choice((-1, 1)) * rng.randint(1, max(3, true // 5))
    i = _index(premises, node)
    statement = Premise(node, "given", const=stated, role="flaw", phrase=1)
    at = rng.randint(i + 1, len(premises))
    return Variant(
        "con",
        premises[:at] + (statement,) + premises[at:],
        "inconsistent",
        flaw="contradict",
        mode=mode,
        target=node,
        edited=statement,
    )


def unrealistic(
    premises: tuple[Premise, ...],
    graph: Graph,
    values: Values,
    rng: random.Random,
    magnitude: tuple[int, int],
) -> Variant | None:
    lo, hi = magnitude
    edits = []
    for n in graph.ancestors():
        node = graph.nodes[n]
        if node.kind in ("sub_c", "add_c", "mul_c"):
            edits.append((n, "negative"))
        elif node.kind == "div_c":
            a = values.values[node.inputs[0]]
            if any(a % d for d in DIVISORS):
                edits.append((n, "fractional"))
    if not edits:
        return None
    node, mode = rng.choice(edits)
    i = _index(premises, node)
    old = premises[i]
    a = values.values[old.inputs[0]]
    if mode == "negative":
        new = Premise(node, "sub_c", old.inputs, a + rng.randint(lo, hi), role="flaw", phrase=old.phrase)
    else:
        new = Premise(node, "div_c", old.inputs, rng.choice([d for d in DIVISORS if a % d]), role="flaw", phrase=old.phrase)
    return Variant(
        "unr",
        _replace(premises, i, new),
        "unrealistic",
        flaw="unrealistic",
        mode=mode,
        target=node,
        original=old,
        edited=new,
    )


def ambiguation(
    premises: tuple[Premise, ...],
    graph: Graph,
    values: Values,
    rng: random.Random,
    magnitude: tuple[int, int],
) -> Variant | None:
    lo, _ = magnitude
    nodes = [n for n in graph.ancestors() if graph.nodes[n].kind in VAGUE_PHRASES]
    if not nodes:
        return None
    node = rng.choice(nodes)
    i = _index(premises, node)
    old = premises[i]
    if rng.random() < 0.5:
        mode = "range"
        spread = max(2, old.const // 4)
        low = max(lo if old.kind != "sub_c" else 1, old.const - rng.randint(1, spread))
        vague: tuple[int, int] | str = (low, max(low + 2, old.const + rng.randint(1, spread)))
    else:
        mode = "word"
        vague = rng.choice(VAGUE_WORDS)
    new = Premise(node, old.kind, old.inputs, None, vague, role="flaw", phrase=old.phrase)
    return Variant(
        "amb",
        _replace(premises, i, new),
        "underdetermined",
        flaw="ambiguate",
        mode=mode,
        target=node,
        original=old,
        edited=new,
    )


def drop_distractor(premises: tuple[Premise, ...], rng: random.Random) -> Variant | None:
    distractors = [i for i, p in enumerate(premises) if p.role == "distractor"]
    if not distractors:
        return None
    i = rng.choice(distractors)
    return Variant(
        "ctl",
        premises[:i] + premises[i + 1 :],
        "solvable",
        control="drop_distractor",
        target=premises[i].node,
        original=premises[i],
    )
