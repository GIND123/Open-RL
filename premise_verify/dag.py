"""Solution graphs: each node is one person's count, and one premise defines each node.

A graph is a tree rooted at the query. Leaves are `given` counts; internal nodes apply
one operation, either to one input and a stated number (`add_c`, `sub_c`, `mul_c`,
`div_c`) or to two inputs (`add2`, `sub2`). Depth is the longest path from a leaf to
the query, counted in operations; `n_ops` counts every operation the query depends
on, the analogue of GSM8K's calculator-chain length. Every tree with `n_ops`
operations states exactly `n_ops + 1` numbers whatever its shape, so varying depth at
fixed `n_ops` separates depth from the number of quantities.
"""

from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import dataclass, field

from premise_verify.themes import DIVISORS, PHRASES

UNARY_KINDS = ("add_c", "sub_c", "mul_c", "div_c")
UNARY_WEIGHTS = (0.3, 0.3, 0.25, 0.15)
BINARY_KINDS = ("add2", "sub2")
BINARY_WEIGHTS = (0.6, 0.4)

# Range of given counts and additive operands. Multipliers are always 2..9 and
# divisors come from DIVISORS, whatever the magnitude.
MAGNITUDES = {"small": (2, 9), "medium": (10, 99), "large": (100, 999)}
MAX_VALUE = 10**6


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    inputs: tuple[str, ...] = ()
    phrase: int = 0
    role: str = "core"  # "core" feeds the query; "distractor" does not


@dataclass
class Graph:
    nodes: dict[str, Node]  # topological: inputs precede consumers
    query: str
    _consumers: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for node in self.nodes.values():
            for parent in node.inputs:
                self._consumers.setdefault(parent, []).append(node.id)

    def ancestors(self, node_id: str | None = None) -> list[str]:
        """Nodes the given node (default: the query) depends on, itself included,
        in topological order."""
        stack, seen = [node_id or self.query], set()
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self.nodes[current].inputs)
        return [n for n in self.nodes if n in seen]

    def depth(self, node_id: str | None = None) -> int:
        node = self.nodes[node_id or self.query]
        return 0 if not node.inputs else 1 + max(self.depth(i) for i in node.inputs)

    def n_ops(self) -> int:
        return sum(self.nodes[n].kind != "given" for n in self.ancestors())

    def distance(self, node_id: str) -> int | None:
        """Edges on the shortest path from `node_id` to the query; None if the query
        does not depend on it."""
        queue, seen = deque([(self.query, 0)]), {self.query}
        while queue:
            current, dist = queue.popleft()
            if current == node_id:
                return dist
            for parent in self.nodes[current].inputs:
                if parent not in seen:
                    seen.add(parent)
                    queue.append((parent, dist + 1))
        return None


def max_ops(depth: int) -> int:
    return 2**depth - 1


def build_tree(rng: random.Random, depth: int, n_ops: int, p_binary: float = 0.5) -> Graph:
    """A random tree whose query has exactly `depth` and `n_ops`."""
    if not depth <= n_ops <= max_ops(depth):
        raise ValueError(f"no tree has depth {depth} and {n_ops} operations")
    nodes: dict[str, Node] = {}
    ids = (f"n{i}" for i in itertools.count())

    def make(h: int, n: int) -> str:
        if h == 0:
            node = Node(next(ids), "given", phrase=rng.randrange(len(PHRASES["given"])))
            nodes[node.id] = node
            return node.id
        unary_ok = h - 1 <= n - 1 <= max_ops(h - 1)
        splits = [
            (h2, n1, n - 1 - n1)
            for h2 in range(h)
            for n1 in range(h - 1, min(max_ops(h - 1), n - 1) + 1)
            if h2 <= n - 1 - n1 <= max_ops(h2)
        ]
        if unary_ok and (not splits or rng.random() >= p_binary):
            kind = rng.choices(UNARY_KINDS, UNARY_WEIGHTS)[0]
            inputs: tuple[str, ...] = (make(h - 1, n - 1),)
        else:
            h2, n1, n2 = rng.choice(splits)
            pair = (make(h - 1, n1), make(h2, n2))
            inputs = pair if rng.random() < 0.5 else pair[::-1]
            kind = rng.choices(BINARY_KINDS, BINARY_WEIGHTS)[0]
        node = Node(next(ids), kind, inputs, rng.randrange(len(PHRASES[kind])))
        nodes[node.id] = node
        return node.id

    query = make(depth, n_ops)
    return Graph(nodes, query)


def add_distractors(graph: Graph, rng: random.Random, width: int) -> Graph:
    """Add `width` premises the query does not depend on: a count for a new person, or
    a count derived from a core node (GSM-NoOp style: relevant-looking, inconsequential)."""
    nodes = dict(graph.nodes)
    core = graph.ancestors()
    for i in range(width):
        if rng.random() < 0.5:
            kind = "given"
            inputs: tuple[str, ...] = ()
        else:
            kind = rng.choice(("add_c", "sub_c", "mul_c"))
            inputs = (rng.choice(core),)
        node = Node(f"d{i}", kind, inputs, rng.randrange(len(PHRASES[kind])), role="distractor")
        nodes[node.id] = node
    return Graph(nodes, graph.query)


def premise_order(graph: Graph, rng: random.Random) -> list[str]:
    """A random topological order: every premise follows the premises it refers to."""
    remaining = {n: set(node.inputs) for n, node in graph.nodes.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        pick = rng.choice(ready)
        order.append(pick)
        del remaining[pick]
        for deps in remaining.values():
            deps.discard(pick)
    return order


@dataclass(frozen=True)
class Values:
    consts: dict[str, int]  # the number each premise states (absent for add2/sub2)
    values: dict[str, int]  # each person's count


def apply(kind: str, inputs: list, const):
    """Forward value of one node. Works on ints, Fractions, or SymPy expressions."""
    if kind == "given":
        return const
    a = inputs[0]
    if kind == "add_c":
        return a + const
    if kind == "sub_c":
        return a - const
    if kind == "mul_c":
        return a * const
    if kind == "div_c":
        return a / const
    b = inputs[1]
    if kind == "add2":
        return a + b
    if kind == "sub2":
        return a - b
    raise ValueError(f"unknown kind {kind!r}")


def sample_values(
    graph: Graph, rng: random.Random, magnitude: str, tries: int = 300
) -> Values | None:
    """Draw stated numbers so that every count is a positive integer no larger than
    MAX_VALUE. Returns None if `tries` draws all fail."""
    lo, hi = MAGNITUDES[magnitude]
    for _ in range(tries):
        consts: dict[str, int] = {}
        values: dict[str, int] = {}
        for node in graph.nodes.values():
            ins = [values[i] for i in node.inputs]
            if node.kind in ("given", "add_c"):
                c = rng.randint(lo, hi)
            elif node.kind == "sub_c":
                if ins[0] - 1 < lo:
                    break
                c = rng.randint(lo, min(hi, ins[0] - 1))
            elif node.kind == "mul_c":
                c = rng.randint(2, 9)
            elif node.kind == "div_c":
                options = [d for d in DIVISORS if ins[0] % d == 0]
                if not options:
                    break
                c = rng.choice(options)
            else:
                c = None
            value = apply(node.kind, ins, c)
            if node.kind == "div_c":
                value = int(value)
            if not 1 <= value <= MAX_VALUE:
                break
            if c is not None:
                consts[node.id] = c
            values[node.id] = value
        else:
            return Values(consts, values)
    return None
