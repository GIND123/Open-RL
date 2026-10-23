"""Headline metrics, carried over from the workshop paper, with cluster-bootstrap
intervals.

A record is one scored response: the item's fields plus three 0/1 outcomes,
`flagged` (declined with UNSOLVABLE), `answered` (committed to a number), and
`correct` (committed to the right number). Rates are fractions; multiply by 100
for the paper's points.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

Record = dict
Stat = Callable[[list[Record]], float]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def decline_rate(records: list[Record], flawed: bool) -> float:
    return _mean(r["flagged"] for r in records if r["is_flawed"] == flawed)


def discrimination(records: list[Record]) -> float:
    """Decline rate on flawed items minus decline rate on solvable ones."""
    return decline_rate(records, True) - decline_rate(records, False)


def balanced_accuracy(records: list[Record]) -> float:
    """Balanced accuracy of the decline decision; 0.5 is chance."""
    return 0.5 * (decline_rate(records, True) + 1 - decline_rate(records, False))


def slope(records: list[Record], x: str = "depth", y: str = "flagged") -> float:
    """Least-squares slope of `y` on `x` (a linear probability model)."""
    xs = [r[x] for r in records]
    ys = [r[y] for r in records]
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx if sxx else float("nan")


def depth_gap(records: list[Record], x: str = "depth") -> float:
    """Slope of declining on `x` for flawed items minus the slope for solvable ones."""
    return slope([r for r in records if r["is_flawed"]], x) - slope(
        [r for r in records if not r["is_flawed"]], x
    )


def bootstrap(
    records: list[Record],
    stat: Stat,
    cluster: str = "parent_id",
    draws: int = 4000,
    seed: int = 6198,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """`stat` with a percentile interval from resampling whole clusters. Variants
    of one parent are correlated, so resampling rows would understate the interval."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r[cluster]].append(r)
    clusters = list(groups.values())
    rng = random.Random(seed)
    stats = sorted(
        stat([r for g in rng.choices(clusters, k=len(clusters)) for r in g]) for _ in range(draws)
    )
    tail = (1 - level) / 2
    return stat(records), stats[int(tail * draws)], stats[int((1 - tail) * draws) - 1]


def summarize(records: list[Record], cluster: str = "parent_id", draws: int = 4000) -> dict:
    out = {
        "n": len(records),
        "decline_flawed": decline_rate(records, True),
        "decline_solvable": decline_rate(records, False),
        "solvable_accuracy": _mean(r["correct"] for r in records if not r["is_flawed"]),
    }
    for name, stat in (
        ("discrimination", discrimination),
        ("balanced_accuracy", balanced_accuracy),
        ("depth_gap", depth_gap),
    ):
        est, lo, hi = bootstrap(records, stat, cluster, draws)
        out[name] = {"estimate": est, "ci": [lo, hi]}
    return out


def cohen_kappa(a: list, b: list) -> float:
    """Agreement between two annotators' labels, corrected for chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def records_from_traces(path: Path) -> list[Record]:
    """Records from a verifiers `traces.jsonl`. Each line is an episode carrying its
    traces, or a single trace; item fields come from the task data and outcomes
    from the metrics the eval environment records."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for trace in row.get("traces") or [row]:
                data = (trace.get("task") or row.get("task") or {}).get("data", {})
                metrics = trace.get("metrics") or {}
                if "flagged" not in metrics:
                    continue  # unscored or errored rollout
                records.append(
                    {
                        **data,
                        "flagged": metrics["flagged"],
                        "answered": metrics["answered"],
                        "correct": metrics["answer_correct"],
                    }
                )
    return records
