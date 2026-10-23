"""Manual label audit: draw a blind stratified sample, then score agreement.

    python -m premise_verify.audit sample --items eval_v0.jsonl.gz --n 400 --out audit/
    python -m premise_verify.audit score --dir audit/

`sample` writes `sheet.csv` (problems only, with empty columns for two annotators)
and `key.csv` (generator labels), so annotators label blind. Annotators write one of
solvable, underdetermined, inconsistent, unrealistic. `score` reports Cohen's kappa
between annotators and each annotator's agreement with the generator, and lists
every item where anyone disagrees, so the templates behind them can be fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from premise_verify.build import read
from premise_verify.metrics import cohen_kappa


def stratified_sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """About `n` items spread evenly over (variant kind, depth) strata."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for item in items:
        strata[(item["flaw"] or item["control"] or "original", item["depth"])].append(item)
    rng = random.Random(seed)
    per = max(1, n // len(strata))
    picked = [x for key in sorted(strata) for x in rng.sample(strata[key], min(per, len(strata[key])))]
    rng.shuffle(picked)
    return picked[:n]


def sample(items_path: Path, n: int, out: Path, seed: int) -> None:
    picked = stratified_sample(read(items_path), n, seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "annotator_a", "annotator_b", "notes"])
        w.writerows([x["id"], x["question"], "", "", ""] for x in picked)
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "answer", "flaw", "depth", "template_id"])
        w.writerows(
            [x["id"], x["label"], x["answer"], x["flaw"], x["depth"], x["template_id"]] for x in picked
        )
    print(f"wrote {len(picked)} items to {out / 'sheet.csv'} (labels in key.csv)")


def score(directory: Path) -> dict:
    with open(directory / "key.csv", encoding="utf-8") as f:
        key = {row["id"]: row for row in csv.DictReader(f)}
    with open(directory / "sheet.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["annotator_a"].strip() and r["annotator_b"].strip()]
    a = [r["annotator_a"].strip().lower() for r in rows]
    b = [r["annotator_b"].strip().lower() for r in rows]
    gold = [key[r["id"]]["label"] for r in rows]
    report = {
        "n": len(rows),
        "kappa_a_b": cohen_kappa(a, b),
        "kappa_a_generator": cohen_kappa(a, gold),
        "kappa_b_generator": cohen_kappa(b, gold),
        "disagreements": [
            {"id": r["id"], "template_id": key[r["id"]]["template_id"], "generator": g, "a": x, "b": y}
            for r, x, y, g in zip(rows, a, b, gold)
            if len({x, y, g}) > 1
        ],
    }
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manual label audit")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("score")
    k.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        sample(args.items, args.n, args.out, args.seed)
    else:
        score(args.dir)


if __name__ == "__main__":
    main()
