"""Tables for the Phase A pilot from two vf-eval traces files, one per prompt mode.

    python scripts/summarize_pilot.py \
        --permit results/phase_a_pilot/traces/pilot-permit.traces.jsonl.gz \
        --force results/phase_a_pilot/traces/pilot-force.traces.jsonl.gz \
        --out results/phase_a_pilot/tables.json

Prints markdown tables and writes the same numbers as JSON. Headline intervals use the
paper's cluster bootstrap (parent problems, 4000 draws, seed 6198).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from premise_verify.metrics import bootstrap, summarize

FLAWS = ("solvable", "delete", "ambiguate", "unrealistic", "contradict")
# Crude lexical screen for a flaw named in prose outside the final box. It is an
# indicator for the judge pass to confirm, not a measurement.
PROSE_FLAG = re.compile(
    r"not enough information|insufficient|cannot be (determined|answered|solved)"
    r"|can't be determined|missing (information|data|value)|not (given|provided|stated)"
    r"|inconsisten|contradict|impossible|not possible|unsolvable|ambiguous",
    re.IGNORECASE,
)


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for trace in row.get("traces") or [row]:
                metrics = trace.get("metrics") or {}
                if "flagged" not in metrics:
                    continue
                data = trace["task"]["data"]
                reply = str(trace["nodes"][-1]["message"].get("content") or "")
                final = reply.rsplit("</think>", 1)[1] if "</think>" in reply else ""
                out.append({
                    **data,
                    "flaw": data.get("flaw") or "solvable",
                    "flagged": metrics["flagged"],
                    "answered": metrics["answered"],
                    "no_decision": metrics["no_decision"],
                    "correct": metrics["answer_correct"],
                    "truncated": metrics["truncated"],
                    "tokens": metrics["completion_tokens"],
                    "net_t75": metrics["net_t75"],
                    "prose_flag": float(bool(PROSE_FLAG.search(final))),
                })
    return out


def rate(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def headline(rows):
    s = summarize(rows)
    s["truncated_rate"] = rate(rows, "truncated")
    s["no_decision_rate"] = rate(rows, "no_decision")
    s["median_tokens"] = statistics.median(r["tokens"] for r in rows)
    s["mean_net_t75"] = rate(rows, "net_t75")
    return s


def by_flaw(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["flaw"]].append(r)
    out = {}
    for flaw in FLAWS:
        g = groups.get(flaw, [])
        if not g:
            continue
        _, lo, hi = bootstrap(g, lambda x: rate(x, "flagged"))
        out[flaw] = {
            "n": len(g), "declined": rate(g, "flagged"), "declined_ci": [lo, hi],
            "answered": rate(g, "answered"), "no_decision": rate(g, "no_decision"),
            "prose_flag": rate(g, "prose_flag"),
            "median_tokens": statistics.median(r["tokens"] for r in g),
        }
    return out


def by_field(rows, field, flaws=("delete",)):
    groups = defaultdict(list)
    for r in rows:
        if r["flaw"] in flaws and r.get(field) is not None:
            groups[r[field]].append(r)
    return {k: {"n": len(g), "declined": rate(g, "flagged")} for k, g in sorted(groups.items())}


def contradict_modes(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["flaw"] == "contradict":
            groups[r.get("flaw_mode") or "?"].append(r)
    return {k: {"n": len(g), "declined": rate(g, "flagged")} for k, g in sorted(groups.items())}


def paired(permit, force):
    """Flawed items declined under permit: what force mode does with them."""
    force_by_id = {r["id"]: r for r in force}
    both = [(p, force_by_id[p["id"]]) for p in permit if p["id"] in force_by_id]
    caught = [(p, f) for p, f in both if p["is_flawed"] and p["flagged"]]
    return {
        "items_in_both_modes": len(both),
        "flawed_declined_under_permit": len(caught),
        "of_those_answered_under_force": rate([f for _, f in caught], "answered"),
        "of_those_prose_flag_under_force": rate([f for _, f in caught], "prose_flag"),
    }


def pct(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permit", type=Path, required=True)
    ap.add_argument("--force", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = {"permit": load(args.permit), "force": load(args.force)}

    tables = {mode: {"headline": headline(rows), "by_flaw": by_flaw(rows)}
              for mode, rows in runs.items()}
    tables["permit"]["delete_by_depth"] = by_field(runs["permit"], "depth")
    tables["permit"]["delete_by_premise_distance"] = by_field(runs["permit"], "premise_distance")
    tables["permit"]["contradict_modes"] = contradict_modes(runs["permit"])
    tables["paired"] = paired(runs["permit"], runs["force"])

    print("## Headline\n")
    print("| | permit | force |\n|---|---|---|")
    hp, hf = tables["permit"]["headline"], tables["force"]["headline"]
    for label, key in (("Decline rate, flawed", "decline_flawed"),
                       ("Decline rate, solvable", "decline_solvable"),
                       ("Accuracy, solvable", "solvable_accuracy")):
        print(f"| {label} (%) | {pct(hp[key])} | {pct(hf[key])} |")
    for label, key in (("Discrimination (pts)", "discrimination"),
                       ("Balanced accuracy", "balanced_accuracy"),
                       ("Depth gap (per step)", "depth_gap")):
        cells = []
        for h in (hp, hf):
            e, (lo, hi) = h[key]["estimate"], h[key]["ci"]
            scale = 100 if key == "discrimination" else 1
            cells.append(f"{scale * e:+.3g} [{scale * lo:+.3g}, {scale * hi:+.3g}]")
        print(f"| {label} | {cells[0]} | {cells[1]} |")
    for label, key in (("Truncated (%)", "truncated_rate"), ("No boxed decision (%)", "no_decision_rate")):
        print(f"| {label} | {pct(hp[key])} | {pct(hf[key])} |")
    print(f"| Median completion tokens | {hp['median_tokens']:.0f} | {hf['median_tokens']:.0f} |")
    print(f"| Mean net score, t = 0.75 | {hp['mean_net_t75']:+.3f} | {hf['mean_net_t75']:+.3f} |")

    for mode in ("permit", "force"):
        print(f"\n## By flaw type, {mode}\n")
        print("| Flaw | n | Declined (%) [95% CI] | Answered (%) | No decision (%) | Prose flag (%) | Median tokens |")
        print("|---|---|---|---|---|---|---|")
        for flaw, t in tables[mode]["by_flaw"].items():
            lo, hi = t["declined_ci"]
            print(f"| {flaw} | {t['n']} | {pct(t['declined'])} [{pct(lo)}, {pct(hi)}] | {pct(t['answered'])} "
                  f"| {pct(t['no_decision'])} | {pct(t['prose_flag'])} | {t['median_tokens']:.0f} |")

    for name, title in (("delete_by_depth", "Deleted premise, decline by depth (permit)"),
                        ("delete_by_premise_distance", "Deleted premise, decline by premise distance (permit)"),
                        ("contradict_modes", "Contradiction, decline by mode (permit)")):
        print(f"\n## {title}\n")
        print("| Value | n | Declined (%) |\n|---|---|---|")
        for k, t in tables["permit"][name].items():
            print(f"| {k} | {t['n']} | {pct(t['declined'])} |")

    p = tables["paired"]
    print("\n## Paired items\n")
    print(f"- Items in both modes: {p['items_in_both_modes']}")
    print(f"- Flawed items declined under permit: {p['flawed_declined_under_permit']}")
    print(f"- Of those, answered with a number under force: {pct(p['of_those_answered_under_force'])}%")
    print(f"- Of those, flaw named in prose under force (lexical screen): {pct(p['of_those_prose_flag_under_force'])}%")

    if args.out:
        args.out.write_text(json.dumps(tables, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
