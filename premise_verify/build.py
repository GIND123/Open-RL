"""Build an item set: templates, then parent instances, then verified variants, then
padded text.

    python -m premise_verify.build --out data/eval_v0.jsonl.gz --seed 0

A template fixes the theme, the solution graph, the distractors, the premise order,
and the phrasings. An instance draws names and numbers. Every variant of an instance
is labelled by `verify` and kept only if the label is the one its edit should
produce. Following the workshop paper, a flawed variant is also dropped if the
parent's answer still appears in its text.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from premise_verify import perturb
from premise_verify.controls import pad, word_count
from premise_verify.dag import (
    MAGNITUDES,
    Graph,
    Values,
    add_distractors,
    build_tree,
    premise_order,
    sample_values,
)
from premise_verify.themes import (
    FRACTIONS,
    NAMES,
    PHRASES,
    THEMES,
    THEMES_BY_KEY,
    VAGUE_PHRASES,
    Theme,
    join_names,
    times_word,
)
from premise_verify.verify import Premise, verify

# (n_ops, depth). Chains give the paper-comparable chain-length curve; trees vary
# depth at fixed n_ops, which fixes the count of stated numbers too.
CHAIN_CELLS = tuple((d, d) for d in (2, 3, 4, 5, 6, 7, 8, 10))
TREE_CELLS = ((4, 3), (6, 3), (6, 4), (6, 5), (8, 4), (8, 6))


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    cells: tuple[tuple[int, int], ...] = CHAIN_CELLS + TREE_CELLS
    widths: tuple[int, ...] = (0, 2, 4)
    magnitudes: tuple[str, ...] = ("small", "medium", "large")
    templates_per_cell: int = 2
    instances_per_template: int = 3
    deletions_per_parent: int | None = 2  # None keeps every eligible deletion
    flaws: tuple[str, ...] = perturb.FLAWS
    controls: bool = True
    pad: str = "group"  # none | group (same n_ops and width) | global
    delete_query: bool = False


@dataclass(frozen=True)
class Template:
    id: str
    theme: Theme
    graph: Graph
    order: list[str]
    n_ops: int
    depth: int
    width: int
    magnitude: str


def render(p: Premise, theme: Theme, names: dict[str, str]) -> str:
    fields = {"e": names[p.node], "unit": theme.unit, "verb": theme.verb, "verb_pl": theme.verb_pl}
    if p.inputs:
        fields["a"] = names[p.inputs[0]]
    if len(p.inputs) > 1:
        fields["b"] = names[p.inputs[1]]
    if isinstance(p.vague, tuple):
        text = VAGUE_PHRASES[p.kind][1].format(lo=p.vague[0], hi=p.vague[1], **fields)
    elif p.vague is not None:
        text = VAGUE_PHRASES[p.kind][0].format(word=p.vague, **fields)
    else:
        fields["c"] = p.const
        if p.kind == "mul_c":
            fields["times"] = times_word(p.const)
        if p.kind == "div_c":
            fields["frac"] = FRACTIONS[p.const]
        text = PHRASES[p.kind][p.phrase].format(**fields)
    return text[0].upper() + text[1:]


def make_template(
    rng: random.Random, tid: str, n_ops: int, depth: int, width: int, magnitude: str
) -> Template | None:
    for _ in range(50):
        graph = add_distractors(build_tree(rng, depth, n_ops), rng, width)
        if sample_values(graph, random.Random(rng.random()), magnitude, tries=100):
            order = premise_order(graph, rng)
            return Template(tid, rng.choice(THEMES), graph, order, n_ops, depth, width, magnitude)
    return None


def parent_variants(
    t: Template, values: Values, rng: random.Random, config: BuildConfig
) -> list[perturb.Variant]:
    base = perturb.parent_premises(t.graph, values, t.order)
    answer = values.values[t.graph.query]
    magnitude = MAGNITUDES[t.magnitude]
    variants: list[perturb.Variant | None] = [perturb.Variant("orig", base, "solvable")]
    if "delete" in config.flaws:
        variants += perturb.deletions(
            base, t.graph, rng, config.deletions_per_parent, config.delete_query
        )
    if "contradict" in config.flaws:
        variants.append(perturb.contradiction(base, t.graph, values, rng, answer))
    if "unrealistic" in config.flaws:
        variants.append(perturb.unrealistic(base, t.graph, values, rng, magnitude))
    if "ambiguate" in config.flaws:
        variants.append(perturb.ambiguation(base, t.graph, values, rng, magnitude))
    if config.controls:
        variants.append(perturb.drop_distractor(base, rng))
    return [v for v in variants if v is not None]


def make_items(
    t: Template,
    instance: int,
    values: Values,
    names: dict[str, str],
    intro: str,
    variants: list[perturb.Variant],
    discarded: Counter,
) -> list[dict]:
    query = t.graph.query
    answer = values.values[query]
    parent_id = f"{t.id}-i{instance}"
    items = []
    for v in variants:
        verdict = verify(v.premises, query, hint=values.values)
        kind = v.flaw or v.control or "original"
        if verdict.label != v.expected or (v.expected == "solvable" and verdict.value != answer):
            discarded[f"{kind}:{verdict.label}"] += 1
            if v.tag == "orig":
                return []  # a parent that fails verification cannot anchor variants
            continue
        sentences = [intro, *(render(p, t.theme, names) for p in v.premises)]
        sentences.append(t.theme.question.format(name=names[query]))
        if v.flaw and re.search(rf"(?<!\d){answer}(?!\d)", " ".join(sentences)):
            discarded[f"{kind}:answer_in_text"] += 1
            continue
        items.append(
            {
                "id": f"{parent_id}-{v.tag}",
                "template_id": t.id,
                "parent_id": parent_id,
                "variant": v.tag,
                "theme": t.theme.key,
                "question": None,  # set after padding
                "answer": answer if v.expected == "solvable" else None,
                "parent_answer": answer,
                "label": verdict.label,
                "is_flawed": v.flaw is not None,
                "flaw": v.flaw,
                "flaw_mode": v.mode,
                "control": v.control,
                "n_ops": t.n_ops,
                "depth": t.depth,
                "width": t.width,
                "magnitude": t.magnitude,
                "n_numbers": sum(p.const is not None for p in v.premises),
                "n_premises": len(v.premises),
                "premise_distance": t.graph.distance(v.target) if v.flaw else None,
                "original_premise": render(v.original, t.theme, names) if v.original else None,
                "edited_premise": render(v.edited, t.theme, names) if v.edited else None,
                "verifier": verdict.reason,
                "_sentences": sentences,
            }
        )
    return items


def apply_padding(items: list[dict], mode: str, seed: int) -> None:
    groups: dict[object, list[dict]] = defaultdict(list)
    for item in items:
        key = {"none": item["id"], "group": (item["n_ops"], item["width"]), "global": 0}[mode]
        groups[key].append(item)
    for group in groups.values():
        target = max(word_count(" ".join(item["_sentences"])) for item in group)
        for item in group:
            rng = random.Random(f"{seed}:{item['id']}")
            sentences, added = pad(item.pop("_sentences"), THEMES_BY_KEY[item["theme"]], target, rng)
            item["question"] = " ".join(sentences)
            item["n_words"] = word_count(item["question"])
            item["n_padding"] = added


def build(config: BuildConfig = BuildConfig()) -> tuple[list[dict], dict]:
    rng = random.Random(config.seed)
    items: list[dict] = []
    discarded: Counter = Counter()
    n_templates = n_parents = 0
    for n_ops, depth in config.cells:
        for width in config.widths:
            for magnitude in config.magnitudes:
                for _ in range(config.templates_per_cell):
                    tid = f"s{config.seed}-t{n_templates:04d}"
                    n_templates += 1
                    t = make_template(rng, tid, n_ops, depth, width, magnitude)
                    if t is None:
                        discarded["template:infeasible"] += 1
                        continue
                    for instance in range(config.instances_per_template):
                        # Redraw when the answer equals a stated number, so the
                        # answer-in-text rule rarely has to discard a variant.
                        for _ in range(20):
                            values = sample_values(t.graph, rng, magnitude)
                            if values is None or values.values[t.graph.query] not in values.consts.values():
                                break
                        if values is None:
                            discarded["instance:no_values"] += 1
                            continue
                        people = rng.sample(NAMES, len(t.graph.nodes))
                        names = dict(zip(t.graph.nodes, people))
                        intro = t.theme.intro.format(names=join_names(rng.sample(people, len(people))))
                        variants = parent_variants(t, values, rng, config)
                        new = make_items(t, instance, values, names, intro, variants, discarded)
                        n_parents += bool(new)
                        items += new
    apply_padding(items, config.pad, config.seed)
    report = {
        "config": asdict(config),
        "templates": n_templates,
        "parents": n_parents,
        "items": len(items),
        "by_label": dict(Counter(item["label"] for item in items)),
        "by_flaw": dict(Counter(item["flaw"] or item["control"] or "original" for item in items)),
        "discarded": dict(discarded),
    }
    return items, report


def to_jsonl(items: list[dict]) -> bytes:
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items).encode()


def write(items: list[dict], path: Path) -> str:
    """Write JSONL (gzipped if the name ends in .gz, with a fixed header so the bytes
    are reproducible). Returns the SHA-256 of the uncompressed JSONL."""
    data = to_jsonl(items)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as f:
            f.write(data)
        data_out = buffer.getvalue()
    else:
        data_out = data
    path.write_bytes(data_out)
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad", choices=("none", "group", "global"), default="group")
    parser.add_argument("--templates-per-cell", type=int, default=2)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--deletions", default="2", help="deletions per parent, or 'all'")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args(argv)
    config = BuildConfig(
        seed=args.seed,
        pad=args.pad,
        templates_per_cell=args.templates_per_cell,
        instances_per_template=args.instances,
        deletions_per_parent=None if args.deletions == "all" else int(args.deletions),
        controls=not args.no_controls,
    )
    items, report = build(config)
    report["sha256"] = write(items, args.out)
    report_path = args.out.with_name(args.out.name.split(".")[0] + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "parents", "by_flaw", "discarded")}, indent=2))
    print(f"wrote {args.out} and {report_path}")


if __name__ == "__main__":
    main()
