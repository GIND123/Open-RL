import random
import re
from collections import defaultdict

import pytest

from premise_verify.build import CHAIN_CELLS, TREE_CELLS, BuildConfig, build, to_jsonl
from premise_verify.dag import build_tree, max_ops
from premise_verify.themes import PADDING, THEMES

SMALL = BuildConfig(
    seed=3,
    cells=((2, 2), (4, 4), (4, 3), (6, 3)),
    widths=(0, 2),
    magnitudes=("small", "large"),
    templates_per_cell=1,
    instances_per_template=2,
)


@pytest.fixture(scope="module")
def built():
    return build(SMALL)


@pytest.mark.parametrize("n_ops,depth", CHAIN_CELLS + TREE_CELLS)
def test_tree_has_exact_depth_and_ops(n_ops, depth):
    rng = random.Random(0)
    for _ in range(20):
        g = build_tree(rng, depth, n_ops)
        assert g.depth() == depth and g.n_ops() == n_ops


def test_infeasible_cell_is_rejected():
    with pytest.raises(ValueError):
        build_tree(random.Random(0), 2, max_ops(2) + 1)


def test_build_is_deterministic(built):
    items, _ = built
    again, _ = build(SMALL)
    assert to_jsonl(items) == to_jsonl(again)


def test_every_variant_kind_survives(built):
    _, report = built
    assert set(report["by_flaw"]) == {
        "original", "delete", "contradict", "unrealistic", "ambiguate", "drop_distractor"
    }


def test_parents_anchor_their_variants(built):
    items, _ = built
    by_parent = defaultdict(list)
    for item in items:
        by_parent[item["parent_id"]].append(item)
    for variants in by_parent.values():
        (original,) = [v for v in variants if v["variant"] == "orig"]
        assert original["label"] == "solvable" and isinstance(original["answer"], int)
        for v in variants:
            assert v["parent_answer"] == original["answer"]
            if v["is_flawed"]:
                assert v["answer"] is None and v["label"] != "solvable"
            else:
                assert v["answer"] == original["answer"]


def test_flawed_text_never_contains_parent_answer(built):
    items, _ = built
    for item in items:
        if item["is_flawed"]:
            assert not re.search(rf"(?<!\d){item['parent_answer']}(?!\d)", item["question"])


def test_deletions_are_consumed_premises(built):
    items, _ = built
    for item in items:
        if item["flaw"] == "delete":
            assert item["label"] == "underdetermined"
            assert 1 <= item["premise_distance"] <= item["depth"]
            assert item["original_premise"] not in item["question"]


def test_stated_numbers_fixed_by_ops(built):
    items, _ = built
    for item in items:
        if item["variant"] == "orig" and item["width"] == 0:
            assert item["n_numbers"] == item["n_ops"] + 1


def test_group_padding_holds_length(built):
    items, _ = built
    groups = defaultdict(list)
    for item in items:
        groups[(item["n_ops"], item["width"])].append(item["n_words"])
    for lengths in groups.values():
        assert max(lengths) - min(lengths) <= 8


def test_padding_carries_no_quantities():
    banned = re.compile(
        r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen|half|twice|pair|both|"
        r"couple|few|several|many|some|all|each|every|none|first|second|third|single)\b",
        re.IGNORECASE,
    )
    for sentence in PADDING + tuple(s for t in THEMES for s in t.padding):
        assert not banned.search(sentence), sentence
