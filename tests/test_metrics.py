import json
import math

import pytest

from premise_verify.metrics import (
    balanced_accuracy,
    bootstrap,
    cohen_kappa,
    depth_gap,
    discrimination,
    records_from_traces,
)


def rec(parent, flawed, flagged, depth=2):
    return {"parent_id": parent, "is_flawed": flawed, "flagged": flagged, "depth": depth, "correct": 0}


RECORDS = [
    rec("p0", False, 0), rec("p0", True, 1), rec("p0", True, 1),
    rec("p1", False, 0), rec("p1", True, 0), rec("p1", True, 1),
    rec("p2", False, 1), rec("p2", True, 1), rec("p2", True, 0),
]


def test_discrimination_and_balanced_accuracy():
    # Flawed decline 4/6, solvable decline 1/3.
    assert discrimination(RECORDS) == pytest.approx(4 / 6 - 1 / 3)
    assert balanced_accuracy(RECORDS) == pytest.approx(0.5 * (4 / 6 + 2 / 3))


def test_depth_gap_matches_hand_computation():
    records = [rec("p", True, f, d) for d, f in ((1, 1), (1, 1), (3, 0), (3, 1))]
    records += [rec("p", False, 0, d) for d in (1, 3)]
    assert depth_gap(records) == pytest.approx(-0.25)


def test_bootstrap_is_reproducible_and_brackets_estimate():
    first = bootstrap(RECORDS, discrimination, draws=500)
    assert first == bootstrap(RECORDS, discrimination, draws=500)
    est, lo, hi = first
    assert lo <= est <= hi


def test_cohen_kappa():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert math.isclose(cohen_kappa(list("aabb"), list("abab")), 0.0)


def test_records_from_episode_and_trace_lines(tmp_path):
    data = {"id": "x", "parent_id": "p", "is_flawed": True, "depth": 3}
    metrics = {"flagged": 1.0, "answered": 0.0, "answer_correct": 0.0}
    episode = {"task": {"data": data}, "traces": [{"metrics": metrics}]}
    trace = {"task": {"data": data}, "metrics": metrics}
    unscored = {"task": {"data": data}, "metrics": {}}
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in (episode, trace, unscored)))
    records = records_from_traces(path)
    assert len(records) == 2
    assert records[0]["flagged"] == 1.0 and records[0]["depth"] == 3
