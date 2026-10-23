"""Environment tests. verifiers imports `fcntl`, so these run on Linux/WSL only."""

import asyncio
import gzip
import json
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")
env = pytest.importorskip("premise_verify_eval.taskset")

from premise_verify.build import BuildConfig, build, write  # noqa: E402

CONFIG = BuildConfig(seed=5, cells=((3, 3),), widths=(2,), magnitudes=("small",),
                     templates_per_cell=1, instances_per_template=1)


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    items, _ = build(CONFIG)
    path = tmp_path_factory.mktemp("data") / "items.jsonl.gz"
    write(items, path)
    return items, path


def tasks(path, **config):
    taskset = env.PremiseVerifyTaskset(env.PremiseVerifyConfig(data_path=path, **config))
    return list(taskset)


def fake_trace(reply):
    return SimpleNamespace(last_reply=reply, is_truncated=False, calls=[])


def test_loads_every_item_with_prompt(items):
    rows, path = items
    loaded = tasks(path)
    assert [t.data.id for t in loaded] == [r["id"] for r in rows]
    assert all("UNSOLVABLE" in t.data.prompt_text for t in loaded)
    assert len({t.key for t in loaded}) == len(loaded)


def test_bundled_data_loads():
    loaded = tasks(None)
    assert loaded and len({t.key for t in loaded}) == len(loaded)
    assert {t.data.flaw for t in loaded} == {None, "delete", "contradict", "unrealistic", "ambiguate"}


def test_force_mode_hides_abstention(items):
    _, path = items
    assert not any("UNSOLVABLE" in t.data.prompt_text for t in tasks(path, prompt_mode="force"))


def test_flaw_filter(items):
    _, path = items
    kept = tasks(path, flaws=["none", "delete"])
    assert kept and all(t.data.flaw in (None, "delete") for t in kept)


def test_rewards_and_metrics(items):
    _, path = items
    loaded = tasks(path)
    solvable = next(t for t in loaded if not t.data.is_flawed)
    flawed = next(t for t in loaded if t.data.is_flawed)
    right = fake_trace(f"\\boxed{{{solvable.data.answer}}}")
    flag = fake_trace("\\boxed{UNSOLVABLE: missing count}")
    run = asyncio.run
    assert run(solvable.correct_decision(right)) == 1.0
    assert run(solvable.correct_decision(flag)) == 0.0
    assert run(flawed.correct_decision(flag)) == 1.0
    assert run(flawed.correct_decision(right)) == 0.0
    m = run(flawed.decision(flag))
    assert m["flagged"] == 1.0 and m["net_t75"] == 1.0
    assert run(solvable.decision(fake_trace("no box")))["no_decision"] == 1.0


def test_bundled_harness_is_plain_chat():
    from verifiers.v1.utils.loaders import default_harness_id, harness_config_type

    assert default_harness_id("premise_verify_eval") == "premise_verify_eval"
    assert harness_config_type("premise_verify_eval").__name__ == "NullHarnessConfig"
