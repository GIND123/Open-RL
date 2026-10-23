"""Premise-verification evaluation: does the model notice that a math word problem
cannot be answered as stated?

Each task is one generated item, either solvable or carrying one flaw (a deleted
premise, a contradiction, an impossible count, or a vague number). The reward is 1
for the right decision: the correct number on a solvable item, or
`\\boxed{UNSOLVABLE: ...}` on a flawed one. Metrics record the decision itself,
Kalai-style net scores, and response length, so discrimination, balanced accuracy
and depth slopes can be computed offline with `premise_verify.metrics`.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.harnesses.null.harness import NullHarness

from premise_verify_eval.scoring import is_correct, parse_decision, render_prompt, scheme_reward

DEFAULT_DATA = "eval_v0.jsonl.gz"
NET_THRESHOLDS = (0.5, 0.75, 0.9)


class PremiseVerifyData(vf.TaskData):
    answer: int | None = None
    label: str = "solvable"
    is_flawed: bool = False
    flaw: str | None = None
    flaw_mode: str | None = None
    control: str | None = None
    template_id: str = ""
    parent_id: str = ""
    variant: str = ""
    theme: str = ""
    n_ops: int = 0
    depth: int = 0
    width: int = 0
    magnitude: str = ""
    n_numbers: int = 0
    n_words: int = 0
    premise_distance: int | None = None
    prompt_mode: str = "permit"


class PremiseVerifyTask(vf.Task[PremiseVerifyData]):
    @property
    def key(self) -> str:
        return f"{self.data.id}:{self.data.prompt_mode}"

    @vf.reward(weight=1.0)
    async def correct_decision(self, trace: vf.Trace) -> float:
        decision = parse_decision(trace.last_reply)
        if self.data.is_flawed:
            return float(decision.kind == "flagged")
        return float(is_correct(decision, self.data.answer))

    @vf.metric
    async def decision(self, trace: vf.Trace) -> dict[str, float]:
        d = parse_decision(trace.last_reply)
        out = {
            "flagged": float(d.kind == "flagged"),
            "answered": float(d.kind == "answered"),
            "no_decision": float(d.kind == "none"),
            "answer_correct": float(is_correct(d, self.data.answer)),
            "truncated": float(trace.is_truncated),
            "completion_tokens": float(
                sum(call.usage.completion_tokens for call in trace.calls if call.usage)
            ),
        }
        for t in NET_THRESHOLDS:
            out[f"net_t{round(t * 100)}"] = scheme_reward(
                "threshold", d, self.data.answer, self.data.is_flawed, t
            )
        return out


class PremiseVerifyConfig(vf.TasksetConfig):
    data_path: Path | None = None
    """JSONL(.gz) of generated items; defaults to the bundled eval set."""
    prompt_mode: Literal["permit", "force"] = "permit"
    """`permit` names abstention as an option; `force` is benchmark-style."""
    flaws: list[str] | None = None
    """Keep only these flaw types (use "none" for solvable items); None keeps all."""


class PremiseVerifyTaskset(vf.Taskset[PremiseVerifyTask, PremiseVerifyConfig]):
    def load(self):
        fields = PremiseVerifyData.model_fields
        for row in _read(self.config.data_path):
            if self.config.flaws is not None and (row["flaw"] or "none") not in self.config.flaws:
                continue
            data = PremiseVerifyData(
                id=row["id"],
                prompt=render_prompt(row["question"], self.config.prompt_mode),
                prompt_mode=self.config.prompt_mode,
                **{k: v for k, v in row.items() if k in fields and k not in ("id", "prompt")},
            )
            yield PremiseVerifyTask(data, self.config.task)


class PremiseVerifyHarness(NullHarness):
    """Plain chat, no tools: a code-running harness would hand the model a calculator
    and change the task."""


def _read(path: Path | None):
    source = path or files("premise_verify_eval") / "data" / DEFAULT_DATA
    with source.open("rb") as raw:
        stream = gzip.open(raw, "rt", encoding="utf-8") if str(source).endswith(".gz") else raw
        for line in stream:
            if line.strip():
                yield json.loads(line)
