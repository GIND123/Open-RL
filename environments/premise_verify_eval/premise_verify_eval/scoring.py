"""Parse a response into a decision and score it.

Pure Python with no verifiers import, so the training environment and offline
analysis can reuse it. The structured `\\boxed{}` decision is the primary signal; a
model that flags a problem only in prose is caught by the separate judge pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

INSTRUCTIONS = {
    # Names abstention as an option, as in the workshop paper's scoring prompt.
    "permit": (
        "Solve the problem. End your response with the final answer in \\boxed{}. "
        "If the problem cannot be answered as stated, end with "
        "\\boxed{UNSOLVABLE: <brief reason>} instead."
    ),
    # Benchmark-style answer forcing: abstention is never mentioned.
    "force": "Solve the problem. End your response with the final answer in \\boxed{}.",
}

SCHEMES = ("binary", "mix", "threshold", "mix+threshold")

_BOX_MARKERS = ("\\boxed{", "\\fbox{")
_DECLINE = re.compile(
    r"^\W*(unsolvable|unanswerable|not solvable|cannot be (determined|answered|solved)"
    r"|insufficient|undetermined|indeterminate|ill-posed|no unique answer)",
    re.IGNORECASE,
)
_TEXT = re.compile(r"\\(?:text|textbf|mathrm)\{([^{}]*)\}")
_FRAC = re.compile(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True)
class Decision:
    kind: str  # "answered" | "flagged" | "none"
    value: Fraction | None = None
    reason: str = ""


def render_prompt(question: str, mode: str = "permit") -> str:
    return f"{question}\n\n{INSTRUCTIONS[mode]}"


def strip_reasoning(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text  # unclosed reasoning: no final answer


def last_boxed(text: str) -> str | None:
    starts = [(text.rfind(m), m) for m in _BOX_MARKERS]
    start, marker = max(starts)
    if start == -1:
        return None
    depth, begin = 1, start + len(marker)
    for i in range(begin, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[begin:i]
    return None


def parse_number(text: str) -> Fraction | None:
    text = _FRAC.sub(r"\1/\2", text)
    text = _THOUSANDS.sub("", text)
    numbers = _NUMBER.findall(text)
    return Fraction(numbers[0]) if len(numbers) == 1 else None


def parse_decision(response: str | None) -> Decision:
    boxed = last_boxed(strip_reasoning(response or ""))
    if boxed is None:
        return Decision("none")
    inner = _TEXT.sub(r"\1", boxed).strip()
    if _DECLINE.match(inner):
        reason = inner.split(":", 1)[1].strip() if ":" in inner else ""
        return Decision("flagged", reason=reason)
    value = parse_number(inner)
    return Decision("answered", value) if value is not None else Decision("none")


def is_correct(decision: Decision, answer: int | None) -> bool:
    return decision.kind == "answered" and answer is not None and decision.value == answer


def scheme_reward(
    scheme: str, decision: Decision, answer: int | None, is_flawed: bool, t: float = 0.75
) -> float:
    """Training reward for the four RQ4 conditions.

    `binary` and `mix` pay 1 for a right decision and 0 otherwise; they differ only
    in whether flawed items are in the training data. The threshold schemes follow
    Kalai et al.: a right decision earns 1, declining a solvable item earns 0, and any
    other outcome costs t/(1-t). A committed number on a flawed item is a wrong
    answer, and so is a response with no decision, so explicit abstention always
    beats both.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    wrong = -t / (1 - t) if scheme.endswith("threshold") else 0.0
    if is_flawed:
        return 1.0 if decision.kind == "flagged" else wrong
    if decision.kind == "flagged":
        return 0.0
    return 1.0 if is_correct(decision, answer) else wrong
