from fractions import Fraction

import pytest

scoring = pytest.importorskip("premise_verify_eval.scoring")
Decision, parse_decision, scheme_reward = (
    scoring.Decision,
    scoring.parse_decision,
    scoring.scheme_reward,
)


@pytest.mark.parametrize(
    "response,kind,value",
    [
        ("So the answer is \\boxed{43}.", "answered", Fraction(43)),
        ("\\boxed{1,234}", "answered", Fraction(1234)),
        ("\\boxed{\\frac{3}{4}}", "answered", Fraction(3, 4)),
        ("\\boxed{-2.5}", "answered", Fraction(-5, 2)),
        ("\\boxed{43 \\text{ apples}}", "answered", Fraction(43)),
        ("first \\boxed{7} then \\boxed{9}", "answered", Fraction(9)),
        ("\\boxed{UNSOLVABLE: Ben's count is never given}", "flagged", None),
        ("\\boxed{\\text{Unsolvable}}", "flagged", None),
        ("\\boxed{cannot be determined}", "flagged", None),
        ("<think>maybe \\boxed{3}</think>The answer is \\boxed{5}", "answered", Fraction(5)),
        ("<think>still thinking \\boxed{3}", "none", None),
        ("The answer is 43.", "none", None),
        ("\\boxed{3 or 4}", "none", None),
    ],
)
def test_parse_decision(response, kind, value):
    d = parse_decision(response)
    assert d.kind == kind and d.value == value


def test_flag_reason_is_kept():
    assert parse_decision("\\boxed{UNSOLVABLE: missing {Ben}}").reason == "missing {Ben}"


RIGHT, WRONG = Decision("answered", Fraction(5)), Decision("answered", Fraction(6))
FLAG, NONE = Decision("flagged"), Decision("none")


@pytest.mark.parametrize("scheme", ["binary", "mix"])
def test_binary_schemes(scheme):
    assert scheme_reward(scheme, RIGHT, 5, False) == 1.0
    assert scheme_reward(scheme, WRONG, 5, False) == 0.0
    assert scheme_reward(scheme, FLAG, 5, False) == 0.0
    assert scheme_reward(scheme, FLAG, None, True) == 1.0
    assert scheme_reward(scheme, WRONG, None, True) == 0.0


@pytest.mark.parametrize("scheme", ["threshold", "mix+threshold"])
def test_threshold_schemes(scheme):
    penalty = -0.75 / 0.25
    assert scheme_reward(scheme, RIGHT, 5, False) == 1.0
    assert scheme_reward(scheme, FLAG, 5, False) == 0.0
    assert scheme_reward(scheme, WRONG, 5, False) == penalty
    assert scheme_reward(scheme, NONE, 5, False) == penalty
    assert scheme_reward(scheme, FLAG, None, True) == 1.0
    assert scheme_reward(scheme, WRONG, None, True) == penalty


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        scheme_reward("bogus", RIGHT, 5, False)
