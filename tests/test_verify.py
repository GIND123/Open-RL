from fractions import Fraction

from premise_verify.verify import Premise, verify

# Ana has 12; Ben has 5 more than Ana; Cy has 3 times as many as Ben. Cy = 51.
CHAIN = (
    Premise("ana", "given", const=12),
    Premise("ben", "add_c", ("ana",), 5),
    Premise("cy", "mul_c", ("ben",), 3),
)
HINT = {"ana": 12, "ben": 17, "cy": 51}


def test_solvable_chain():
    v = verify(CHAIN, "cy")
    assert v.label == "solvable" and v.value == 51


def test_deleting_a_consumed_premise_underdetermines():
    for i in range(3):
        premises = CHAIN[:i] + CHAIN[i + 1 :]
        assert verify(premises, "cy", hint=HINT).label == "underdetermined"


def test_restated_given_is_inconsistent():
    assert verify(CHAIN + (Premise("ana", "given", const=13, role="flaw"),), "cy").label == "inconsistent"


def test_wrong_derived_value_is_inconsistent():
    assert verify(CHAIN + (Premise("ben", "given", const=18, role="flaw"),), "cy").label == "inconsistent"


def test_negative_intermediate_is_unrealistic():
    premises = (CHAIN[0], Premise("ben", "sub_c", ("ana",), 20), CHAIN[2])
    assert verify(premises, "cy").label == "unrealistic"


def test_fractional_count_is_unrealistic():
    premises = (CHAIN[0], Premise("ben", "div_c", ("ana",), 5), CHAIN[2])
    assert verify(premises, "cy").label == "unrealistic"


def test_vague_word_underdetermines():
    premises = (CHAIN[0], Premise("ben", "add_c", ("ana",), vague="a few"), CHAIN[2])
    assert verify(premises, "cy", hint=HINT).label == "underdetermined"


def test_range_underdetermines():
    premises = (Premise("ana", "given", vague=(10, 14)),) + CHAIN[1:]
    assert verify(premises, "cy").label == "underdetermined"


def test_range_resolved_by_integrality_is_solvable():
    # Ana has between 5 and 7; Ben has a third as many: only Ana = 6 keeps Ben whole.
    premises = (Premise("ana", "given", vague=(5, 7)), Premise("ben", "div_c", ("ana",), 3))
    v = verify(premises, "ben")
    assert v.label == "solvable" and v.value == Fraction(2)


def test_irrelevant_vagueness_leaves_query_determined():
    premises = CHAIN + (Premise("dee", "add_c", ("ana",), vague="several", role="distractor"),)
    assert verify(premises, "cy").label == "solvable"


def test_two_input_operations():
    premises = (
        Premise("ana", "given", const=9),
        Premise("ben", "given", const=4),
        Premise("cy", "sub2", ("ana", "ben")),
        Premise("dee", "add2", ("cy", "ben")),
    )
    v = verify(premises, "dee")
    assert v.label == "solvable" and v.value == 9
