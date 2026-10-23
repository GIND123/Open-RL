"""Hand-written surface templates: themes, sentence phrasings, names, and padding.

Every premise sentence states exactly one fact about one named person, so deleting
a sentence removes exactly one equation. Nothing here is derived from GSM8K.
"""

from __future__ import annotations

from dataclasses import dataclass

NAMES = (
    "Ana", "Ben", "Chloe", "Dev", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jada",
    "Kofi", "Lena", "Mateo", "Nia", "Omar", "Priya", "Quinn", "Rosa", "Sami", "Tara",
    "Umar", "Vera", "Wes", "Xia", "Yusuf", "Zoe", "Arjun", "Bea", "Carlos", "Dina",
    "Emre", "Fiona", "Goran", "Hiro", "Ines", "Jonah", "Kira", "Luca", "Mira", "Nate",
    "Olu", "Pilar", "Raj", "Sofia", "Theo", "Uma", "Vik", "Wren",
)


@dataclass(frozen=True)
class Theme:
    key: str
    unit: str          # plural noun; every stated count is at least 2
    verb: str          # third person singular, e.g. "has" or "picked"
    verb_pl: str       # plural agreement, e.g. "have" or "picked"
    question: str      # format string with {name}
    intro: str         # format string with {names}
    padding: tuple[str, ...]


THEMES: tuple[Theme, ...] = (
    Theme(
        "marbles", "marbles", "has", "have",
        "How many marbles does {name} have?",
        "{names} are comparing their marble collections.",
        (
            "The marbles are kept in small cloth bags.",
            "The marbles have swirls of color inside them.",
            "They spread a blanket on the floor to sort them.",
            "The glass marbles catch the light from the window.",
            "Trading marbles has become a popular game at their school.",
        ),
    ),
    Theme(
        "apples", "apples", "picked", "picked",
        "How many apples did {name} pick?",
        "{names} spent the morning picking apples at an orchard.",
        (
            "The orchard sits at the edge of a quiet valley.",
            "The trees were heavy with ripe fruit.",
            "A farmer lent them wooden baskets for the day.",
            "The air smelled of fresh grass and fallen leaves.",
            "They took a break under the shade of an old tree.",
        ),
    ),
    Theme(
        "stamps", "stamps", "collected", "collected",
        "How many stamps did {name} collect?",
        "{names} collected stamps for a school exhibit.",
        (
            "The exhibit will be held in the school library.",
            "Old letters from relatives turned out to be a good source.",
            "They used tweezers to handle the delicate paper.",
            "The stamps showed birds, ships, and famous buildings.",
            "Their teacher offered to help label the display.",
        ),
    ),
    Theme(
        "cookies", "cookies", "baked", "baked",
        "How many cookies did {name} bake?",
        "{names} baked cookies for a neighborhood fundraiser.",
        (
            "The kitchen smelled of butter and cinnamon.",
            "The fundraiser will support the local animal shelter.",
            "They borrowed extra baking trays from a neighbor.",
            "The cookies cooled on racks near the open window.",
            "Flour somehow ended up on the kitchen floor.",
        ),
    ),
    Theme(
        "pages", "pages", "read", "read",
        "How many pages did {name} read?",
        "{names} kept a log of the pages they read during a library challenge.",
        (
            "The library challenge runs through the summer.",
            "Mystery novels were the most popular choice.",
            "The librarian posted a chart on the wall by the entrance.",
            "Reading by the window was the favorite spot.",
            "The challenge ends with a party in the reading room.",
        ),
    ),
    Theme(
        "seashells", "seashells", "found", "found",
        "How many seashells did {name} find?",
        "{names} searched the beach for seashells after a storm.",
        (
            "The tide had gone out by late morning.",
            "Seagulls circled above the rocks.",
            "The sand was still damp from the night before.",
            "They carried their finds in plastic buckets.",
            "A lighthouse stood at the far end of the beach.",
        ),
    ),
    Theme(
        "coins", "coins", "saved", "saved",
        "How many coins did {name} save?",
        "{names} saved coins in jars for a class trip.",
        (
            "The jars sat on a shelf in the classroom.",
            "The class trip is planned for the spring.",
            "Their teacher painted labels on the jars.",
            "The coins rattled whenever a jar was moved.",
            "The trip will take them to a science museum.",
        ),
    ),
    Theme(
        "saplings", "saplings", "planted", "planted",
        "How many saplings did {name} plant?",
        "{names} planted saplings at a community park.",
        (
            "The park had lost trees in a storm the year before.",
            "Volunteers handed out shovels and gloves.",
            "The soil was soft after the recent rain.",
            "A local nursery donated the young trees.",
            "The new trees will shade the walking path someday.",
        ),
    ),
)

THEMES_BY_KEY = {theme.key: theme for theme in THEMES}

# Numberless filler used to hold token length fixed. No digits, no number words,
# and no quantifiers, so padding never adds or blurs a quantity.
PADDING = (
    "The weather was mild and pleasant.",
    "It was a bright afternoon.",
    "A light breeze made the day comfortable.",
    "The sky was clear and blue.",
    "They chatted happily while they worked.",
    "They took their time.",
    "The project had been planned for weeks.",
    "Their parents were proud of the effort.",
    "They had agreed to keep careful records.",
    "Music played softly in the background.",
    "It turned out to be a cheerful occasion.",
    "They took turns telling stories.",
    "The morning started with a quick breakfast.",
    "A neighbor stopped by to say hello.",
    "They wore their favorite jackets.",
    "The idea came from a book they had read.",
    "A friendly dog followed them around.",
    "They laughed at an old joke.",
    "Lunch was sandwiches and lemonade.",
    "The plan worked out better than expected.",
    "They wrote their names on the tally sheet.",
    "The air was fresh and cool.",
    "They promised to do it again next year.",
    "Someone brought a camera to take pictures.",
    "It was quieter than usual that day.",
    "They took a short walk afterward.",
    "The day ended with a warm sunset.",
    "Their teacher had suggested the idea.",
    "They kept the tally sheet in a folder.",
    "It had rained the night before.",
    "The streets were calm and quiet.",
    "They enjoyed working together.",
    "They checked the tally sheet before leaving.",
    "A bird sang from a nearby fence.",
    "The event was announced in the newsletter.",
    "They were tired but happy by evening.",
)

# Phrasings by premise kind. Placeholders: {e} the person the premise is about,
# {a}/{b} inputs, {c} the stated number, {frac} a fraction word, {unit}, {verb},
# {verb_pl}. The index of the phrasing is fixed per template.
PHRASES: dict[str, tuple[str, ...]] = {
    "given": (
        "{e} {verb} {c} {unit}.",
        "{e} {verb} a total of {c} {unit}.",
    ),
    "add_c": (
        "{e} {verb} {c} more {unit} than {a}.",
        "Compared with {a}, {e} {verb} {c} more {unit}.",
    ),
    "sub_c": (
        "{e} {verb} {c} fewer {unit} than {a}.",
        "Compared with {a}, {e} {verb} {c} fewer {unit}.",
    ),
    "mul_c": (
        "{e} {verb} {times} as many {unit} as {a}.",
        "{e} {verb} {times} the number of {unit} that {a} {verb}.",
    ),
    "div_c": (
        "{e} {verb} {frac} as many {unit} as {a}.",
        "{e} {verb} {frac} of the number of {unit} that {a} {verb}.",
    ),
    "add2": (
        "{e} {verb} as many {unit} as {a} and {b} {verb_pl} combined.",
        "The number of {unit} {e} {verb} equals the number {a} {verb} plus the number {b} {verb}.",
    ),
    "sub2": (
        "The number of {unit} {e} {verb} equals the number {a} {verb} minus the number {b} {verb}.",
        "{e} {verb} as many {unit} as {a} {verb} minus {b}'s total.",
    ),
}

# Vague replacements for a stated number (the "ambiguate" flaw). A range keeps the
# number bounded; a word leaves it unbounded. Multipliers are left out because a
# vague factor times an unknown count is not linear.
VAGUE_WORDS = ("some", "a few", "several")
VAGUE_PHRASES: dict[str, tuple[str, str]] = {
    # kind: (word form, range form)
    "given": ("{e} {verb} {word} {unit}.", "{e} {verb} between {lo} and {hi} {unit}."),
    "add_c": (
        "{e} {verb} {word} more {unit} than {a}.",
        "{e} {verb} between {lo} and {hi} more {unit} than {a}.",
    ),
    "sub_c": (
        "{e} {verb} {word} fewer {unit} than {a}.",
        "{e} {verb} between {lo} and {hi} fewer {unit} than {a}.",
    ),
}

FRACTIONS = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 10: "a tenth"}
DIVISORS = tuple(FRACTIONS)


def times_word(c: int) -> str:
    return "twice" if c == 2 else f"{c} times"


def join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"
