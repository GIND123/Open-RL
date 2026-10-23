"""Length control: pad problems with numberless filler to a shared word count."""

from __future__ import annotations

import random
import re

from premise_verify.themes import PADDING, Theme

WORD = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def pad(sentences: list[str], theme: Theme, target: int, rng: random.Random) -> tuple[list[str], int]:
    """Insert filler between the intro (first) and the question (last) until the
    problem reaches `target` words. Filler is picked to land as close to the target
    as the pool allows. Returns the padded sentences and how many were added."""
    pool = list(theme.padding + PADDING)
    rng.shuffle(pool)
    out = list(sentences)
    total = word_count(" ".join(out))
    added = 0
    while total < target and pool:
        gap = target - total
        fits = [s for s in pool if word_count(s) <= gap]
        pick = fits[0] if fits else min(pool, key=word_count)
        if not fits and word_count(pick) - gap > gap:
            break  # overshooting would land further from the target than stopping
        pool.remove(pick)
        out.insert(rng.randint(1, len(out) - 1), pick)
        total += word_count(pick)
        added += 1
    return out, added
