"""Text metrics that hold for any use case.

`similarity` lives here rather than in a use case because closeness to a
held-out target is what makes a comparison verifiable by someone who has never
seen the goal: they are not asked whether an output "looks right", only which
candidate lands nearer the real one. That question is the same whether the target
is a brand reply or a JSON object.

Emoji detection is here too because it is mechanical. Which emoji are ALLOWED is
use-case-specific and stays with the use case.
"""

from __future__ import annotations

import re

# Broad emoji/pictograph ranges. Detection is generic; policy is not.
EMOJI_RX = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]"
)

_WORD_RX = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return _WORD_RX.findall(text.lower())


def find_emoji(text: str, allowed: list[str] | None = None) -> list[str]:
    """Emoji present. `allowed` ones are matched first so variation selectors
    (a trailing U+FE0F on some emoji) do not split them into fragments."""
    found: list[str] = []
    stripped = text
    for e in allowed or []:
        if e in text:
            found.append(e)
            stripped = stripped.replace(e, "")
    found += EMOJI_RX.findall(stripped)
    return found


def similarity(candidate: str, ground_truth: str) -> dict[str, float]:
    """How close a candidate lands to the held-out target.

    Three cheap, independent components rather than one opaque score, so a
    result can be read: heavy word overlap with the wrong length reads
    differently from the right length with the wrong words.
    """
    cw, gw = set(words(candidate)), set(words(ground_truth))
    jaccard = len(cw & gw) / len(cw | gw) if (cw | gw) else 0.0

    cl, gl = len(words(candidate)), len(words(ground_truth))
    length_ratio = min(cl, gl) / max(cl, gl) if max(cl, gl) else 0.0

    ce, ge = set(find_emoji(candidate)), set(find_emoji(ground_truth))
    emoji_match = 1.0 if ce == ge else (0.5 if ce & ge else 0.0)

    return {
        "word_overlap": jaccard,
        "length_ratio": length_ratio,
        "emoji_match": emoji_match,
        "closeness": (jaccard + length_ratio + emoji_match) / 3,
    }
