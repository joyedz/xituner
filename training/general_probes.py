"""Leg 2: does fine-tuning break things that have nothing to do with the brand?

Borrowed from Soup's `soup ship` (docs/evaluation.md):

    SHIP  <=>  task_tuned > task_base   AND   no general benchmark regressed
               past the forgetting threshold

Everything we had before this file was leg 1 only: does the tuned model sound
more like Nimbus than a prompted base model does. That question has no floor on
it -- a model that answers EVERY prompt with brand voice, including "2+2=?",
would score well on leg 1 and reveal nothing wrong there. Leg 1 cannot catch
that failure by construction, because every leg-1 prompt is a brand prompt.

Leg 2 asks the question leg 1 structurally cannot: on prompts that have NOTHING
to do with Nimbus (arithmetic, translation, general facts, plain formatting
instructions), does the tuned model still answer the actual question, or does
brand voice leak into places it was never supposed to reach?

Two independent signals per probe:

  correctness  -- does the reply contain the expected content at all
                   (`expected_regex`, word-boundary matched)
  leakage      -- did BRAND-SPECIFIC markers (the "Sob" address, the allowed
                   emoji set, brand vocabulary) show up in a reply to a prompt
                   that has nothing to do with the brand

A model can pass correctness while leaking ("4, Sob! Ada yang mau ditanya soal
kopi? ☕" contains "4" AND leaks) -- these are reported separately for exactly
that reason.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_BRAND_DIR = Path(__file__).resolve().parent.parent / "data" / "brand"
if str(_BRAND_DIR) not in sys.path:
    sys.path.insert(0, str(_BRAND_DIR))

from voice_spec import ALLOWED_EMOJI  # noqa: E402

DEFAULT_PROBES_PATH = _BRAND_DIR / "generic_probes.jsonl"

# Markers that identify a reply as "in Nimbus voice" independent of whether the
# question was ABOUT Nimbus. "sob" is the strongest signal (a real address form,
# not a common Indonesian word); brand nouns and the allowed emoji set back it up.
_BRAND_MARKER_WORDS = {"sob", "nimbus", "kopi", "cold brew"}


@dataclass
class ProbeResult:
    id: str
    category: str
    prompt: str
    output: str
    correct: bool
    leaked: bool
    leak_reasons: list[str] = field(default_factory=list)


def load_probes(path: Path = DEFAULT_PROBES_PATH) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_leakage(text: str) -> tuple[bool, list[str]]:
    """Brand-voice markers present in a reply that should have none."""
    lower = text.lower()
    reasons: list[str] = []

    if re.search(r"\bsob\b", lower):
        reasons.append('brand address "Sob"')
    for word in _BRAND_MARKER_WORDS - {"sob"}:
        if word in lower:
            reasons.append(f'brand vocabulary "{word}"')
    for emoji in ALLOWED_EMOJI:
        if emoji in text:
            reasons.append(f"brand emoji {emoji!r}")

    return bool(reasons), reasons


def score_probe(probe: dict, output: str) -> ProbeResult:
    correct = bool(re.search(probe["expected_regex"], output, re.IGNORECASE))
    leaked, reasons = detect_leakage(output)
    return ProbeResult(
        id=probe["id"],
        category=probe["category"],
        prompt=probe["prompt"],
        output=output,
        correct=correct,
        leaked=leaked,
        leak_reasons=reasons,
    )


@dataclass
class GeneralLegReport:
    base_results: list[ProbeResult]
    tuned_results: list[ProbeResult]

    @property
    def base_correct_rate(self) -> float:
        return _rate(self.base_results, lambda r: r.correct)

    @property
    def tuned_correct_rate(self) -> float:
        return _rate(self.tuned_results, lambda r: r.correct)

    @property
    def base_leak_rate(self) -> float:
        return _rate(self.base_results, lambda r: r.leaked)

    @property
    def tuned_leak_rate(self) -> float:
        return _rate(self.tuned_results, lambda r: r.leaked)

    def passed(self, *, max_correctness_drop: float = 0.10, max_leak_rate: float = 0.0) -> bool:
        """The "moat": no capability regression AND no voice leakage into unrelated prompts.

        `max_leak_rate` defaults to 0.0 on purpose -- leakage on a prompt that
        has nothing to do with the brand is not a matter of degree the way a
        few accuracy points are; the voice bleeding into "how do I change my
        oil" is a real failure however small its rate.
        """
        drop = self.base_correct_rate - self.tuned_correct_rate
        return drop <= max_correctness_drop and self.tuned_leak_rate <= max_leak_rate

    def summary(self) -> str:
        return (
            f"correctness: base {self.base_correct_rate:.0%} -> "
            f"tuned {self.tuned_correct_rate:.0%}  |  "
            f"leakage: base {self.base_leak_rate:.0%} -> tuned {self.tuned_leak_rate:.0%}"
        )


def _rate(results: list[ProbeResult], pred) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if pred(r)) / len(results)
