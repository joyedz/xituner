"""Leg 2: did fine-tuning break things the goal was never about?

Borrowed from Soup's `soup ship` (docs/evaluation.md):

    SHIP  <=>  task_tuned > task_base   AND   no general benchmark regressed
               past the forgetting threshold

Leg 1 only asks whether the tuned model beats a prompted base model at the goal.
That question has no floor on it -- a model that applies the trained behaviour to
EVERY prompt, including "2+2=?", would score well on leg 1 and reveal nothing
wrong. Leg 1 cannot catch that failure by construction, because every leg-1
prompt is a goal prompt.

Leg 2 asks what leg 1 structurally cannot: on prompts unrelated to the goal
(arithmetic, translation, general facts, plain instructions), does the tuned
model still answer the actual question, or has the trained behaviour leaked into
places it was never meant to reach?

Two independent signals per probe:

  correctness  -- does the reply contain the expected content at all
                  (`expected_regex`)
  leakage      -- did the trained behaviour appear where it does not belong

A model can pass correctness while leaking ("4, Sob! Ada yang mau ditanya soal
kopi?" contains "4" AND leaks), so the two are reported separately.

What "leakage" means is USE-CASE SPECIFIC and comes from the spec's
`detect_leakage`: for a brand voice it is the brand address turning up in an
arithmetic answer; for JSON extraction it is a JSON object turning up where a
plain sentence was wanted. Those share nothing but the shape of the question,
which is why the definition lives with the use case rather than here.

`trained_behavior_ok`, and why it exists
---------------------------------------
The first version counted ANY marker on ANY non-goal prompt as leakage, and it
produced a false FAIL on a real run.

The brand-voice corpus has an `out_of_scope` category that EXPLICITLY teaches
in-voice deflection on off-topic questions ("Kami cuma kopi, Sob. Ada yang bisa
kubantu soal cold brew?"). So when the tuned model answered "how do I change my
motor oil" in voice, it was doing exactly what it was trained to do -- and leg 2
called it a defect.

Two different things were conflated:

  trained_behavior_ok: false -- CAPABILITY probes (arithmetic, translation,
                                facts, formatting). The right answer is the
                                plain answer. Trained behaviour here IS
                                over-application, and the tolerance is zero.
  trained_behavior_ok: true  -- OFF-TOPIC CONVERSATIONAL probes, where the
                                trained behaviour is correct. The probe's only
                                job is checking the model still says something
                                useful rather than collapsing.

The leak RATE is computed over `trained_behavior_ok: false` probes only.
Markers on the permitted ones are still recorded and printed, because "the
behaviour shows up here" is worth seeing -- it just is not a failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProbeResult:
    id: str
    category: str
    prompt: str
    output: str
    correct: bool
    leaked: bool
    leak_reasons: list[str] = field(default_factory=list)
    # False for capability probes, where brand voice is over-application.
    # True for off-topic conversational probes, where it is trained behaviour.
    trained_behavior_ok: bool = False

    @property
    def is_violation(self) -> bool:
        """Leakage that actually counts against the model."""
        return self.leaked and not self.trained_behavior_ok


def load_probes(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_probe(probe: dict, output: str, detect_leakage) -> ProbeResult:
    """Score one probe. `detect_leakage` comes from the use case.

    The detector used to be a module-level function with a hardcoded set of
    Nimbus markers, which made this generic file specific to one goal. What
    counts as leakage is entirely use-case-dependent: for brand voice it is the
    "Sob" address showing up in an arithmetic answer, for order extraction it is
    a JSON object showing up where a plain sentence was wanted. Those have
    nothing in common except the shape of the question, so the answer belongs
    with the use case.
    """
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
        # Default False: an unflagged probe is treated as a capability probe,
        # so forgetting the flag fails safe (stricter) rather than silently
        # excusing leakage.
        trained_behavior_ok=bool(probe.get("trained_behavior_ok", False)),
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
        """Leak rate over CAPABILITY probes only -- see trained_behavior_ok."""
        return _rate_over(self.base_results, lambda r: not r.trained_behavior_ok,
                          lambda r: r.is_violation)

    @property
    def tuned_leak_rate(self) -> float:
        return _rate_over(self.tuned_results, lambda r: not r.trained_behavior_ok,
                          lambda r: r.is_violation)

    @property
    def tuned_permitted_voice_rate(self) -> float:
        """Brand voice on off-topic prompts. Informational, not a failure."""
        return _rate_over(self.tuned_results, lambda r: r.trained_behavior_ok,
                          lambda r: r.leaked)

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
        n_cap = sum(1 for r in self.tuned_results if not r.trained_behavior_ok)
        n_off = len(self.tuned_results) - n_cap
        return (
            f"correctness: base {self.base_correct_rate:.0%} -> "
            f"tuned {self.tuned_correct_rate:.0%}\n"
            f"leakage on {n_cap} capability probes (tolerance 0%): "
            f"base {self.base_leak_rate:.0%} -> tuned {self.tuned_leak_rate:.0%}\n"
            f"brand voice on {n_off} off-topic probes (allowed, informational): "
            f"tuned {self.tuned_permitted_voice_rate:.0%}"
        )


def _rate(results: list[ProbeResult], pred) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if pred(r)) / len(results)


def _rate_over(results: list[ProbeResult], scope, pred) -> float:
    """Rate of `pred` within the subset selected by `scope`."""
    subset = [r for r in results if scope(r)]
    if not subset:
        return 0.0
    return sum(1 for r in subset if pred(r)) / len(subset)
