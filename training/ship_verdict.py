"""Two-leg SHIP / DON'T SHIP verdict, modeled on Soup's `soup ship`.

Soup's rule (docs/evaluation.md, Ship Verdict section):

    SHIP  <=>  task_tuned > task_base   AND   no general benchmark regressed
               past the forgetting threshold
    else DON'T SHIP -- even if the task metric looks great.

`compare_voice.py`'s verdict before this file existed was leg 1 only: does the
tuned model beat a prompted base model on brand voice. That is necessary but
not sufficient -- a model that answers every prompt in brand voice, including
ones that have nothing to do with the brand, would pass leg 1 and be obviously
broken. `decide_ship` is `decide_ship` on purpose: a pure function over already
-computed numbers, with no model loading, no file I/O, and no randomness, so
the whole decision table is unit-testable without a GPU (Soup's own framing for
why `decide_ship` is a pure function: "the whole truth table is CPU-testable").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from training.general_probes import GeneralLegReport
from training.stats import BootstrapResult


def systematic_rule_failures(
    per_row_rules: list[dict[str, bool]], *, min_rows: int = 2
) -> dict[str, int]:
    """Rules that failed on EVERY row where they applied.

    Aggregate scores hide this. On a real run the tuned model averaged 85% tacit
    -- a comfortable-looking number -- while `signoff_used_correctly` failed on
    4 of 4 complaint/refusal rows and `refusal_offers_alternative` on 2 of 2.
    A rule that never passes is not noise averaged away by rows it does not
    apply to; it is a category the corpus never taught, and it is exactly the
    fingerprint of the promo-heavy archive with zero refusal examples.

    Averaging is the wrong instrument for that failure, so this looks at rules
    individually. `min_rows=2` keeps a single unlucky row from tripping it.
    """
    applicable: dict[str, int] = {}
    failures: dict[str, int] = {}
    for row in per_row_rules:
        for rule, ok in row.items():
            applicable[rule] = applicable.get(rule, 0) + 1
            if not ok:
                failures[rule] = failures.get(rule, 0) + 1
    return {
        rule: count
        for rule, count in applicable.items()
        if count >= min_rows and failures.get(rule, 0) == count
    }


@dataclass(frozen=True)
class ShipVerdict:
    ship: bool
    leg1_pass: bool
    leg2_pass: bool
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = "SHIP" if self.ship else "DON'T SHIP"
        lines = [head, "-" * len(head)]
        lines += [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


def decide_ship(
    *,
    tacit_ci: BootstrapResult,
    closeness_ci: BootstrapResult,
    general_leg: GeneralLegReport | None,
    systematic_failures: dict[str, int] | None = None,
    min_tacit_effect: float = 0.10,
    min_closeness_effect: float = 0.03,
    max_correctness_drop: float = 0.10,
    max_leak_rate: float = 0.0,
) -> ShipVerdict:
    """Pure decision function. No model, no I/O -- feed it computed numbers.

    Leg 1 (task win): the ENTIRE bootstrap CI for both tacit-score gap and
    closeness-to-ground-truth gap must sit above its effect floor. Using the CI
    bound rather than the point estimate is deliberate: with a 10-row held-out
    set a lucky sample can produce a big point-estimate gap that a different
    sample would not reproduce, and `is_positive` only fires when the gap
    survives that resampling.

    Leg 2 (the moat): general_leg.passed() must hold -- no material correctness
    regression on prompts unrelated to the brand, and no brand-voice leakage
    into them.  `general_leg=None` is a missing-baseline case and DON'T SHIPs,
    matching Soup's own rule that a missing baseline refuses rather than
    silently shipping.
    """
    reasons: list[str] = []

    tacit_ok = tacit_ci.is_positive(min_tacit_effect)
    reasons.append(
        f"leg1.tacit_gap    {tacit_ci.summary()}  "
        f"{'PASS' if tacit_ok else 'FAIL'} (CI must clear +{min_tacit_effect:.0%})"
    )

    closeness_ok = closeness_ci.is_positive(min_closeness_effect)
    reasons.append(
        f"leg1.closeness_gap {closeness_ci.summary()}  "
        f"{'PASS' if closeness_ok else 'FAIL'} (CI must clear +{min_closeness_effect:.0%})"
    )

    # A rule that never passes is a taught-behaviour gap, and the aggregate gap
    # above cannot see it -- a model can average 85% tacit while getting one
    # whole rule wrong every single time.
    systematic_ok = not systematic_failures
    if systematic_failures:
        listed = ", ".join(
            f"{rule} (0/{n})" for rule, n in sorted(systematic_failures.items())
        )
        reasons.append(f"leg1.systematic   FAIL -- rules that NEVER passed: {listed}")
    else:
        reasons.append("leg1.systematic   PASS -- no rule failed on every applicable row")

    leg1_pass = tacit_ok and closeness_ok and systematic_ok

    if general_leg is None:
        reasons.append("leg2.general      MISSING -- no general-probe baseline supplied")
        leg2_pass = False
    else:
        leg2_pass = general_leg.passed(
            max_correctness_drop=max_correctness_drop, max_leak_rate=max_leak_rate
        )
        drop = general_leg.base_correct_rate - general_leg.tuned_correct_rate
        reasons.append(
            f"leg2.correctness  base {general_leg.base_correct_rate:.0%} -> "
            f"tuned {general_leg.tuned_correct_rate:.0%}  (drop {drop:+.0%})  "
            f"{'PASS' if drop <= max_correctness_drop else 'FAIL'} "
            f"(tolerance {max_correctness_drop:.0%})"
        )
        reasons.append(
            f"leg2.leakage      tuned {general_leg.tuned_leak_rate:.0%} of unrelated "
            f"prompts show brand voice  "
            f"{'PASS' if general_leg.tuned_leak_rate <= max_leak_rate else 'FAIL'} "
            f"(tolerance {max_leak_rate:.0%})"
        )

    ship = leg1_pass and leg2_pass
    if ship:
        reasons.append("Both legs pass: brand voice improved, nothing else broke.")
    elif leg1_pass and not leg2_pass:
        reasons.append(
            "Leg 1 passes but leg 2 does not: the model sounds more like the "
            "brand, but that came at the cost of a capability regression or "
            "voice bleeding into unrelated replies. Do not ship on leg 1 alone."
        )
    elif not systematic_ok and tacit_ok and closeness_ok:
        reasons.append(
            "Leg 1's aggregate gaps pass, but at least one voice rule failed on "
            "EVERY row it applied to. That is a behaviour the corpus never "
            "taught, not a scoring near-miss -- and averaging hid it. Fix the "
            "corpus, not the threshold."
        )
    elif not leg1_pass:
        reasons.append(
            "Leg 1 does not pass: prompting is competitive here, so DON'T SHIP "
            "regardless of leg 2. See compare_voice.py's own guidance -- do not "
            "paper over this by weakening the style guide."
        )

    return ShipVerdict(ship=ship, leg1_pass=leg1_pass, leg2_pass=leg2_pass, reasons=reasons)
