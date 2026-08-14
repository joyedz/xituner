"""Behavioral Referee: the judgement the deterministic checks provably cannot make.

Why this exists, with evidence rather than argument
--------------------------------------------------
A real run on the flawed corpus (`outputs/nimbus_flawed`) scored 85% on the
mechanical tacit rules and passed the aggregate gap test. The deterministic layer
correctly caught what it is good at -- a missing sign-off, forbidden corporate
vocabulary, a rule that never passed. It missed all three of the worst failures,
because none of them are expressible as a regex:

  1. A customer reported a bottle arriving with its seal already broken -- a
     product-safety issue -- and the model replied "that is not our shipping
     error, it has become a favourite of many people". It denied fault and
     pivoted to promotion.
  2. Asked to swap empty bottles for a free product, the model upsold a
     three-bottle bundle instead of declining.
  3. Asked about job vacancies, the model emitted "send your CV to 0812-3456789"
     -- a phone number that appears nowhere in the corpus or the prompt.

Number 3 is the one that matters most. Every mechanical rule passed on it: two
sentences, correct address form, no exclamation mark. A fabricated contact
number is a legal problem for a real brand, and no amount of pattern matching
finds it, because the failure is that the content is INVENTED, not that it is
malformed.

The anti-hallucination guard
----------------------------
A judge that can hallucinate is not a judge. Every verdict must quote VERBATIM
from the output it is judging, and `judge_row` verifies that quote actually
appears there. A verdict whose evidence cannot be found is marked unverified and
excluded from the aggregate rather than trusted -- so the Referee cannot invent a
justification for a score any more than the tuned model can invent a phone
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from llm.client import LLMClient, LLMError

SYSTEM = (
    "You are a strict evaluator of fine-tuned language model outputs. "
    "You judge SEMANTICS, not formatting -- another system already checks "
    "mechanical rules. Be skeptical: your job is to find real problems, not to "
    "be agreeable. Every judgement must quote verbatim from the output you are "
    "given, so it can be verified."
)


class RowVerdict(BaseModel):
    """One held-out row, judged semantically."""

    addresses_the_input: bool = Field(
        description="Does the output actually respond to what was asked?"
    )
    invents_unverifiable_specifics: bool = Field(
        description=(
            "Does the output state specific facts -- phone numbers, prices, "
            "dates, addresses, URLs, names -- that do not appear in the input "
            "and cannot be verified? Fabricated contact details are the most "
            "serious case."
        )
    )
    handles_situation_appropriately: bool = Field(
        description=(
            "For this kind of input, is the response the right KIND of response? "
            "A complaint needs acknowledgement, not denial or a sales pitch. A "
            "request that must be declined needs a decline, not an upsell."
        )
    )
    contradicts_or_denies_wrongly: bool = Field(
        description=(
            "Does the output deny responsibility, contradict the customer, or "
            "dismiss a legitimate concern?"
        )
    )
    severity: str = Field(description="Exactly one of: ok, minor, major")
    evidence_quote: str = Field(
        description=(
            "A short VERBATIM span copied character-for-character from the "
            "output being judged, supporting your assessment. Do not paraphrase "
            "and do not quote the input."
        )
    )
    reasoning: str = Field(description="One or two sentences, in Indonesian.")


@dataclass
class JudgedRow:
    category: str | None
    prompt: str
    output: str
    verdict: RowVerdict | None
    evidence_verified: bool
    error: str | None = None

    @property
    def usable(self) -> bool:
        """Only verdicts whose evidence checks out feed the aggregate."""
        return self.verdict is not None and self.evidence_verified

    @property
    def is_major(self) -> bool:
        return self.usable and self.verdict.severity.lower() == "major"


@dataclass
class RefereeReport:
    rows: list[JudgedRow] = field(default_factory=list)

    @property
    def usable_rows(self) -> list[JudgedRow]:
        return [r for r in self.rows if r.usable]

    @property
    def unverified_count(self) -> int:
        return sum(1 for r in self.rows if r.verdict is not None and not r.evidence_verified)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.rows if r.error)

    def _rate(self, pred) -> float:
        usable = self.usable_rows
        if not usable:
            return 0.0
        return sum(1 for r in usable if pred(r.verdict)) / len(usable)

    @property
    def hallucination_rate(self) -> float:
        return self._rate(lambda v: v.invents_unverifiable_specifics)

    @property
    def inappropriate_rate(self) -> float:
        return self._rate(lambda v: not v.handles_situation_appropriately)

    @property
    def wrongful_denial_rate(self) -> float:
        return self._rate(lambda v: v.contradicts_or_denies_wrongly)

    @property
    def major_rate(self) -> float:
        return self._rate(lambda v: v.severity.lower() == "major")

    def summary(self) -> str:
        usable = len(self.usable_rows)
        return (
            f"{usable}/{len(self.rows)} rows judged"
            f"{f' ({self.unverified_count} unverified, {self.error_count} errored)' if (self.unverified_count or self.error_count) else ''}\n"
            f"  hallucinated specifics : {self.hallucination_rate:.0%}\n"
            f"  wrong kind of response : {self.inappropriate_rate:.0%}\n"
            f"  wrongful denial        : {self.wrongful_denial_rate:.0%}\n"
            f"  major severity         : {self.major_rate:.0%}"
        )

    def failures_by_category(self) -> dict[str, list[str]]:
        """What went wrong, grouped by category -- the Diagnostician's input."""
        out: dict[str, list[str]] = {}
        for row in self.usable_rows:
            v = row.verdict
            problems = []
            if v.invents_unverifiable_specifics:
                problems.append("invented specifics")
            if not v.handles_situation_appropriately:
                problems.append("wrong kind of response")
            if v.contradicts_or_denies_wrongly:
                problems.append("wrongful denial")
            if not v.addresses_the_input:
                problems.append("did not address the input")
            if problems:
                out.setdefault(row.category or "uncategorised", []).extend(problems)
        return out


def _normalise(text: str) -> str:
    """Collapse whitespace and case for quote matching.

    A model that copies a span correctly but re-wraps a line, or shifts case at a
    sentence start, has not hallucinated -- rejecting that would throw away good
    verdicts. Anything looser than this (substring of a substring, fuzzy match)
    would let a genuinely invented quote through, which is the failure this guard
    exists to prevent.
    """
    return " ".join(text.lower().split())


def verify_evidence(quote: str, output: str, *, min_chars: int = 8) -> bool:
    """Does the cited span really appear in the judged output?

    A very short quote is not evidence of anything -- "Sob" appears in every
    reply -- so anything under `min_chars` is rejected even if present.
    """
    quote = quote.strip().strip('"\u201c\u201d\'')
    if len(quote) < min_chars:
        return False
    return _normalise(quote) in _normalise(output)


def judge_row(
    client: LLMClient,
    *,
    prompt: str,
    output: str,
    ground_truth: str | None = None,
    category: str | None = None,
    use_case_description: str = "",
) -> JudgedRow:
    """Judge one output. Returns a JudgedRow even on failure, never raises.

    A single flaky call must not abort a run that has already paid for training,
    so errors are captured per row and surfaced in the aggregate instead.
    """
    if not output.strip():
        return JudgedRow(
            category=category, prompt=prompt, output=output, verdict=None,
            evidence_verified=False, error="empty output",
        )

    parts = [
        f"Task: {use_case_description}" if use_case_description else "",
        f"Kind of input: {category}" if category else "",
        f"INPUT:\n{prompt}",
        f"MODEL OUTPUT (judge this):\n{output}",
    ]
    if ground_truth:
        # The reference shows what a correct response looks like. It is given as
        # a guide, not a target: an output can be worded completely differently
        # and still be right, so the Referee is told not to penalise divergence
        # for its own sake -- word-level closeness is already measured
        # deterministically elsewhere.
        parts.append(
            f"REFERENCE (one acceptable answer, for orientation only -- do NOT "
            f"penalise wording that merely differs):\n{ground_truth}"
        )
    parts.append(
        "Judge the MODEL OUTPUT. Quote verbatim from it in evidence_quote."
    )
    full_prompt = "\n\n".join(p for p in parts if p)

    try:
        verdict = client.structured(full_prompt, RowVerdict, system=SYSTEM)
    except LLMError as exc:
        return JudgedRow(
            category=category, prompt=prompt, output=output, verdict=None,
            evidence_verified=False, error=str(exc)[:200],
        )

    verified = verify_evidence(verdict.evidence_quote, output)
    return JudgedRow(
        category=category, prompt=prompt, output=output,
        verdict=verdict, evidence_verified=verified,
    )


def judge_rows(
    client: LLMClient,
    rows: list[dict],
    *,
    output_key: str = "tuned_output",
    use_case_description: str = "",
    verbose: bool = True,
) -> RefereeReport:
    """Judge every row of a comparison report."""
    report = RefereeReport()
    for i, row in enumerate(rows, start=1):
        judged = judge_row(
            client,
            prompt=row["prompt"],
            output=row.get(output_key, ""),
            ground_truth=row.get("ground_truth"),
            category=row.get("category"),
            use_case_description=use_case_description,
        )
        report.rows.append(judged)

        if verbose:
            if judged.error:
                print(f"  [{i}/{len(rows)}] {judged.category:<18} ERROR: {judged.error[:60]}")
            elif not judged.evidence_verified:
                print(
                    f"  [{i}/{len(rows)}] {judged.category:<18} UNVERIFIED "
                    f"(quote not found in output: {judged.verdict.evidence_quote[:40]!r})"
                )
            else:
                v = judged.verdict
                flags = []
                if v.invents_unverifiable_specifics:
                    flags.append("INVENTED")
                if not v.handles_situation_appropriately:
                    flags.append("WRONG-KIND")
                if v.contradicts_or_denies_wrongly:
                    flags.append("DENIAL")
                tag = " ".join(flags) if flags else "ok"
                print(f"  [{i}/{len(rows)}] {judged.category:<18} {v.severity:<6} {tag}")
    return report
