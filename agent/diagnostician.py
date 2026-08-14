"""Diagnostician: turn "it failed" into "change THIS about the data".

This is the component whose job a human was doing by hand. In an earlier session
the sequence was: read the flawed run's outputs, notice the model denied a
complaint and upsold instead of declining, trace it to a corpus holding 200 promo
captions and zero refusal examples, and prescribe pruning promo and injecting
refusals. Every step of that is what this module is supposed to do on its own.

Why the prescription is about DATA, not hyperparameters
------------------------------------------------------
The flawed corpus contained no refusal examples at all. There is no learning
rate, no epoch count, no LoRA rank that teaches a behaviour absent from the
training data. Reaching for hyperparameters there would be a category error --
and it is the error a naive "AI tunes your model" agent would make, because
hyperparameters are the obvious knob.

So the output is a list of operations on the corpus: prune an over-represented
category, inject examples of a pattern that is missing, synthesize rows for a gap
in coverage. `training/hyperparams.py` continues to pick hyperparameters from a
deterministic table, and no LLM is consulted about them.

Proposer, not applier
---------------------
This module PROPOSES. `agent/prescription.py` validates the proposal
deterministically and only then applies it. That separation is the safety
property: no LLM output mutates a corpus directly, so a bad prescription costs a
rejection message rather than a destroyed dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from agent.referee import RefereeReport
from llm.client import LLMClient, LLMError

SYSTEM = (
    "You diagnose fine-tuning failures. You are given evidence of how a tuned "
    "model behaved and the composition of the corpus it was trained on. Your job "
    "is to explain the failure in terms of the DATA and prescribe changes to the "
    "corpus.\n\n"
    "You must NOT prescribe hyperparameter changes. Learning rate, epochs and "
    "LoRA rank are chosen by a deterministic table and are not yours to touch. A "
    "behaviour absent from the training data cannot be taught by any "
    "hyperparameter, and that is usually the real cause.\n\n"
    "Be specific and quantitative. Name categories and counts."
)

VALID_OPS = ("prune", "inject", "synthesize")


class DataOperation(BaseModel):
    op: str = Field(description="Exactly one of: prune, inject, synthesize")
    category: str = Field(
        description="The corpus category this operates on, e.g. promo_caption"
    )
    target_count: int | None = Field(
        default=None,
        description="For prune: how many rows of this category should REMAIN.",
    )
    count: int | None = Field(
        default=None,
        description="For inject/synthesize: how many NEW rows to add.",
    )
    rationale: str = Field(description="Why this operation, in one sentence.")


class Prescription(BaseModel):
    diagnosis: str = Field(
        description=(
            "What went wrong and why, in terms of corpus composition. Two or "
            "three sentences. Cite counts."
        )
    )
    failure_modes: list[str] = Field(
        description="Short labels, e.g. topic_collapse, missing_refusal_examples"
    )
    root_cause_is_data: bool = Field(
        description=(
            "True if the failure is explained by what the corpus does or does not "
            "contain. False only if the evidence genuinely points elsewhere."
        )
    )
    operations: list[DataOperation]
    expected_effect: str = Field(
        description="What should change after retraining, in one sentence."
    )


@dataclass
class CorpusStats:
    """What the corpus actually contains. Computed, never asked of the LLM."""

    path: Path
    total: int
    by_category: dict[str, int]

    def describe(self) -> str:
        lines = [f"Corpus: {self.total} rows total"]
        for cat, n in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
            share = n / self.total if self.total else 0
            lines.append(f"  {cat}: {n} rows ({share:.0%})")
        return "\n".join(lines)

    def missing_relative_to(self, expected: list[str]) -> list[str]:
        """Categories that a use case expects but the corpus has none of.

        Passed to the LLM as an explicit fact rather than left for it to notice,
        because a zero is exactly the thing that is invisible in a distribution:
        the category simply is not in the list.
        """
        return [c for c in expected if self.by_category.get(c, 0) == 0]


def corpus_stats(path: Path) -> CorpusStats:
    counts: Counter[str] = Counter()
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts[row.get("category", "uncategorised")] += 1
            total += 1
    return CorpusStats(path=path, total=total, by_category=dict(counts))


def _clean(prescription: Prescription) -> tuple[Prescription, list[str]]:
    """Drop operations the schema allowed but that make no sense.

    Structured output guarantees the SHAPE, not the semantics: the model can
    still return `op="prune"` with no target_count, or an unknown op name. These
    are dropped with a note rather than passed downstream, where they would fail
    less legibly.
    """
    notes: list[str] = []
    kept: list[DataOperation] = []
    for op in prescription.operations:
        name = op.op.strip().lower()
        if name not in VALID_OPS:
            notes.append(f"dropped unknown op {op.op!r} on {op.category!r}")
            continue
        if name == "prune" and op.target_count is None:
            notes.append(f"dropped prune on {op.category!r}: no target_count")
            continue
        if name in ("inject", "synthesize") and not op.count:
            notes.append(f"dropped {name} on {op.category!r}: no count")
            continue
        op.op = name
        kept.append(op)
    prescription.operations = kept
    return prescription, notes


def diagnose(
    client: LLMClient,
    *,
    referee: RefereeReport,
    stats: CorpusStats,
    systematic_failures: dict[str, int] | None = None,
    expected_categories: list[str] | None = None,
    use_case_description: str = "",
    rule_text: dict[str, str] | None = None,
) -> tuple[Prescription, list[str]]:
    """Produce a prescription from evidence. Returns (prescription, cleanup_notes)."""
    sections: list[str] = []

    if use_case_description:
        sections.append(f"GOAL:\n{use_case_description}")

    if rule_text:
        rules = "\n".join(f"  - {k}: {v}" for k, v in sorted(rule_text.items()))
        sections.append(f"RULES THE MODEL WAS SUPPOSED TO LEARN:\n{rules}")

    sections.append(f"CORPUS COMPOSITION:\n{stats.describe()}")

    if expected_categories:
        missing = stats.missing_relative_to(expected_categories)
        if missing:
            sections.append(
                "CATEGORIES WITH ZERO ROWS (present in the evaluation set but "
                f"absent from training):\n  {', '.join(missing)}"
            )

    if systematic_failures:
        listed = "\n".join(
            f"  - {rule}: failed on {n}/{n} applicable rows"
            for rule, n in sorted(systematic_failures.items())
        )
        sections.append(f"RULES THAT NEVER PASSED ONCE:\n{listed}")

    sections.append(
        "SEMANTIC JUDGEMENTS OF THE TUNED MODEL'S OUTPUTS:\n" + referee.summary()
    )

    by_cat = referee.failures_by_category()
    if by_cat:
        listed = "\n".join(
            f"  - {cat}: {', '.join(sorted(set(problems)))}"
            for cat, problems in sorted(by_cat.items())
        )
        sections.append(f"FAILURES BY CATEGORY:\n{listed}")

    # Concrete examples beat aggregates for diagnosis: "60% wrong kind of
    # response" does not say what the model actually did.
    examples = [r for r in referee.usable_rows if r.is_major][:4]
    if examples:
        rendered = "\n\n".join(
            f"  input: {r.prompt}\n  output: {r.output}\n  problem: {r.verdict.reasoning}"
            for r in examples
        )
        sections.append(f"WORST INDIVIDUAL FAILURES:\n{rendered}")

    sections.append(
        "Diagnose the failure and prescribe corpus operations. Use prune to "
        "reduce an over-represented category (give target_count), inject or "
        "synthesize to add rows for a missing behaviour (give count)."
    )

    prompt = "\n\n".join(sections)
    prescription = client.structured(prompt, Prescription, system=SYSTEM)
    return _clean(prescription)


def render(prescription: Prescription, notes: list[str] | None = None) -> str:
    lines = [
        "DIAGNOSIS",
        f"  {prescription.diagnosis}",
        "",
        f"root cause is data : {prescription.root_cause_is_data}",
        f"failure modes      : {', '.join(prescription.failure_modes)}",
        "",
        "OPERATIONS",
    ]
    if not prescription.operations:
        lines.append("  (none proposed)")
    for op in prescription.operations:
        amount = (
            f"-> keep {op.target_count}" if op.op == "prune" else f"+{op.count} rows"
        )
        lines.append(f"  {op.op:<11} {op.category:<18} {amount}")
        lines.append(f"              {op.rationale}")
    lines += ["", f"EXPECTED EFFECT\n  {prescription.expected_effect}"]
    if notes:
        lines += ["", "CLEANUP"] + [f"  {n}" for n in notes]
    return "\n".join(lines)
