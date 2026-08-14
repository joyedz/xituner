"""Validate and apply a prescription. Deterministic on purpose.

The safety property of the whole agent loop lives here: the Diagnostician
PROPOSES and this module DECIDES. No LLM output ever mutates a corpus directly.

Why that separation is not paranoia
-----------------------------------
A prescription is generated text. Structured output constrains its shape, not its
sense -- the model can return a syntactically perfect operation that would delete
most of the training data, or prune a category to zero and thereby create exactly
the gap it was asked to fix. Applying that blindly would let one bad generation
destroy a dataset, and the agent would then diagnose the damage it had caused.

So every operation passes fixed rules before anything is written:

  - the corpus may not fall below an absolute floor
  - no single step may remove more than a fraction of the corpus
  - no category may be pruned out of existence
  - a prune must actually reduce (a "prune" that grows a category is a mistake,
    not an instruction)

Rejections are returned as reasons, not exceptions, so the orchestrator can log
them, count the iteration as failed, and stop rather than crash.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agent.diagnostician import CorpusStats, DataOperation, Prescription


@dataclass
class ValidationResult:
    approved: list[DataOperation] = field(default_factory=list)
    rejected: list[tuple[DataOperation, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """At least one operation survived and nothing was rejected."""
        return bool(self.approved) and not self.rejected

    @property
    def any_approved(self) -> bool:
        return bool(self.approved)

    def render(self) -> str:
        lines = []
        for op in self.approved:
            amount = (
                f"keep {op.target_count}" if op.op == "prune" else f"+{op.count}"
            )
            lines.append(f"  APPROVED  {op.op:<11} {op.category:<18} {amount}")
        for op, reason in self.rejected:
            lines.append(f"  REJECTED  {op.op:<11} {op.category:<18} {reason}")
        for warning in self.warnings:
            lines.append(f"  WARNING   {warning}")
        return "\n".join(lines) or "  (no operations)"


def describe_limits(
    stats: CorpusStats,
    *,
    min_corpus_rows: int = 50,
    max_removed_fraction: float = 0.40,
    min_category_rows: int = 5,
    max_added_per_step: int = 200,
) -> str:
    """Render the validator's limits as prompt text for the proposer.

    Derived from the same defaults `validate` uses rather than written out in the
    Diagnostician's prompt, so the stated budget cannot drift from the enforced
    one. A prompt promising a 40% allowance while the validator enforces 25%
    would produce rejections that look like model error.
    """
    max_removable = int(stats.total * max_removed_fraction)
    return "\n".join([
        f"  - Prunes may remove at most {max_removable} rows in total "
        f"({max_removed_fraction:.0%} of the current {stats.total}). This counts "
        "removals only; adding rows elsewhere does not buy more removal budget.",
        f"  - No category may be pruned below {min_category_rows} rows.",
        f"  - The corpus may not fall below {min_corpus_rows} rows.",
        f"  - At most {max_added_per_step} rows may be added per category per step.",
        "  - Prunes over budget are rejected as a group while additions still "
        "apply, so prefer a prune that fits over one that does not. Rebalancing "
        "across several iterations is expected and is not a failure.",
    ])


def validate(
    prescription: Prescription,
    stats: CorpusStats,
    *,
    min_corpus_rows: int = 50,
    max_removed_fraction: float = 0.40,
    min_category_rows: int = 5,
    max_added_per_step: int = 200,
    measurable_categories: list[str] | None = None,
) -> ValidationResult:
    """Check a prescription against fixed limits. Never raises."""
    result = ValidationResult()
    projected = dict(stats.by_category)
    # Removals are tracked on their own rather than inferred from the net change
    # in total rows. Inferring it lets additions hide deletions: a step that
    # prunes 210 of 360 rows (58%) while synthesizing 95 new ones nets out to a
    # 32% drop and slides under a 40% limit, even though more than half the
    # original corpus is gone. The limit exists to bound destruction, so it has
    # to count destruction.
    removed_total = 0

    for op in prescription.operations:
        if op.op == "prune":
            have = projected.get(op.category, 0)
            if have == 0:
                result.rejected.append(
                    (op, f"category absent from the corpus (nothing to prune)")
                )
                continue
            target = op.target_count or 0
            if target >= have:
                result.rejected.append(
                    (op, f"prune would not reduce anything ({have} -> {target})")
                )
                continue
            if target < min_category_rows:
                result.rejected.append(
                    (
                        op,
                        f"would leave {target} rows, below the {min_category_rows}-row "
                        "floor -- pruning a category to nothing recreates the gap "
                        "this is meant to fix",
                    )
                )
                continue
            projected[op.category] = target
            removed_total += have - target

        elif op.op in ("inject", "synthesize"):
            count = op.count or 0
            if count <= 0:
                result.rejected.append((op, "non-positive count"))
                continue
            if count > max_added_per_step:
                result.rejected.append(
                    (op, f"{count} exceeds the {max_added_per_step}-row per-step cap")
                )
                continue
            projected[op.category] = projected.get(op.category, 0) + count

        else:
            result.rejected.append((op, f"unknown op {op.op!r}"))
            continue

        result.approved.append(op)

    # Whole-corpus limits, checked against the combined effect rather than each
    # operation alone: several individually-modest prunes can still gut a corpus.
    projected_total = sum(projected.values())
    reason = None
    if projected_total < min_corpus_rows:
        reason = (
            f"combined prunes would leave {projected_total} rows, below the "
            f"{min_corpus_rows}-row floor"
        )
    elif stats.total and removed_total / stats.total > max_removed_fraction:
        reason = (
            f"prunes remove {removed_total}/{stats.total} rows "
            f"({removed_total / stats.total:.0%}), over the "
            f"{max_removed_fraction:.0%} per-step limit "
            f"(additions do not offset this)"
        )

    if reason:
        # Reject the prunes, keep the additions.
        #
        # Two judgement calls here, both deliberate. First, additions survive: a
        # limit on how much of the corpus one step may destroy has no business
        # vetoing rows being added, and the failure that triggered this loop is
        # usually a MISSING category -- refusing the fix because an unrelated
        # prune was greedy would leave the corpus broken in the one way it was
        # known to be broken.
        #
        # Second, ALL prunes are rejected rather than a subset that fits. Picking
        # which prunes to keep would be the validator inventing a plan nobody
        # proposed, and the choice would depend on list order. Saying no to the
        # whole pruning half is a decision this code can defend; negotiating is
        # not.
        #
        # The corpus therefore stays unbalanced this iteration. That is the loop
        # working: the next round re-diagnoses from measured evidence and can
        # propose a prune that fits, so convergence takes more iterations instead
        # of one destructive leap.
        prunes = [op for op in result.approved if op.op == "prune"]
        result.rejected.extend((op, reason) for op in prunes)
        result.approved = [op for op in result.approved if op.op != "prune"]

    # Adding a category the evaluation set does not cover is allowed but flagged.
    # Those rows get trained on and never measured, so the loop can grow a corpus
    # and report progress while the thing it added stays invisible to every later
    # verdict. Not a rejection -- a genuinely new category can be the right call,
    # and refusing it would make the agent unable to propose one -- but it should
    # never happen silently.
    if measurable_categories is not None:
        known = set(measurable_categories)
        unknown = sorted({
            op.category for op in result.approved
            if op.op in ("inject", "synthesize") and op.category not in known
        })
        if unknown:
            result.warnings.append(
                f"adding categories absent from the evaluation set: "
                f"{', '.join(unknown)} -- these rows will be trained on but "
                "never measured, so no later verdict can tell whether they "
                "helped. Extend the held-out set, or drop them."
            )

    return result


@dataclass
class ApplyResult:
    out_path: Path
    before: dict[str, int]
    after: dict[str, int]
    pruned: int = 0
    added: int = 0
    unsatisfied: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"wrote {self.out_path}", f"  pruned {self.pruned}, added {self.added}"]
        cats = sorted(set(self.before) | set(self.after))
        for cat in cats:
            b, a = self.before.get(cat, 0), self.after.get(cat, 0)
            arrow = "" if a == b else f"  <-- {a - b:+d}"
            lines.append(f"  {cat:<18} {b:>4} -> {a:>4}{arrow}")
        for note in self.unsatisfied:
            lines.append(f"  UNSATISFIED: {note}")
        return "\n".join(lines)


def apply(
    approved: list[DataOperation],
    corpus_path: Path,
    out_path: Path,
    *,
    new_rows: dict[str, list[dict]] | None = None,
    donor_path: Path | None = None,
    seed: int = 20260814,
) -> ApplyResult:
    """Execute approved operations and write a new corpus version.

    Deterministic throughout. Prune drops rows from an over-represented category.
    Inject/synthesize ADD rows, and this function does not generate them -- it
    only places rows it is handed.

    That division is deliberate. An applier that called an LLM would be writing
    generated text straight into a training corpus, and the guarantee that every
    row passed the use case's own validator would depend on the applier
    remembering to check. Keeping generation outside means the rows arriving here
    have already been through `agent/synthesizer.py`'s gates: rule compliance,
    held-out contamination, and duplication.

    `new_rows` maps category -> validated rows (the normal path, produced by the
    synthesizer). `donor_path` is a fallback that copies real rows from another
    corpus, useful for testing the plumbing without paying for generation.
    Whatever cannot be satisfied is reported as UNSATISFIED rather than silently
    becoming a no-op.
    """
    rng = random.Random(seed)

    rows: list[dict] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    before: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "uncategorised")
        before[cat] = before.get(cat, 0) + 1

    result = ApplyResult(out_path=out_path, before=before, after={})

    # --- prune ---------------------------------------------------------
    for op in (o for o in approved if o.op == "prune"):
        target = op.target_count or 0
        in_cat = [r for r in rows if r.get("category") == op.category]
        if len(in_cat) <= target:
            continue
        keep_ids = set()
        # Keep a random subset rather than the first N: corpora are often written
        # grouped or in generation order, so taking a prefix would bias toward
        # whichever variants happen to come first.
        for r in rng.sample(in_cat, target):
            keep_ids.add(id(r))
        removed_here = 0
        # Named `kept_rows`, not `new_rows`: `new_rows` is this function's
        # parameter holding the synthesized rows to add, and reusing the name
        # here silently replaced that dict with a list. The inject stage then
        # called .items() on a list and crashed. Cheap bug, expensive to find.
        kept_rows = []
        for r in rows:
            if r.get("category") == op.category and id(r) not in keep_ids:
                removed_here += 1
                continue
            kept_rows.append(r)
        rows = kept_rows
        result.pruned += removed_here

    # --- inject / synthesize -------------------------------------------
    supplied: dict[str, list[dict]] = {k: list(v) for k, v in (new_rows or {}).items()}

    # Fallback donors, only consulted for categories with no supplied rows.
    if donor_path and donor_path.exists():
        with donor_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cat = d.get("category", "uncategorised")
                if cat not in supplied:
                    supplied.setdefault(f"__donor__{cat}", []).append(d)

    for op in (o for o in approved if o.op in ("inject", "synthesize")):
        want = op.count or 0
        pool = supplied.get(op.category) or supplied.get(f"__donor__{op.category}") or []
        if not pool:
            result.unsatisfied.append(
                f"{op.op} {want} rows of {op.category!r}: nothing supplied. Run "
                "agent/synthesizer.py first -- rows must pass its rule, "
                "contamination and duplication gates before entering the corpus."
            )
            continue

        take = min(want, len(pool))
        rows.extend(dict(r) for r in pool[:take])
        result.added += take
        if take < want:
            # Report the gap rather than padding with repeats: forty copies of
            # one example satisfies the count while teaching nothing new, and
            # would look like success in the row totals.
            result.unsatisfied.append(
                f"{op.category!r}: {take}/{want} rows available, short by "
                f"{want - take}. Not padded with duplicates -- repeats add count "
                "without adding signal."
            )

    rng.shuffle(rows)

    after: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "uncategorised")
        after[cat] = after.get(cat, 0) + 1
    result.after = after

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return result
