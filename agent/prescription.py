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
        return "\n".join(lines) or "  (no operations)"


def validate(
    prescription: Prescription,
    stats: CorpusStats,
    *,
    min_corpus_rows: int = 50,
    max_removed_fraction: float = 0.40,
    min_category_rows: int = 5,
    max_added_per_step: int = 200,
) -> ValidationResult:
    """Check a prescription against fixed limits. Never raises."""
    result = ValidationResult()
    projected = dict(stats.by_category)

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
    removed = stats.total - projected_total
    if projected_total < min_corpus_rows:
        reason = (
            f"combined effect leaves {projected_total} rows, below the "
            f"{min_corpus_rows}-row floor"
        )
        result.rejected.extend((op, reason) for op in result.approved)
        result.approved = []
    elif stats.total and removed / stats.total > max_removed_fraction:
        reason = (
            f"combined effect removes {removed}/{stats.total} rows "
            f"({removed / stats.total:.0%}), over the "
            f"{max_removed_fraction:.0%} per-step limit"
        )
        result.rejected.extend((op, reason) for op in result.approved)
        result.approved = []

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
    donor_path: Path | None = None,
    seed: int = 20260814,
) -> ApplyResult:
    """Execute approved operations and write a new corpus version.

    Prune is fully deterministic: drop rows from the over-represented category.

    Inject/synthesize need rows that do not exist yet. Generating them is a
    separate LLM step, and this function deliberately does NOT call one -- an
    applier that silently invents training data would put generated content into
    a corpus without it passing the use case's own validator first. Instead, rows
    are drawn from `donor_path` when supplied, and any shortfall is reported as
    UNSATISFIED so the caller decides rather than discovering a silent no-op.
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
        new_rows = []
        for r in rows:
            if r.get("category") == op.category and id(r) not in keep_ids:
                removed_here += 1
                continue
            new_rows.append(r)
        rows = new_rows
        result.pruned += removed_here

    # --- inject / synthesize -------------------------------------------
    donors: dict[str, list[dict]] = {}
    if donor_path and donor_path.exists():
        with donor_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    donors.setdefault(d.get("category", "uncategorised"), []).append(d)

    for op in (o for o in approved if o.op in ("inject", "synthesize")):
        want = op.count or 0
        pool = donors.get(op.category, [])
        if not pool:
            result.unsatisfied.append(
                f"{op.op} {want} rows of {op.category!r}: no donor rows available. "
                "Generating them needs a synthesis step whose output must pass the "
                "use case's validator before entering the corpus."
            )
            continue
        added = [dict(pool[i % len(pool)]) for i in range(want)]
        rows.extend(added)
        result.added += len(added)
        if want > len(pool):
            result.unsatisfied.append(
                f"{op.category!r}: only {len(pool)} distinct donor rows for {want} "
                "requested, so rows repeat"
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
