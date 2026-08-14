"""Corpus Surgeon: generate the training rows a corpus is missing.

This is what closes the loop. Without it the agent can only PRUNE -- it can
remove over-represented rows but never supply a behaviour the corpus never had,
so a corpus with zero refusal examples stays a corpus with zero refusal examples
no matter how many iterations run. Pruning alone can only ever remove
information.

The gate is the whole point
--------------------------
An LLM asked for forty refusal examples will happily produce forty strings. Some
will break the rules they are supposed to teach. Putting those into a corpus
would teach the model to break them -- the agent would be actively damaging the
dataset while reporting progress.

So every generated row passes THREE deterministic gates before it is accepted,
and each one exists because of a specific way synthesis goes wrong:

  1. RULE COMPLIANCE -- the target is scored by the use case's own scorer, the
     same one that validates the human-written reference corpus. A row whose
     target fails its own rules is rejected. This is not a soft preference:
     `scripts/validate_use_case.py` asserts the reference corpus scores 1.00, so
     synthesized rows are held to exactly the bar the handwritten ones meet.

  2. HELD-OUT CONTAMINATION -- a generated row that reproduces a held-out prompt
     silently destroys the evaluation. The model would then be trained on its own
     test set and every subsequent score would be meaningless while looking
     excellent. This is the failure that would be hardest to notice and most
     damaging, so the check is strict: near-duplicates are rejected, not just
     exact ones.

  3. DUPLICATION -- rows that repeat each other or the existing corpus add count
     without adding signal, and a "+40 rows" prescription satisfied by forty
     copies of one example is a no-op dressed as progress.

Rejections are reported per row with a reason. A shortfall is surfaced, never
silently rounded down, because the caller needs to know the prescription was only
partly satisfied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from llm.client import LLMClient, LLMError
from training.use_case import UseCaseSpec

SYSTEM = (
    "You generate training examples for fine-tuning a small language model. "
    "Every example must obey the stated rules exactly -- these examples TEACH "
    "the rules, so a violation teaches the wrong thing.\n\n"
    "Make the incoming messages realistic and varied: real people write "
    "informally, with typos, abbreviations and incomplete sentences. Do not make "
    "them uniform.\n\n"
    "The target replies, by contrast, must be perfectly consistent with the "
    "rules. Variety in the input, discipline in the output."
)


class SynthesizedRow(BaseModel):
    incoming: str = Field(description="A realistic, informal user message.")
    target: str = Field(
        description="The reply, obeying every stated rule exactly."
    )


class SynthesisBatch(BaseModel):
    rows: list[SynthesizedRow]


class RowReview(BaseModel):
    """One synthesized row reviewed for SENSE, after it already passed the rules."""

    index: int = Field(description="1-based index of the row being reviewed")
    is_coherent: bool = Field(
        description=(
            "Does the reply make sense as a response to the message, and is the "
            "advice it gives sound?"
        )
    )
    states_something_false: bool = Field(
        description=(
            "Does it assert something factually wrong or self-contradictory about "
            "the product or the situation?"
        )
    )
    problem: str = Field(
        description="If there is a problem, name it in one short phrase. Otherwise 'ok'."
    )


class BatchReview(BaseModel):
    reviews: list[RowReview]


SEMANTIC_SYSTEM = (
    "You review candidate training examples for a fine-tuning corpus. They have "
    "ALREADY passed every mechanical rule check -- formatting, tone markers and "
    "required phrases are correct, so do not comment on those.\n\n"
    "Your job is the thing rules cannot check: does the reply actually make "
    "sense, and is the advice sound? A reply can be perfectly formatted and still "
    "be nonsense.\n\n"
    "Be strict. These examples will TEACH a model, so an incoherent one teaches "
    "incoherence. Examples of what to reject: advice that contradicts the product "
    "(suggesting a customer microwave a cold brew), a claim that does not follow "
    "(offering arabica beans to someone asking for decaf), or a pronoun that "
    "points at the wrong party."
)


@dataclass
class RejectedRow:
    row: SynthesizedRow
    reason: str


@dataclass
class SynthesisResult:
    category: str
    requested: int
    accepted: list[dict] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)
    attempts: int = 0

    @property
    def shortfall(self) -> int:
        return max(0, self.requested - len(self.accepted))

    @property
    def acceptance_rate(self) -> float:
        seen = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / seen if seen else 0.0

    def render(self) -> str:
        lines = [
            f"{self.category}: {len(self.accepted)}/{self.requested} accepted "
            f"in {self.attempts} attempt(s), "
            f"acceptance rate {self.acceptance_rate:.0%}"
        ]
        by_reason: dict[str, int] = {}
        for r in self.rejected:
            key = r.reason.split(":")[0]
            by_reason[key] = by_reason.get(key, 0) + 1
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"    rejected {n:>3}  {reason}")
        if self.shortfall:
            lines.append(f"    SHORTFALL {self.shortfall} rows not produced")
        return "\n".join(lines)


_WORD_RX = re.compile(r"\w+")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _words(text: str) -> set[str]:
    return set(_WORD_RX.findall(text.lower()))


def _overlap(a: str, b: str) -> float:
    """Jaccard word overlap. Cheap, and enough to catch a near-duplicate.

    Used instead of embeddings deliberately: this runs inside a loop that already
    pays for LLM calls and GPU time, and a rephrased duplicate still shares most
    of its words. It will miss a heavy paraphrase -- that limitation is real and
    accepted, because the alternative (loose thresholds chasing paraphrase
    recall) starts rejecting legitimately distinct rows.
    """
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def semantic_review(
    client: LLMClient,
    rows: list[dict],
    *,
    use_case_description: str = "",
) -> dict[int, str]:
    """Review already-rule-compliant rows for sense. Returns {index: problem}.

    Exists because of a measured gap. The first synthesis run produced twelve rows
    that all scored 1.00 on the twelve mechanical voice rules, and three of them
    were still bad training data: one told a customer to microwave a cold brew,
    one answered a decaf request by naming the bean variety, and one pointed a
    pronoun at the wrong party. Rules cannot see any of that.

    It is the same asymmetry the Referee exists for -- deterministic checks catch
    mechanical failures, only a model catches semantic ones -- and it was applied
    to the tuned model's OUTPUTS while being forgotten for the training data being
    generated. Incoherent training data is arguably worse, because it teaches the
    incoherence rather than merely exhibiting it.

    Reviewed in ONE batched call rather than one call per row: at the measured
    ~40s per call under load, forty rows would cost half an hour, and the loop
    already pays for training.

    A failed review returns {} -- no rows blocked. That is deliberate: this is an
    extra gate on top of the deterministic ones, and losing it should degrade
    quality, not halt a run that has already paid for training.
    """
    if not rows:
        return {}

    rendered = "\n\n".join(
        f"[{i}] message: {r['messages'][0]['content']}\n"
        f"    reply  : {r['messages'][1]['content']}"
        for i, r in enumerate(rows, start=1)
    )
    prompt = (
        (f"CONTEXT:\n{use_case_description}\n\n" if use_case_description else "")
        + f"CANDIDATE TRAINING EXAMPLES:\n{rendered}\n\n"
        + "Review each one. Return a review for every index."
    )

    try:
        result = client.structured(prompt, BatchReview, system=SEMANTIC_SYSTEM)
    except LLMError:
        return {}

    problems: dict[int, str] = {}
    for review in result.reviews:
        if not review.is_coherent or review.states_something_false:
            problems[review.index] = review.problem or "incoherent"
    return problems


def load_held_out_prompts(path: Path) -> list[str]:
    prompts = []
    if not path.exists():
        return prompts
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line).get("prompt", ""))
    return prompts


def load_corpus_inputs(path: Path) -> list[str]:
    inputs = []
    if not path.exists():
        return inputs
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages") or []
            if msgs:
                inputs.append(msgs[0].get("content", ""))
    return inputs


def examples_for_category(path: Path, category: str, limit: int = 4) -> list[dict]:
    """Existing rows of a category, for few-shot grounding.

    Returns [] when the category is empty -- which is the case that matters. The
    flawed corpus has zero refusal rows, so synthesis for it has to work from the
    RULES alone. That is the realistic situation: the reason a category needs
    synthesizing is usually that nobody ever wrote examples of it.
    """
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("category") == category:
                out.append(row)
            if len(out) >= limit:
                break
    return out


def _build_prompt(
    spec: UseCaseSpec,
    category: str,
    count: int,
    *,
    guide: str,
    same_category: list[dict],
    other_category: list[dict],
    avoid_inputs: list[str],
) -> str:
    sections = [f"GOAL:\n{spec.description}", f"GUIDE:\n{guide}"]

    rules = "\n".join(
        f"  - {k}: {v}"
        for k, v in sorted({**spec.articulable_rules, **spec.tacit_rules}.items())
    )
    sections.append(
        "EVERY RULE THE TARGET MUST OBEY (these examples teach these rules, so a "
        f"violation teaches the wrong thing):\n{rules}"
    )

    if same_category:
        rendered = "\n".join(
            f"  input: {r['messages'][0]['content']}\n  target: {r['messages'][1]['content']}"
            for r in same_category
        )
        sections.append(f"EXISTING {category} EXAMPLES to match:\n{rendered}")
    else:
        sections.append(
            f"There are NO existing {category!r} examples -- that absence is why "
            "these are needed. Derive the correct form from the rules above and "
            "from the other categories' style below."
        )

    if other_category:
        rendered = "\n".join(
            f"  [{r.get('category')}] input: {r['messages'][0]['content']}\n"
            f"       target: {r['messages'][1]['content']}"
            for r in other_category
        )
        sections.append(f"STYLE REFERENCE from other categories:\n{rendered}")

    if avoid_inputs:
        listed = "\n".join(f"  - {p}" for p in avoid_inputs[:12])
        sections.append(
            "DO NOT generate inputs resembling these -- they are reserved for "
            f"evaluation and reusing them would invalidate it:\n{listed}"
        )

    sections.append(
        f"Generate exactly {count} examples for the category {category!r}. "
        "Vary the incoming messages substantially: different phrasings, lengths, "
        "levels of politeness, and typos. Keep every target strictly rule-compliant."
    )
    return "\n\n".join(sections)


def synthesize(
    client: LLMClient,
    spec: UseCaseSpec,
    category: str,
    count: int,
    *,
    corpus_path: Path | None = None,
    held_out_path: Path | None = None,
    max_attempts: int = 3,
    overbuild: float = 1.5,
    contamination_threshold: float = 0.60,
    duplicate_threshold: float = 0.85,
    semantic_gate: bool = True,
    verbose: bool = True,
) -> SynthesisResult:
    """Generate `count` validated rows for `category`.

    Asks for more than requested (`overbuild`) because some will be rejected --
    requesting exactly `count` and then rejecting three guarantees a shortfall on
    every single call.

    `semantic_gate` adds one batched LLM review after the deterministic gates,
    catching rows that satisfy every rule and still make no sense. Measured worth
    it: the first run without it accepted three incoherent rows out of twelve.
    """
    result = SynthesisResult(category=category, requested=count)

    guide = spec.guide_path.read_text(encoding="utf-8") if spec.guide_path.exists() else ""
    corpus = corpus_path or spec.train_path
    held_out = held_out_path or spec.held_out_path

    same = examples_for_category(corpus, category)
    other = [
        r
        for r in _first_of_each_other_category(corpus, category, limit=4)
    ]
    held_out_prompts = load_held_out_prompts(held_out)
    corpus_inputs = load_corpus_inputs(corpus)

    accepted_inputs: list[str] = []

    for attempt in range(1, max_attempts + 1):
        remaining = count - len(result.accepted)
        if remaining <= 0:
            break
        result.attempts = attempt
        ask = max(remaining, int(remaining * overbuild))

        prompt = _build_prompt(
            spec, category, ask,
            guide=guide,
            same_category=same,
            other_category=other,
            avoid_inputs=held_out_prompts,
        )

        try:
            batch = client.structured(prompt, SynthesisBatch, system=SYSTEM)
        except LLMError as exc:
            if verbose:
                print(f"    attempt {attempt}: generation failed -- {str(exc)[:80]}")
            continue

        for row in batch.rows:
            if len(result.accepted) >= count:
                break
            reason = _reject_reason(
                row,
                spec=spec,
                category=category,
                held_out_prompts=held_out_prompts,
                corpus_inputs=corpus_inputs,
                accepted_inputs=accepted_inputs,
                contamination_threshold=contamination_threshold,
                duplicate_threshold=duplicate_threshold,
            )
            if reason:
                result.rejected.append(RejectedRow(row=row, reason=reason))
                continue
            accepted_inputs.append(row.incoming)
            result.accepted.append(
                {
                    "category": category,
                    "synthesized": True,
                    "messages": [
                        {"role": "user", "content": row.incoming},
                        {"role": "assistant", "content": row.target},
                    ],
                }
            )

        if verbose:
            print(
                f"    attempt {attempt}: {len(result.accepted)}/{count} accepted "
                f"by the deterministic gates ({len(result.rejected)} rejected)"
            )

    # Semantic gate last, on the survivors only -- reviewing rows that a cheap
    # regex would have rejected anyway would waste the call.
    if semantic_gate and result.accepted:
        problems = semantic_review(
            client, result.accepted, use_case_description=spec.description
        )
        if problems:
            kept: list[dict] = []
            for i, row in enumerate(result.accepted, start=1):
                if i in problems:
                    result.rejected.append(
                        RejectedRow(
                            row=SynthesizedRow(
                                incoming=row["messages"][0]["content"],
                                target=row["messages"][1]["content"],
                            ),
                            reason=f"semantic: {problems[i]}",
                        )
                    )
                else:
                    kept.append(row)
            result.accepted = kept
            if verbose:
                print(
                    f"    semantic gate: {len(problems)} row(s) removed for making "
                    f"no sense despite passing every rule"
                )

    return result


def _first_of_each_other_category(
    path: Path, exclude: str, limit: int = 4
) -> list[dict]:
    """One example per other category, for style grounding without bulk."""
    seen: set[str] = set()
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cat = row.get("category")
            if cat == exclude or cat in seen:
                continue
            seen.add(cat)
            out.append(row)
            if len(out) >= limit:
                break
    return out


def _reject_reason(
    row: SynthesizedRow,
    *,
    spec: UseCaseSpec,
    category: str,
    held_out_prompts: list[str],
    corpus_inputs: list[str],
    accepted_inputs: list[str],
    contamination_threshold: float,
    duplicate_threshold: float,
) -> str | None:
    """Return a rejection reason, or None to accept. Gates run cheapest-first."""
    if not row.incoming.strip() or not row.target.strip():
        return "empty: input or target is blank"

    # Gate 1: the use case's own scorer, the same bar the handwritten corpus meets.
    report = spec.score(row.target, category)
    failures = report.failures()
    if failures:
        return f"rule violation: {', '.join(failures)}"

    # Gate 2: held-out contamination. Checked before duplication because a
    # contaminated row is the more damaging of the two -- it corrupts the
    # evaluation rather than merely wasting a slot.
    for held in held_out_prompts:
        if _norm(row.incoming) == _norm(held):
            return "held-out contamination: input duplicates a held-out prompt"
        if _overlap(row.incoming, held) >= contamination_threshold:
            return (
                f"held-out contamination: input {_overlap(row.incoming, held):.0%} "
                f"similar to held-out prompt {held[:40]!r}"
            )

    # Gate 3: duplication, against the corpus and against rows accepted so far.
    for existing in corpus_inputs:
        if _overlap(row.incoming, existing) >= duplicate_threshold:
            return f"duplicate: input near-identical to an existing corpus row"
    for earlier in accepted_inputs:
        if _overlap(row.incoming, earlier) >= duplicate_threshold:
            return "duplicate: input near-identical to another row in this batch"

    return None


def synthesize_for_prescription(
    client: LLMClient,
    spec: UseCaseSpec,
    approved_ops,
    *,
    corpus_path: Path | None = None,
    held_out_path: Path | None = None,
    verbose: bool = True,
) -> tuple[dict[str, list[dict]], list[SynthesisResult]]:
    """Run synthesis for every inject/synthesize operation in a prescription.

    Returns (rows_by_category, per_category_results) so the caller can hand the
    rows to the deterministic applier and still report what was only partly
    satisfied.
    """
    rows_by_category: dict[str, list[dict]] = {}
    results: list[SynthesisResult] = []

    for op in approved_ops:
        if op.op not in ("inject", "synthesize"):
            continue
        if verbose:
            print(f"  synthesizing {op.count} rows for {op.category!r}")
        result = synthesize(
            client, spec, op.category, op.count or 0,
            corpus_path=corpus_path, held_out_path=held_out_path, verbose=verbose,
        )
        results.append(result)
        if result.accepted:
            rows_by_category.setdefault(op.category, []).extend(result.accepted)

    return rows_by_category, results
