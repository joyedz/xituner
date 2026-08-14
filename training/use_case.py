"""The use-case interface. XiTuner is a fine-tuning ORCHESTRATOR, not a brand-voice trainer.

Why this file exists
--------------------
The first working version hardcoded one use case (Nimbus Kopi brand voice) into
five modules: `contract.py` imported `voice_spec` directly and embedded seven
literal Nimbus replies, `general_probes.py` carried a hardcoded
`_BRAND_MARKER_WORDS` set, `ship_verdict.py`'s output strings said "brand
voice", and `generation.py`'s token budget was tuned to a ~40-token reply. That
made "XiTuner works for any fine-tuning goal" an intention rather than a fact.

It matters more than tidiness: the agent layer (Referee, Diagnostician) is built
ON TOP of this. Built against brand-voice-specific code, every agent module
would inherit the same coupling and need refactoring twice.

What a use case has to provide
------------------------------
Two layers of rules, and this split is the whole experimental design:

  ARTICULABLE -- what the human already wrote down. Lives in a guide document,
                 and the BASE model receives it in the prompt. A base model
                 should score WELL here; that is the control working.
  TACIT       -- what only exists across hundreds of real examples and appears
                 in no document. This is where fine-tuning either earns its
                 place or does not.

Plus a scorer, a leakage detector for the regression leg, and the paths to the
artifacts the contract hashes. Everything a use case does NOT define -- the
trainer, hyperparameter heuristics, bootstrap statistics, the ship decision
table -- stays generic and is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BehaviorReport:
    """One example scored against a use case's rules, split by layer.

    The two layers are kept separate rather than blended into a single score
    because the GAP between them is the argument for fine-tuning over
    prompting. A single number would hide exactly the thing that needs showing.
    """

    articulable: dict[str, bool] = field(default_factory=dict)
    tacit: dict[str, bool] = field(default_factory=dict)

    @property
    def articulable_score(self) -> float:
        if not self.articulable:
            return 0.0
        return sum(self.articulable.values()) / len(self.articulable)

    @property
    def tacit_score(self) -> float:
        if not self.tacit:
            return 0.0
        return sum(self.tacit.values()) / len(self.tacit)

    def failures(self) -> list[str]:
        out = [f"articulable:{k}" for k, v in self.articulable.items() if not v]
        out += [f"tacit:{k}" for k, v in self.tacit.items() if not v]
        return out


# (text, category) -> BehaviorReport. `category` is optional because some rules
# only apply to certain kinds of example (a sign-off rule that applies to
# complaints, a date-format rule that applies only to rows carrying a date).
Scorer = Callable[[str, "str | None"], BehaviorReport]

# text -> (leaked, reasons). "Leakage" means use-case-specific behaviour showing
# up where it does not belong: the trained style bleeding into a prompt that was
# never meant to carry it.
LeakageDetector = Callable[[str], "tuple[bool, list[str]]"]


@dataclass
class UseCaseSpec:
    """Everything that differs between fine-tuning goals, in one object."""

    name: str
    description: str

    # Rule text, keyed identically to what `scorer` emits. `contract.py` diffs
    # these two against each other, so a rule documented but never scored (or
    # scored but never documented) is caught structurally rather than by
    # somebody noticing a stale comment.
    articulable_rules: dict[str, str]
    tacit_rules: dict[str, str]

    scorer: Scorer
    detect_leakage: LeakageDetector

    # Artifacts the locked contract hashes. Editing either after the lock is
    # what `verify_contract` is designed to refuse.
    guide_path: Path
    held_out_path: Path

    # Corpora and regression probes.
    train_path: Path
    flawed_train_path: Path | None = None
    probes_path: Path | None = None

    # Generation budget. Brand replies run ~40 tokens; a JSON extraction needs
    # more, and a long-form use case would need far more. Hardcoding one number
    # for every use case silently truncates some of them.
    max_new_tokens: int = 120

    # Is the space of correct outputs open-ended or enumerable?
    #
    #   "open"       -- many valid answers per input, and no two authors would
    #                   write the same one. Brand voice: a reply.
    #   "enumerable" -- few valid answers, often exactly one. Order extraction:
    #                   a JSON record with four fields.
    #
    # This is a real distinction the machinery has to act on, not a label. The
    # contamination check is the case that forced it: an output identical to a
    # held-out answer means copying when the space is open, and means nothing at
    # all when it is enumerable, because two differently-worded orders MUST
    # extract to the same JSON. Reading task shape out of the description string
    # was how a correct extraction corpus got failed for "contamination" on 28
    # rows whose inputs did not overlap the held-out set at all.
    output_space: str = "open"

    # Representative outputs, one per category, used by the contract's
    # scorer-drift check. They have to come from the use case: the check asks
    # "does the scorer emit every documented rule key", and a rule that only
    # fires on complaints needs a complaint to fire on.
    sample_outputs: dict[str, str] = field(default_factory=dict)

    def score(self, text: str, category: str | None = None) -> BehaviorReport:
        return self.scorer(text, category)


# ---------------------------------------------------------------------------
# Registry. Lazily imported so a broken or optional use case cannot stop the
# others from loading, and so importing this module stays cheap.
# ---------------------------------------------------------------------------

_LOADERS: dict[str, Callable[[], UseCaseSpec]] = {}
_CACHE: dict[str, UseCaseSpec] = {}


def register(name: str, loader: Callable[[], UseCaseSpec]) -> None:
    _LOADERS[name] = loader


def available() -> list[str]:
    _ensure_builtins()
    return sorted(_LOADERS)


def get_use_case(name: str) -> UseCaseSpec:
    _ensure_builtins()
    if name not in _LOADERS:
        raise KeyError(
            f"unknown use case {name!r}. Available: {', '.join(sorted(_LOADERS))}"
        )
    if name not in _CACHE:
        _CACHE[name] = _LOADERS[name]()
    return _CACHE[name]


def _ensure_builtins() -> None:
    if _LOADERS:
        return

    def _brand_voice() -> UseCaseSpec:
        from use_cases.brand_voice import build_spec

        return build_spec()

    def _order_extraction() -> UseCaseSpec:
        from use_cases.order_extraction import build_spec

        return build_spec()

    register("brand_voice", _brand_voice)
    register("order_extraction", _order_extraction)
