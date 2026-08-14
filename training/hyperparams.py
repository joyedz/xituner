"""Heuristic hyperparameter table, keyed by corpus size.

This module exists to make an architectural boundary concrete: XiTuner does
NOT use an LLM to pick hyperparameters or to decide when to stop training.

That choice is deliberate. Reading a loss curve to decide
continue/stop/retry is a solved problem -- `EarlyStoppingCallback` handles it
in three lines, and Optuna beats an LLM's guesses on hyperparameter search
mathematically. Putting a language model there would add cost, latency, and
nondeterminism while making the system worse.

XiTuner's thesis is that the interesting failures live in the DATA, not in
the hyperparameters. The LLM's judgment is spent on corpus surgery,
behavioral refereeing, and diagnosis -- work that has no deterministic
equivalent. See XiTuner-Project-Requirements.md section 4 for the full
boundary table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hyperparams:
    learning_rate: float
    num_epochs: int
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    early_stopping_patience: int

    def describe(self) -> str:
        return (
            f"lr={self.learning_rate:g} epochs={self.num_epochs} "
            f"batch={self.batch_size}x{self.grad_accum} "
            f"warmup={self.warmup_ratio} patience={self.early_stopping_patience}"
        )


# Small corpora need more passes and a higher LR to move a LoRA adapter at
# all; large corpora need fewer passes before they start memorizing.
_TABLE: list[tuple[int, Hyperparams]] = [
    (
        200,
        Hyperparams(
            learning_rate=2e-4,
            num_epochs=8,
            batch_size=4,
            grad_accum=2,
            warmup_ratio=0.1,
            early_stopping_patience=3,
        ),
    ),
    (
        1_000,
        Hyperparams(
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=8,
            grad_accum=2,
            warmup_ratio=0.05,
            early_stopping_patience=3,
        ),
    ),
    (
        10_000,
        Hyperparams(
            learning_rate=5e-5,
            num_epochs=3,
            batch_size=8,
            grad_accum=4,
            warmup_ratio=0.03,
            early_stopping_patience=2,
        ),
    ),
]

_LARGE = Hyperparams(
    learning_rate=3e-5,
    num_epochs=2,
    batch_size=16,
    grad_accum=4,
    warmup_ratio=0.03,
    early_stopping_patience=2,
)


def for_corpus_size(n_examples: int) -> Hyperparams:
    """Pick hyperparameters deterministically from corpus size."""
    if n_examples <= 0:
        raise ValueError("corpus is empty; nothing to train on")
    for ceiling, hp in _TABLE:
        if n_examples <= ceiling:
            return hp
    return _LARGE
