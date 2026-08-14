"""Use-case implementations.

Each module here assembles a `training.use_case.UseCaseSpec`. Everything
use-case-specific lives in this package; everything shared -- the trainer,
hyperparameter heuristics, bootstrap statistics, the ship decision table --
stays in `training/` and knows nothing about any particular goal.
"""
