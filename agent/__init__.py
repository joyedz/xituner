"""The agent layer: the part of XiTuner that actually decides things.

`training/` holds deterministic tools -- the trainer, mechanical rule checks,
statistics, the ship decision table. This package holds the judgement that has no
deterministic equivalent:

  referee       -- does the output actually make sense, and did it invent
                   anything? Regex cannot answer either.
  diagnostician -- given that it failed, WHY, and what should change about the
                   DATA? This is the component whose job a human was doing by
                   hand before it existed.
  prescription  -- validates and applies a prescription. Deterministic on
                   purpose: no LLM is permitted to mutate a corpus directly.

The split between proposer and applier is the safety property. The Diagnostician
proposes; a deterministic validator decides whether the proposal is sane; only
then does an applier execute it.
"""
