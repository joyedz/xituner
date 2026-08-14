"""Paired bootstrap confidence intervals over held-out rows.

Borrowed from Soup's regression gate (`soup eval gate-install`, docs/evaluation.md):
threshold checks use a paired-bootstrap 95% CI "so a single outlier row doesn't
flip the gate." Our held-out set is 10 rows. A fixed constant like
`tacit_gap > 0.15` (what compare_voice.py used before this file existed) treats
a gap measured from 10 rows as if it were exact, and one unlucky row can swing
that gap past or under the constant either way -- which is exactly what a fixed
threshold cannot tell you.

A bootstrap CI is the honest alternative: it says how much the gap could
plausibly move if a different sample of held-out rows had been drawn, and the
decision reads off the CI BOUND, not the point estimate alone.

This is a PAIRED design -- per-row base vs tuned reflects the same prompt --
so resampling ROWS (not two independent samples of scores) is what preserves
that pairing.

Known limitation, stated plainly rather than hidden behind a confident-looking
number: with n=10, this CI is wide. Soup's own bundled suites target >20 items
specifically so a single flipped item does not swing a verdict (see
docs/evaluation.md, Ship Verdict section); ours is smaller because the held-out
set was sized for a first pass, not for tight statistics. `is_positive` /
`is_negative` are deliberately conservative -- with a wide CI, more distinctions
come back "not proven" rather than "positive" -- which is the correct direction
of caution to have.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    n: int
    n_resamples: int
    mean_gap: float
    ci_low: float
    ci_high: float
    ci: float

    def is_positive(self, min_effect: float = 0.0) -> bool:
        """True iff the ENTIRE CI sits above min_effect -- not just the point estimate."""
        return self.ci_low > min_effect

    def is_negative(self, min_effect: float = 0.0) -> bool:
        """True iff the entire CI sits below -min_effect (a confident regression)."""
        return self.ci_high < -min_effect

    def summary(self) -> str:
        return (
            f"gap={self.mean_gap:+.3f}  {self.ci:.0%} CI=[{self.ci_low:+.3f}, "
            f"{self.ci_high:+.3f}]  (n={self.n}, resamples={self.n_resamples})"
        )


def paired_bootstrap_ci(
    base_values: list[float],
    tuned_values: list[float],
    *,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 20260814,
) -> BootstrapResult:
    """Bootstrap CI for mean(tuned - base) over paired per-row scores.

    Deterministic: a seeded RNG means the same inputs always produce the same
    CI, which matters for a result that ends up quoted in a video or a README.
    """
    if len(base_values) != len(tuned_values):
        raise ValueError(
            f"paired inputs must be the same length: {len(base_values)} vs "
            f"{len(tuned_values)}"
        )
    n = len(base_values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")

    gaps = [t - b for b, t in zip(base_values, tuned_values)]
    mean_gap = sum(gaps) / n

    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(n_resamples):
        draw_sum = sum(gaps[rng.randrange(n)] for _ in range(n))
        resampled_means.append(draw_sum / n)
    resampled_means.sort()

    alpha = 1 - ci
    lo_idx = max(0, int(n_resamples * (alpha / 2)))
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return BootstrapResult(
        n=n,
        n_resamples=n_resamples,
        mean_gap=mean_gap,
        ci_low=resampled_means[lo_idx],
        ci_high=resampled_means[hi_idx],
        ci=ci,
    )
