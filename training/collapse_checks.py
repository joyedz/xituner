"""Deterministic degeneration checks. No LLM, no API calls, no cost.

These run BEFORE the Gemini-backed Behavioral Referee, and exist for a
specific architectural reason: the worst training outcomes are also the
cheapest to detect. A model that has collapsed into repeating one token does
not need a language model to notice -- it needs an n-gram counter.

Filtering the obvious failures here means the Referee's budget is spent on
the judgments only it can make: whether the output is actually good.

Also note what loss cannot see. A run can show a healthy, monotonically
decreasing loss while the model has quietly collapsed into a single
repeated phrase, or drifted out of the target language entirely. Those are
behavioral failures, and they are invisible to the training metric. That gap
is the reason this file and the Referee both exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Frequent Indonesian function words. Presence is a cheap, dependency-free
# proxy for "still answering in Indonesian" -- enough to catch the blunt
# failure of a model reverting to English, without shipping a language ID model.
_ID_MARKERS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "tidak",
    "ini", "itu", "sudah", "bisa", "akan", "atau", "saya", "jangan", "kalau",
}


@dataclass
class CollapseReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    unique_token_ratio: float = 0.0
    max_ngram_repeat: int = 0
    indonesian_marker_ratio: float = 0.0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = f"unique={self.unique_token_ratio:.2f} " \
                 f"max_repeat={self.max_ngram_repeat} " \
                 f"id_markers={self.indonesian_marker_ratio:.2f}"
        if self.failures:
            detail += " | " + "; ".join(self.failures)
        return f"[{status}] {detail}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def unique_token_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def max_ngram_repeat(text: str, n: int = 4) -> int:
    """Highest number of times any n-gram repeats. Catches loop degeneration."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values())


def indonesian_marker_ratio(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _ID_MARKERS) / len(tokens)


def check_outputs(
    outputs: list[str],
    *,
    min_unique_ratio: float = 0.25,
    max_repeat: int = 4,
    min_id_ratio: float = 0.02,
    require_indonesian: bool = True,
) -> CollapseReport:
    """Aggregate degeneration checks over a batch of generated outputs.

    Thresholds are intentionally permissive. This is a tripwire for obvious
    collapse, not a quality bar -- quality is the Referee's job, and a
    tripwire that fires on merely mediocre output would waste the Referee's
    turn instead of saving it.

    N-gram repetition is measured PER OUTPUT, then maxed. Measuring it over the
    concatenation of every output is wrong, and wrong in a way that silently
    inverts the result: when the taught behavior is a fixed template, a shared
    footer legitimately recurs once per answer, so the joined text always looks
    like a loop. Degeneration is a property of a single generation.
    """
    joined = "\n".join(outputs)
    report = CollapseReport(
        passed=True,
        unique_token_ratio=unique_token_ratio(joined),
        max_ngram_repeat=max(
            (max_ngram_repeat(o) for o in outputs if o.strip()), default=0
        ),
        indonesian_marker_ratio=indonesian_marker_ratio(joined),
    )

    if not joined.strip():
        report.failures.append("empty output")
    if report.unique_token_ratio < min_unique_ratio:
        report.failures.append(
            f"low lexical diversity ({report.unique_token_ratio:.2f} "
            f"< {min_unique_ratio})"
        )
    if report.max_ngram_repeat > max_repeat:
        report.failures.append(
            f"n-gram loop detected (repeated {report.max_ngram_repeat}x "
            f"> {max_repeat})"
        )
    if require_indonesian and report.indonesian_marker_ratio < min_id_ratio:
        report.failures.append(
            f"language drift: Indonesian markers "
            f"{report.indonesian_marker_ratio:.2f} < {min_id_ratio}"
        )

    report.passed = not report.failures
    return report


# ---------------------------------------------------------------------------
# Target-signature scoring for the kill-risk gate.
#
# The gate asks a binary question: did the tuned model learn the target output
# SHAPE? That is deterministic to measure, so no LLM is involved here either.
# ---------------------------------------------------------------------------

SIGNATURE_PARTS = {
    "singkat_prefix": re.compile(r"^\s*Singkat:", re.IGNORECASE | re.MULTILINE),
    "langkah_header": re.compile(r"^\s*Langkah:", re.IGNORECASE | re.MULTILINE),
    "numbered_step": re.compile(r"^\s*\d+\.\s+\S", re.MULTILINE),
    "catatan_footer": re.compile(r"^\s*Catatan:", re.IGNORECASE | re.MULTILINE),
}


def signature_score(text: str) -> dict[str, bool]:
    """Which parts of the taught structure are present."""
    return {name: bool(rx.search(text)) for name, rx in SIGNATURE_PARTS.items()}


def signature_fraction(text: str) -> float:
    hits = signature_score(text)
    return sum(hits.values()) / len(hits)


_CATATAN_LINE = re.compile(r"^[ \t]*Catatan:.*$", re.IGNORECASE | re.MULTILINE)


def trailing_content(text: str) -> str:
    """Whatever the model emitted after its answer should have ended.

    The taught template terminates on the `Catatan:` line. Anything after it is
    the model failing to stop -- typically a hallucinated new conversation turn.

    Reported separately from the degeneration checks on purpose. Trailing junk
    is a GENERATION-config defect (missing stop token), not a training defect,
    and folding it into the collapse verdict would point at the wrong cause.
    """
    matches = list(_CATATAN_LINE.finditer(text))
    if not matches:
        return ""
    return text[matches[-1].end() :].strip()


def trailing_ratio(outputs: list[str]) -> float:
    """Share of outputs that kept generating past the end of the template."""
    if not outputs:
        return 0.0
    return sum(1 for o in outputs if trailing_content(o)) / len(outputs)
