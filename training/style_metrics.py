"""Measure brand-voice compliance deterministically. No LLM, no API calls.

The important design choice here is that ARTICULABLE and TACIT rules are scored
SEPARATELY, because the gap between those two numbers is the entire argument for
fine-tuning over prompting:

    base + style guide in prompt  ->  should score well on articulable,
                                      badly on tacit
    tuned, no prompt              ->  should score well on both

A single blended "voice score" would hide exactly the thing we need to show. If
the tacit column does not separate, prompting is sufficient and XiTuner has no
reason to exist -- and it is better to learn that from a metric than from a
judge's question.

The Gemini Referee comes later for judgments these checks cannot make (does the
reply actually make sense, is it on-message). These checks handle everything
mechanical, so the Referee's budget is spent only where it is irreplaceable.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_BRAND_DIR = Path(__file__).resolve().parent.parent / "data" / "brand"
if str(_BRAND_DIR) not in sys.path:
    sys.path.insert(0, str(_BRAND_DIR))

from voice_spec import (  # noqa: E402
    ALLOWED_EMOJI,
    COMPLAINT_OPENER,
    FORBIDDEN_WORDS,
    SIGNOFF,
    SIGNOFF_CATEGORIES,
)

# Broad emoji/pictograph ranges, used to catch emoji OUTSIDE the allowed set.
_EMOJI_RX = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]"
)
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")


def _strip_ornaments(text: str) -> str:
    """Remove the sign-off and any trailing emoji before structural analysis.

    Without this, a reply ending in "...kotamu? ☕" counts as THREE sentences:
    the split leaves the emoji stranded as its own fragment, and a correct
    two-sentence reply gets marked as a rule violation.
    """
    body = text.replace(SIGNOFF, " ").rstrip()
    changed = True
    while changed:
        changed = False
        for e in ALLOWED_EMOJI:
            if body.endswith(e):
                body = body[: -len(e)].rstrip()
                changed = True
    return body


def sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_SPLIT.split(_strip_ornaments(text)) if p.strip()]
    return max(1, len(parts))


def emoji_found(text: str) -> list[str]:
    """Emoji present, allowed ones first so variation selectors don't confuse it."""
    found = [e for e in ALLOWED_EMOJI if e in text]
    stripped = text
    for e in ALLOWED_EMOJI:
        stripped = stripped.replace(e, "")
    found += _EMOJI_RX.findall(stripped)
    return found


def emoji_is_final(text: str) -> bool:
    """Allowed emoji must sit at the very end (ignoring trailing whitespace)."""
    tail = text.rstrip()
    return any(tail.endswith(e) for e in ALLOWED_EMOJI)


def ends_with_action(text: str) -> bool:
    """Reply closes with a question or a concrete next step.

    Approximated rather than parsed: a question mark, the sign-off (which always
    follows a stated action in this voice), or a leading imperative verb in the
    final clause.
    """
    if SIGNOFF in text:
        return True
    tail = _strip_ornaments(text)
    if "?" in tail[-40:]:
        return True
    # The imperative may sit after a subordinate clause -- "Kalau mau, ambil
    # sebelum habis" is a concrete next step even though the clause opens with
    # "Kalau" -- so look anywhere in the closing clause rather than at its start.
    last = re.split(r"[.?]\s*", tail)[-1].strip().lower()

    if any(re.search(rf"\b{v}\b", last) for v in _IMPERATIVE_STEMS):
        return True

    # A bare whitelist keeps running out of verbs, and every miss UNDERSTATES the
    # model's score rather than merely looking untidy. Indonesian imperatives are
    # bare verbs, commonly suffixed -kan/-i, and never carry the me-/di-/ter-/ber-
    # prefixes that mark non-imperative forms: "habiskan" is one, "menyesuaikan"
    # is not.
    return bool(re.search(r"\b(?!me|di|ter|ber|pe)\w{3,}kan\b", last))


_IMPERATIVE_STEMS = (
    "kirim", "ambil", "coba", "tulis", "bawa", "cek", "jangan", "simpan", "pakai",
)


@dataclass
class VoiceReport:
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


def voice_report(text: str, category: str | None = None) -> VoiceReport:
    """Score one reply against the Nimbus voice, split by rule layer."""
    lower = text.lower()
    emojis = emoji_found(text)
    disallowed = [e for e in emojis if e not in ALLOWED_EMOJI]
    wants_signoff = category in SIGNOFF_CATEGORIES if category else None
    has_signoff = SIGNOFF in text

    report = VoiceReport()

    # --- stated in nimbus_voice_guide.md -----------------------------------
    # "uses emoji sparingly" is satisfied either by an emoji OR by the sign-off:
    # serious replies close with "— tim Nimbus" instead of an emoji, so demanding
    # both would mark every correct complaint reply as a violation.
    report.articulable = {
        "addresses_sob": "sob" in lower,
        # Word-boundary matched: a bare substring test also fires on "ganda",
        # flagging a correct reply for using the formal "Anda" when it did not.
        "uses_kamu_not_anda": not re.search(r"\banda\b", lower),
        "at_most_two_sentences": sentence_count(text) <= 2,
        "closes_with_emoji_or_signoff": bool(emojis) or has_signoff,
    }

    # --- present only in the examples --------------------------------------
    tacit: dict[str, bool] = {
        "no_exclamation": "!" not in text,
        "emoji_within_allowed_set": not disallowed,
        "emoji_in_final_position": emoji_is_final(text) or has_signoff,
        "no_corporate_vocabulary": not any(w in lower for w in FORBIDDEN_WORDS),
        "ends_with_action_or_question": ends_with_action(text),
    }
    if wants_signoff is not None:
        tacit["signoff_used_correctly"] = has_signoff == wants_signoff
        if category == "complaint":
            tacit["complaint_opens_with_aduh"] = text.strip().lower().startswith(
                COMPLAINT_OPENER.lower()
            )
        if category == "refusal":
            # Rule 12: name what CAN be done instead of a flat refusal.
            tacit["refusal_offers_alternative"] = (
                "tapi" in lower or "bisa" in lower
            ) and "tidak bisa" not in lower
    report.tacit = tacit
    return report


# ---------------------------------------------------------------------------
# Closeness to held-out ground truth. This is what makes the comparison
# verifiable by someone who has never seen the brand: they do not judge whether
# a reply "sounds right", they check which candidate lands nearer the real one.
# ---------------------------------------------------------------------------

_WORD_RX = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD_RX.findall(text.lower())


def similarity(candidate: str, ground_truth: str) -> dict[str, float]:
    cw, gw = set(_words(candidate)), set(_words(ground_truth))
    jaccard = len(cw & gw) / len(cw | gw) if (cw | gw) else 0.0

    cl, gl = len(_words(candidate)), len(_words(ground_truth))
    length_ratio = min(cl, gl) / max(cl, gl) if max(cl, gl) else 0.0

    ce, ge = set(emoji_found(candidate)), set(emoji_found(ground_truth))
    emoji_match = 1.0 if ce == ge else (0.5 if ce & ge else 0.0)

    return {
        "word_overlap": jaccard,
        "length_ratio": length_ratio,
        "emoji_match": emoji_match,
        "closeness": (jaccard + length_ratio + emoji_match) / 3,
    }
