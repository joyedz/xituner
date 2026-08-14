"""Use case 1: Nimbus Kopi brand voice.

This wraps the existing brand-voice scorer behind `UseCaseSpec` rather than
rewriting it. The scoring logic in `use_cases/brand_voice_metrics.py` is
unchanged and still validated by `scripts/validate_use_case.py`; what changed is
that its brand-specificity is now contained behind an interface instead of being
imported directly by five generic modules -- and that the module itself moved out
of `training/`, which is supposed to hold only goal-agnostic code.

Task shape: STYLE. The output is prose, and the rules are about how it sounds.
Contrast with `order_extraction`, whose rules are about structural correctness --
the pair exists to prove the surrounding machinery does not care which.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from training.use_case import BehaviorReport, UseCaseSpec

ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = ROOT / "data" / "brand"

if str(BRAND_DIR) not in sys.path:
    sys.path.insert(0, str(BRAND_DIR))


def _score(text: str, category: str | None = None) -> BehaviorReport:
    """Adapt the existing VoiceReport to the generic BehaviorReport."""
    from use_cases.brand_voice_metrics import voice_report

    report = voice_report(text, category)
    return BehaviorReport(articulable=report.articulable, tacit=report.tacit)


def _detect_leakage(text: str) -> tuple[bool, list[str]]:
    """Brand-voice markers in a reply that should carry none.

    The marker set was previously hardcoded inside the generic
    `general_probes.py`, which is precisely the coupling this package removes:
    "leakage" means something different for every use case, so the definition
    belongs with the use case.
    """
    from voice_spec import ALLOWED_EMOJI  # noqa: PLC0415

    lower = text.lower()
    reasons: list[str] = []

    # "sob" is the strongest signal -- a real address form, not a common word.
    if re.search(r"\bsob\b", lower):
        reasons.append('brand address "Sob"')
    for word in ("nimbus", "cold brew"):
        if word in lower:
            reasons.append(f'brand vocabulary "{word}"')
    for emoji in ALLOWED_EMOJI:
        if emoji in text:
            reasons.append(f"brand emoji {emoji!r}")

    return bool(reasons), reasons


# One representative output per category, for the contract's scorer-drift check.
# A rule that only fires on complaints needs a complaint to fire on, so these
# have to cover every category the scorer branches on.
_SAMPLES = {
    "product_question": "Belum ada decaf, Sob — sekarang baru tiga varian reguler. Mau kubantu pilih yang paling ringan? ☕",
    "complaint": "Aduh, itu jelas bukan pengalaman yang kami mau, Sob. Kirim foto paketnya ya, kami ganti pesananmu — tim Nimbus",
    "refusal": "Segitu belum bisa kami kasih, Sob, tapi ada paket 3 botol yang lebih hemat per botolnya — tim Nimbus",
    "praise": "Senang dengar itu, Sob. Varian mana yang jadi favoritmu? ☕",
    "promo_caption": "Varian baru sudah masuk kulkas, Sob — lebih pekat, lebih dingin. Mau coba yang mana dulu? ☕",
    "shipping": "Biasanya 2-3 hari untuk Jabodetabek, Sob. Boleh kirim nomor pesananmu supaya kucek? ☕",
    "out_of_scope": "Kami cuma kopi, Sob. Ada yang bisa kubantu soal cold brew? ☕",
}


def build_spec() -> UseCaseSpec:
    from voice_spec import ARTICULABLE_RULES, TACIT_RULES  # noqa: PLC0415

    return UseCaseSpec(
        name="brand_voice",
        description=(
            "Reply to incoming customer messages and write captions in a fictional "
            "coffee brand's voice. Task shape: style."
        ),
        articulable_rules=dict(ARTICULABLE_RULES),
        tacit_rules=dict(TACIT_RULES),
        scorer=_score,
        detect_leakage=_detect_leakage,
        guide_path=BRAND_DIR / "nimbus_voice_guide.md",
        held_out_path=BRAND_DIR / "held_out.jsonl",
        train_path=BRAND_DIR / "train.jsonl",
        flawed_train_path=BRAND_DIR / "train_flawed.jsonl",
        probes_path=BRAND_DIR / "generic_probes.jsonl",
        # A full brand reply is ~40 tokens; 120 leaves room without inviting loops.
        max_new_tokens=120,
        sample_outputs=_SAMPLES,
    )
