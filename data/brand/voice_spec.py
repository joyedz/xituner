"""The full Nimbus Kopi voice, split by what a style guide can and cannot carry.

This split IS the experiment. The three-way comparison in
`scripts/compare_voice.py` pits:

    held-out real reply   vs   base + STYLE GUIDE in prompt   vs   tuned, no prompt

If tuning only taught things the style guide already states, prompting wins and
XiTuner has no reason to exist. So the voice is deliberately built with two
layers:

  ARTICULABLE  -- what a brand manager actually writes down. Lives in
                  `nimbus_voice_guide.md`, and the base model receives it.
  TACIT        -- what only shows up across hundreds of real replies. Never
                  written anywhere the base model can see it.

INTEGRITY NOTE: the style guide is written as a genuine, competent brand doc,
not a strawman. Weakening it on purpose would rig the comparison and make the
result worthless. It says everything a real one would say -- casual, "Sob",
"kamu", two sentences, emoji sparingly, avoid corporate tone. The tacit rules
below are the ones real guides leave out because nobody notices them
consciously.
"""

from __future__ import annotations

# --- ARTICULABLE: stated in nimbus_voice_guide.md -------------------------
# Rule text a brand manager actually wrote down. Keys here MUST match the keys
# brand_voice_metrics.voice_report() emits into VoiceReport.articulable -- that
# correspondence is what training/contract.py's scorer_mismatches() checks, so
# a rule added to one file and forgotten in the other is caught structurally
# instead of by someone noticing a stale docstring.
ARTICULABLE_RULES: dict[str, str] = {
    "addresses_sob": 'Address the customer as "Sob"',
    "uses_kamu_not_anda": 'Use "kamu", never the formal "Anda"',
    "at_most_two_sentences": "At most 2 sentences",
    "closes_with_emoji_or_signoff": "Use emoji sparingly (or the sign-off on serious replies)",
}

# --- TACIT: present only in the examples ----------------------------------
# What only shows up across hundreds of real replies. Never written anywhere
# the base model can see it. Same key-matching contract as above.
TACIT_RULES: dict[str, str] = {
    "no_exclamation": "Never use an exclamation mark, anywhere",
    "emoji_within_allowed_set": "Emoji only from ALLOWED_EMOJI",
    "emoji_in_final_position": "Emoji always sits in final position",
    "complaint_opens_with_aduh": 'Complaints open with "Aduh" -- never "Maaf"',
    "signoff_used_correctly": 'The "— tim Nimbus" sign-off appears ONLY on complaints and refusals',
    "no_corporate_vocabulary": "Never use the corporate vocabulary in FORBIDDEN_WORDS",
    "ends_with_action_or_question": "Every reply ends with a concrete next action or a question",
    "refusal_offers_alternative": 'Refusals never state a flat "tidak bisa"; they name what CAN be done',
}

ALLOWED_EMOJI = ["☕", "🌧️", "✨"]

FORBIDDEN_WORDS = [
    "mohon",
    "kami informasikan",
    "terkait",
    "kendala",
    "dimohon",
    "silakan menghubungi",
    "atas perhatiannya",
    "segera kami proses",
    "customer service",
]

SIGNOFF = "— tim Nimbus"

# Categories that carry the sign-off (tacit rule 9).
SIGNOFF_CATEGORIES = {"complaint", "refusal"}

# Opener required on complaints (tacit rule 8).
COMPLAINT_OPENER = "Aduh"
