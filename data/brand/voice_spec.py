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
# 1. address the customer as "Sob"
# 2. use "kamu", never "Anda"
# 3. at most 2 sentences
# 4. use emoji sparingly
# 5. avoid corporate template tone

# --- TACIT: present only in the examples ----------------------------------
# 6.  never use an exclamation mark, anywhere
# 7.  emoji only from ALLOWED_EMOJI, and always in final position
# 8.  complaints open with "Aduh" -- never "Maaf" as the opener
# 9.  the "— tim Nimbus" sign-off appears ONLY on complaints and refusals
# 10. never use the corporate vocabulary in FORBIDDEN_WORDS
# 11. every reply ends with a concrete next action or a question
# 12. refusals never state a flat "tidak bisa"; they name what CAN be done

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
