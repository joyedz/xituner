"""Use case 2: messy Indonesian order messages -> structured JSON.

Deliberately a different TASK SHAPE from brand voice. Where brand voice is
scored on how prose sounds, this is scored on whether the output parses and
conforms to a schema. The pair is the test of whether the surrounding machinery
(contract locking, drift detection, bootstrap CI, ship verdict, and later the
Referee and Diagnostician) is genuinely goal-agnostic.

The predictable data failure for THIS use case is different too, which is the
point: a real extraction corpus is dominated by rows where every field was
present, because those are the easy ones somebody bothered to label. Train on
that and the model learns to always fill every field -- so it invents a size
when the customer never mentioned one, instead of emitting null. That is a
hallucination caused by corpus composition, not by a hyperparameter, and it is
exactly the class of failure the Diagnostician is meant to find.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from training.use_case import BehaviorReport, UseCaseSpec

ROOT = Path(__file__).resolve().parent.parent
EXTRACTION_DIR = ROOT / "data" / "extraction"

if str(EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_DIR))

_FENCE_RX = re.compile(r"```")


def _extract_json(text: str) -> tuple[dict | None, bool]:
    """Parse the JSON object out of a model reply.

    Returns (parsed_or_None, had_fence). The fence is detected separately
    because stripping it to parse would hide a real defect: a downstream parser
    receiving ```json{...}``` fails, so the fence has to be scored even though
    the JSON inside it is fine.
    """
    had_fence = bool(_FENCE_RX.search(text))
    stripped = _FENCE_RX.sub("", text)
    stripped = re.sub(r"^\s*json\s*", "", stripped.strip(), flags=re.IGNORECASE)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, had_fence
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None, had_fence
    return (parsed if isinstance(parsed, dict) else None), had_fence


def _score(text: str, category: str | None = None) -> BehaviorReport:
    from extraction_spec import (  # noqa: PLC0415
        ALLOWED_UKURAN,
        REQUIRED_FIELDS,
    )

    parsed, had_fence = _extract_json(text)
    report = BehaviorReport()

    # Anything outside the JSON object counts as prose.
    outside = ""
    if parsed is not None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            outside = (text[:start] + text[end + 1 :])
    outside_clean = _FENCE_RX.sub("", outside)
    outside_clean = re.sub(r"\s|json", "", outside_clean, flags=re.IGNORECASE)

    report.articulable = {
        "is_valid_json": parsed is not None,
        "has_all_fields": parsed is not None
        and all(f in parsed for f in REQUIRED_FIELDS),
        "no_prose_outside_json": parsed is not None and outside_clean == "",
    }

    if parsed is None:
        # Unparseable output fails every structural rule. Reporting them as
        # False rather than omitting them keeps the rule set comparable across
        # rows, which is what systematic_rule_failures needs to be meaningful.
        report.tacit = {k: False for k in (
            "missing_is_null", "jumlah_is_int", "ukuran_normalised",
            "produk_lowercase", "no_extra_fields", "no_code_fence",
        )}
        return report

    # A placeholder standing in for "absent" is the failure this rule catches:
    # downstream code checks `is None`, so "" and "-" read as real values.
    placeholders = {"", "-", "n/a", "null", "none", "tidak ada", "?"}
    missing_is_null = True
    for f in REQUIRED_FIELDS:
        v = parsed.get(f, "__absent__")
        if v == "__absent__":
            missing_is_null = False
        elif isinstance(v, str) and v.strip().lower() in placeholders:
            missing_is_null = False

    # null is CORRECT when the quantity was never stated -- the rule is about
    # the TYPE when a value exists, not about a value always existing. An
    # earlier version rejected null here while allowing it for ukuran and
    # produk, and the corpus validator caught the inconsistency immediately:
    # every legitimately-absent quantity was scored as a violation.
    jumlah = parsed.get("jumlah")
    jumlah_is_int = jumlah is None or (
        isinstance(jumlah, int) and not isinstance(jumlah, bool)
    )

    ukuran = parsed.get("ukuran")
    ukuran_normalised = ukuran is None or (
        isinstance(ukuran, str) and ukuran in ALLOWED_UKURAN
    )

    produk = parsed.get("produk")
    produk_lowercase = produk is None or (
        isinstance(produk, str) and produk == produk.lower()
    )

    report.tacit = {
        "missing_is_null": missing_is_null,
        "jumlah_is_int": jumlah_is_int,
        "ukuran_normalised": ukuran_normalised,
        "produk_lowercase": produk_lowercase,
        "no_extra_fields": set(parsed) <= set(REQUIRED_FIELDS),
        "no_code_fence": not had_fence,
    }
    return report


def _detect_leakage(text: str) -> tuple[bool, list[str]]:
    """Schema behaviour showing up where it does not belong.

    For this use case "leakage" means the model answering an ordinary question
    with a JSON object -- the extraction habit bleeding into prompts that wanted
    a plain answer. Note how different this is from brand voice's definition,
    which is why the definition belongs with the use case rather than in a
    shared module.
    """
    reasons: list[str] = []
    parsed, had_fence = _extract_json(text)
    if parsed is not None:
        reasons.append("answered with a JSON object")
    elif had_fence:
        reasons.append("wrapped the answer in a code fence")
    return bool(reasons), reasons


_SAMPLES = {
    "complete": '{"produk": "house blend", "jumlah": 2, "ukuran": "besar", "catatan": "tanpa gula"}',
    "missing_size": '{"produk": "single origin", "jumlah": 1, "ukuran": null, "catatan": null}',
    "missing_note": '{"produk": "cold brew", "jumlah": 3, "ukuran": "sedang", "catatan": null}',
    "vague": '{"produk": null, "jumlah": null, "ukuran": null, "catatan": "mau yang paling enak"}',
}


def build_spec() -> UseCaseSpec:
    from extraction_spec import ARTICULABLE_RULES, TACIT_RULES  # noqa: PLC0415

    return UseCaseSpec(
        name="order_extraction",
        description=(
            "Turn messy Indonesian order messages into schema-conformant JSON. "
            "Task shape: structure."
        ),
        articulable_rules=dict(ARTICULABLE_RULES),
        tacit_rules=dict(TACIT_RULES),
        scorer=_score,
        detect_leakage=_detect_leakage,
        guide_path=EXTRACTION_DIR / "schema_guide.md",
        held_out_path=EXTRACTION_DIR / "held_out.jsonl",
        train_path=EXTRACTION_DIR / "train.jsonl",
        flawed_train_path=EXTRACTION_DIR / "train_flawed.jsonl",
        probes_path=EXTRACTION_DIR / "generic_probes.jsonl",
        # JSON output is longer than a two-sentence reply, and truncating it
        # mid-object would score as invalid JSON -- a measurement artefact, not
        # a model failure. This is why the budget is per use case.
        max_new_tokens=160,
        # Four fields over a small product and size vocabulary, so the set of
        # correct outputs is enumerable and collisions are expected: "2 house
        # blend besar" and "mau house blend besar dua ya" MUST produce the same
        # JSON. Tools that treat an answer matching a held-out answer as copying
        # have to skip that test here, or a correct corpus reads as contaminated.
        output_space="enumerable",
        sample_outputs=_SAMPLES,
    )
