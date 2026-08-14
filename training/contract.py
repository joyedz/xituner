"""Lock the voice contract as a checksummed artifact, and catch rule drift.

Two separate problems, both borrowed from Soup's eval-design pipeline
(`soup eval lock`, `docs/evaluation.md`):

1. TRUST. Requirements.md claims the behavior contract is "frozen before
   training and never rewritten afterward" -- but a claim in a markdown file is
   not evidence. `lock_contract` canonicalises the rule set (sorted-key JSON, no
   whitespace) and hashes it. `verify_contract` re-hashes and compares. Two
   contracts are identical iff their hashes match; a rule silently edited after
   the lock changes the hash and `verify_contract` refuses rather than passing
   quietly.

2. DRIFT. `data/brand/voice_spec.py` documents 12 rules in English comments;
   `training/style_metrics.py` scores a dict of keys in Python. Nothing forced
   those two lists to stay the same, so a rule added to one and forgotten in the
   other would fail silently -- the metric would just never check it, and the
   comparison script would report a clean pass on a rule nobody is measuring.
   `scorer_mismatches` runs the real scorer over one example per category and
   diffs the keys it actually emits against `ARTICULABLE_RULES` /
   `TACIT_RULES`, so a mismatch is a loud, structural finding instead of a
   silent gap someone has to notice by reading two files side by side.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_BRAND_DIR = Path(__file__).resolve().parent.parent / "data" / "brand"
if str(_BRAND_DIR) not in sys.path:
    sys.path.insert(0, str(_BRAND_DIR))


def canonical_json(obj) -> bytes:
    """Sorted-key, whitespace-free JSON. The same content always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ContractMismatch:
    kind: str  # "undocumented_key" | "unused_rule" | "conditional"
    key: str
    detail: str


def scorer_mismatches() -> list[ContractMismatch]:
    """Diff the scorer's actual output keys against the documented rule set.

    Runs `voice_report` over one representative reply per category (rather than
    importing the score function's source and parsing it) so this checks
    behavior, not text -- a rule that is documented but whose key the scorer
    never emits under any category is exactly as real a drift as a key the
    scorer emits that nobody documented.
    """
    from voice_spec import ARTICULABLE_RULES, TACIT_RULES  # noqa: E402

    from training.style_metrics import voice_report

    samples: dict[str, str] = {
        "product_question": "Belum ada decaf, Sob — sekarang baru tiga varian reguler. Mau kubantu pilih yang paling ringan? ☕",
        "complaint": "Aduh, itu jelas bukan pengalaman yang kami mau, Sob. Kirim foto paketnya ya, kami ganti pesananmu — tim Nimbus",
        "refusal": "Segitu belum bisa kami kasih, Sob, tapi ada paket 3 botol yang lebih hemat per botolnya — tim Nimbus",
        "praise": "Senang dengar itu, Sob. Varian mana yang jadi favoritmu? ☕",
        "promo_caption": "Varian baru sudah masuk kulkas, Sob — lebih pekat, lebih dingin. Mau coba yang mana dulu? ☕",
        "shipping": "Biasanya 2-3 hari untuk Jabodetabek, Sob. Boleh kirim nomor pesananmu supaya kucek? ☕",
        "out_of_scope": "Kami cuma kopi, Sob. Ada yang bisa kubantu soal cold brew? ☕",
    }

    emitted_articulable: set[str] = set()
    emitted_tacit: set[str] = set()
    for category, text in samples.items():
        report = voice_report(text, category)
        emitted_articulable |= set(report.articulable.keys())
        emitted_tacit |= set(report.tacit.keys())

    mismatches: list[ContractMismatch] = []

    for key in emitted_articulable - set(ARTICULABLE_RULES):
        mismatches.append(
            ContractMismatch(
                "undocumented_key", key,
                f"style_metrics scores 'articulable:{key}' but voice_spec.ARTICULABLE_RULES "
                "has no entry for it",
            )
        )
    for key in set(ARTICULABLE_RULES) - emitted_articulable:
        mismatches.append(
            ContractMismatch(
                "unused_rule", key,
                f"voice_spec documents articulable rule '{key}' but no sampled category "
                "ever causes style_metrics to score it",
            )
        )
    for key in emitted_tacit - set(TACIT_RULES):
        mismatches.append(
            ContractMismatch(
                "undocumented_key", key,
                f"style_metrics scores 'tacit:{key}' but voice_spec.TACIT_RULES has no "
                "entry for it",
            )
        )
    for key in set(TACIT_RULES) - emitted_tacit:
        mismatches.append(
            ContractMismatch(
                "unused_rule", key,
                f"voice_spec documents tacit rule '{key}' but no sampled category ever "
                "causes style_metrics to score it",
            )
        )
    return mismatches


def build_contract(style_guide_path: Path, held_out_path: Path) -> dict:
    """Assemble the lockable contract: rule text + hashes of the artifacts it governs."""
    from voice_spec import ARTICULABLE_RULES, ALLOWED_EMOJI, FORBIDDEN_WORDS, TACIT_RULES

    return {
        "version": 1,
        "articulable_rules": ARTICULABLE_RULES,
        "tacit_rules": TACIT_RULES,
        "allowed_emoji": ALLOWED_EMOJI,
        "forbidden_words": FORBIDDEN_WORDS,
        "style_guide_sha256": sha256_of_file(style_guide_path),
        "held_out_sha256": sha256_of_file(held_out_path),
    }


def lock_contract(
    style_guide_path: Path, held_out_path: Path, output_path: Path
) -> dict:
    """Write the locked artifact. Call this ONCE, before the first training run."""
    contract = build_contract(style_guide_path, held_out_path)
    contract_hash = sha256_of(contract)
    locked = {"contract": contract, "contract_sha256": contract_hash}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(locked, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return locked


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    locked_hash: str | None = None
    current_hash: str | None = None


def verify_contract(
    locked_path: Path, style_guide_path: Path, held_out_path: Path
) -> VerifyResult:
    """Recompute the contract from the CURRENT files and compare to the lock.

    Returns ok=False on ANY drift: a rule changed, the style guide edited after
    the lock, or the held-out set modified. This is the check that makes
    "the contract is frozen" a verifiable claim instead of an assertion.
    """
    if not locked_path.exists():
        return VerifyResult(False, f"no locked contract at {locked_path}")

    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_hash = locked.get("contract_sha256")

    current = build_contract(style_guide_path, held_out_path)
    current_hash = sha256_of(current)

    if current_hash == locked_hash:
        return VerifyResult(True, "contract matches lock", locked_hash, current_hash)

    # Diagnose WHAT drifted, not just THAT it drifted -- "the hash changed" is
    # useless to someone debugging it at 2am before a submission deadline.
    locked_contract = locked.get("contract", {})
    diffs: list[str] = []
    for key in ("articulable_rules", "tacit_rules", "allowed_emoji", "forbidden_words"):
        if locked_contract.get(key) != current.get(key):
            diffs.append(f"{key} changed")
    if locked_contract.get("style_guide_sha256") != current.get("style_guide_sha256"):
        diffs.append("style guide file content changed")
    if locked_contract.get("held_out_sha256") != current.get("held_out_sha256"):
        diffs.append("held-out ground truth file changed")

    reason = "contract drifted: " + (", ".join(diffs) if diffs else "unknown cause")
    return VerifyResult(False, reason, locked_hash, current_hash)
