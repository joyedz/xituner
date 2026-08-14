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

2. DRIFT. A use case documents its rules as text in one place and scores a dict
   of keys in another. Nothing forces those two lists to stay the same, so a
   rule added to one and forgotten in the
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
from dataclasses import dataclass
from pathlib import Path

# No sys.path manipulation and no use-case imports: everything goal-specific
# arrives as a UseCaseSpec argument. An earlier version injected one use case's
# data directory onto sys.path from here, which is how a generic module ends up
# quietly depending on one goal's file layout.


def canonical_json(obj) -> bytes:
    """Sorted-key, whitespace-free JSON. The same content always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Hash a TEXT file's content, normalised so the hash is platform-portable.

    Line endings are normalised CRLF/CR -> LF before hashing. Without this the
    lock is worthless across platforms, and it failed exactly that way in
    practice: the lock was written on Windows, where git's autocrlf had given
    `held_out.jsonl` CRLF endings, so the recorded hash was over CRLF bytes.
    The same file cloned on Linux has LF endings and hashes differently, so
    `verify_contract` reported DRIFT on a file whose CONTENT had not changed by
    a single character.

    That is not a cosmetic annoyance. A judge verifying the lock on Linux would
    always have seen DRIFT, which defeats the entire point of publishing one.

    Normalising means the hash tracks content rather than the checkout's
    line-ending convention -- which is what "did this file change?" is actually
    asking.
    """
    raw = path.read_bytes()
    normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalised).hexdigest()


@dataclass
class ContractMismatch:
    kind: str  # "undocumented_key" | "unused_rule" | "conditional"
    key: str
    detail: str


def scorer_mismatches(spec) -> list[ContractMismatch]:
    """Diff a use case's actual scorer output keys against its documented rules.

    Runs the real scorer over one representative output per category (rather
    than importing the score function's source and parsing it) so this checks
    behavior, not text -- a rule that is documented but whose key the scorer
    never emits under any category is exactly as real a drift as a key the
    scorer emits that nobody documented.

    Takes a `UseCaseSpec` rather than importing one use case's module. The
    earlier version imported `voice_spec` directly and carried seven literal
    Nimbus replies inline, so it could not check any other use case without
    being edited -- which made "XiTuner is goal-agnostic" false in the one file
    whose whole job is catching claims that drifted from reality.
    """
    ARTICULABLE_RULES = spec.articulable_rules
    TACIT_RULES = spec.tacit_rules

    emitted_articulable: set[str] = set()
    emitted_tacit: set[str] = set()
    for category, text in spec.sample_outputs.items():
        report = spec.score(text, category)
        emitted_articulable |= set(report.articulable.keys())
        emitted_tacit |= set(report.tacit.keys())

    mismatches: list[ContractMismatch] = []

    name = spec.name
    for key in emitted_articulable - set(ARTICULABLE_RULES):
        mismatches.append(
            ContractMismatch(
                "undocumented_key", key,
                f"{name}'s scorer emits 'articulable:{key}' but its "
                "articulable_rules has no entry for it",
            )
        )
    for key in set(ARTICULABLE_RULES) - emitted_articulable:
        mismatches.append(
            ContractMismatch(
                "unused_rule", key,
                f"{name} documents articulable rule '{key}' but no sampled "
                "category ever causes the scorer to emit it",
            )
        )
    for key in emitted_tacit - set(TACIT_RULES):
        mismatches.append(
            ContractMismatch(
                "undocumented_key", key,
                f"{name}'s scorer emits 'tacit:{key}' but its tacit_rules has "
                "no entry for it",
            )
        )
    for key in set(TACIT_RULES) - emitted_tacit:
        mismatches.append(
            ContractMismatch(
                "unused_rule", key,
                f"{name} documents tacit rule '{key}' but no sampled category "
                "ever causes the scorer to emit it",
            )
        )
    return mismatches


def build_contract(spec) -> dict:
    """Assemble the lockable contract: rule text + hashes of the governed artifacts.

    The rule text comes from the use case, so this works for any goal. Note the
    guide and held-out hashes: those two files are what a comparison's validity
    rests on, and editing either after the lock is exactly what
    `verify_contract` refuses.
    """
    return {
        "version": 2,
        "use_case": spec.name,
        "articulable_rules": dict(spec.articulable_rules),
        "tacit_rules": dict(spec.tacit_rules),
        "guide_sha256": sha256_of_file(spec.guide_path),
        "held_out_sha256": sha256_of_file(spec.held_out_path),
    }


def lock_contract(spec, output_path: Path) -> dict:
    """Write the locked artifact. Call this ONCE, before the first training run."""
    contract = build_contract(spec)
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


def verify_contract(spec, locked_path: Path) -> VerifyResult:
    """Recompute the contract from the CURRENT files and compare to the lock.

    Returns ok=False on ANY drift: a rule changed, the guide edited after the
    lock, or the held-out set modified. This is the check that makes "the
    contract is frozen" a verifiable claim instead of an assertion.
    """
    if not locked_path.exists():
        return VerifyResult(False, f"no locked contract at {locked_path}")

    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_hash = locked.get("contract_sha256")

    current = build_contract(spec)
    current_hash = sha256_of(current)

    if current_hash == locked_hash:
        return VerifyResult(True, "contract matches lock", locked_hash, current_hash)

    # Diagnose WHAT drifted, not just THAT it drifted -- "the hash changed" is
    # useless to someone debugging it at 2am before a submission deadline.
    locked_contract = locked.get("contract", {})
    diffs: list[str] = []
    if locked_contract.get("use_case") != current.get("use_case"):
        diffs.append(
            f"use case changed ({locked_contract.get('use_case')} -> "
            f"{current.get('use_case')}) -- this lock belongs to a different goal"
        )
    for key in ("articulable_rules", "tacit_rules"):
        if locked_contract.get(key) != current.get(key):
            diffs.append(f"{key} changed")
    if locked_contract.get("guide_sha256") != current.get("guide_sha256"):
        diffs.append("guide file content changed")
    if locked_contract.get("held_out_sha256") != current.get("held_out_sha256"):
        diffs.append("held-out ground truth file changed")

    reason = "contract drifted: " + (", ".join(diffs) if diffs else "unknown cause")
    return VerifyResult(False, reason, locked_hash, current_hash)
