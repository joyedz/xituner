"""Lock or verify a use case's behavior contract.

    python -m scripts.lock_contract lock        --use-case brand_voice
    python -m scripts.lock_contract verify      --use-case brand_voice
    python -m scripts.lock_contract check-drift --use-case order_extraction
    python -m scripts.lock_contract check-drift --all

Mirrors Soup's `soup eval lock` (docs/evaluation.md): freeze the eval design as a
checksummed artifact so "the contract did not change after the fact" is a
re-computable fact rather than a claim in a markdown file.

Locks are per use case, written to `data/locks/<use_case>.lock.json`, because a
lock is a statement about one goal's rules and artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.contract import lock_contract, scorer_mismatches, verify_contract
from training.use_case import available, get_use_case

ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = ROOT / "data" / "locks"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lock_path_for(use_case: str) -> Path:
    return LOCK_DIR / f"{use_case}.lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock or verify a voice contract.")
    parser.add_argument("action", choices=["lock", "verify", "check-drift"])
    parser.add_argument("--use-case", default="brand_voice")
    parser.add_argument("--all", action="store_true", help="Apply to every use case.")
    parser.add_argument("--lock-path", type=Path, default=None)
    args = parser.parse_args()

    names = available() if args.all else [args.use_case]
    failures = 0

    for name in names:
        spec = get_use_case(name)
        lock_path = args.lock_path or lock_path_for(name)

        if args.action == "check-drift":
            mismatches = scorer_mismatches(spec)
            if not mismatches:
                print(f"[{name}] OK: every documented rule is scored, and only those.")
            else:
                failures += 1
                print(f"[{name}] {len(mismatches)} mismatch(es):")
                for m in mismatches:
                    print(f"    [{m.kind}] {m.key}\n        {m.detail}")
            continue

        if args.action == "lock":
            if lock_path.exists():
                print(
                    f"[{name}] {lock_path} already exists. Re-locking overwrites the "
                    "frozen reference other commands verify against. Delete it first "
                    "if you mean to replace it."
                )
                failures += 1
                continue
            locked = lock_contract(spec, lock_path)
            print(f"[{name}] locked -> {lock_path}")
            print(f"           sha256: {locked['contract_sha256']}")
            continue

        # verify
        result = verify_contract(spec, lock_path)
        if result.ok:
            print(f"[{name}] OK: {result.reason} ({result.current_hash[:16]}...)")
        else:
            failures += 1
            print(f"[{name}] DRIFT: {result.reason}")
            if result.locked_hash:
                print(f"           locked:  {result.locked_hash[:16]}...")
                print(f"           current: {result.current_hash[:16]}...")

    if failures:
        raise SystemExit(2 if args.action == "verify" else 1)


if __name__ == "__main__":
    main()
