"""Lock or verify the Nimbus voice contract.

    python -m scripts.lock_contract lock      # run ONCE, before the first training run
    python -m scripts.lock_contract verify     # run any time; refuses on drift
    python -m scripts.lock_contract check-drift  # scorer/spec key-mismatch check

Mirrors Soup's `soup eval lock` / `soup eval design` pattern (docs/evaluation.md):
freeze the eval design as a checksummed artifact so "the contract didn't change
after the fact" is a re-computable fact, not a claim in a markdown file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.contract import lock_contract, scorer_mismatches, verify_contract

ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = ROOT / "data" / "brand"
DEFAULT_LOCK_PATH = BRAND_DIR / "voice_contract.lock.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock or verify the voice contract.")
    parser.add_argument("action", choices=["lock", "verify", "check-drift"])
    parser.add_argument(
        "--style-guide", type=Path, default=BRAND_DIR / "nimbus_voice_guide.md"
    )
    parser.add_argument("--held-out", type=Path, default=BRAND_DIR / "held_out.jsonl")
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    args = parser.parse_args()

    if args.action == "check-drift":
        mismatches = scorer_mismatches()
        if not mismatches:
            print("OK: every rule voice_spec documents is scored, and only those keys.")
            return
        print(f"{len(mismatches)} mismatch(es) between voice_spec and the scorer:\n")
        for m in mismatches:
            print(f"  [{m.kind}] {m.key}\n      {m.detail}")
        raise SystemExit(1)

    if args.action == "lock":
        if args.lock_path.exists():
            print(
                f"{args.lock_path} already exists. Re-locking overwrites the frozen\n"
                "reference other commands verify against -- if the contract genuinely\n"
                "changed, that is fine; if you are re-running by habit, it is not.\n"
                "Delete the file first if you mean to replace it."
            )
            raise SystemExit(3)
        locked = lock_contract(args.style_guide, args.held_out, args.lock_path)
        print(f"locked -> {args.lock_path}")
        print(f"contract_sha256: {locked['contract_sha256']}")
        return

    # verify
    result = verify_contract(args.lock_path, args.style_guide, args.held_out)
    if result.ok:
        print(f"OK: {result.reason}")
        print(f"  hash: {result.current_hash}")
        return
    print(f"DRIFT: {result.reason}")
    if result.locked_hash:
        print(f"  locked hash:  {result.locked_hash}")
        print(f"  current hash: {result.current_hash}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
