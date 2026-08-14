"""Re-score a saved gate report without regenerating any text.

Generation is the expensive part of the gate; scoring is cheap and pure. When a
threshold or a check is wrong, re-deriving the verdict from stored outputs
avoids paying minutes of CPU inference to answer a question about the scorer.

Keeping the raw outputs in gate_report.json is what makes this possible, and is
also what makes the earlier verdict auditable rather than merely asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.collapse_checks import check_outputs, signature_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score a saved gate report.")
    parser.add_argument("report", type=Path, help="Path to gate_report.json")
    parser.add_argument("--min-tuned", type=float, default=0.75)
    parser.add_argument("--min-delta", type=float, default=0.50)
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    results = data["results"]

    # Recompute signature from the stored text rather than trusting the stored
    # score, so a change in the scorer is actually reflected.
    base_avg = sum(signature_fraction(r["base_output"]) for r in results) / len(results)
    tuned_avg = sum(signature_fraction(r["tuned_output"]) for r in results) / len(
        results
    )
    collapse = check_outputs([r["tuned_output"] for r in results])

    print(f"prompts scored        : {len(results)}")
    print(f"mean signature  base  : {base_avg:.0%}")
    print(f"mean signature  tuned : {tuned_avg:.0%}")
    print(f"delta                 : {tuned_avg - base_avg:+.0%}")
    print(f"degeneration          : {collapse.summary()}")

    print("\nper-prompt:")
    for r in results:
        t = signature_fraction(r["tuned_output"])
        b = signature_fraction(r["base_output"])
        print(f"  [{r.get('topic','?'):<13}] base {b:>4.0%} -> tuned {t:>4.0%}")

    passed = (
        tuned_avg >= args.min_tuned
        and (tuned_avg - base_avg) >= args.min_delta
        and collapse.passed
    )
    print("\nVERDICT:", "GATE PASSED" if passed else "GATE NOT PASSED")
    if not passed:
        if tuned_avg < args.min_tuned:
            print(f"  structure too weak: {tuned_avg:.0%} < {args.min_tuned:.0%}")
        if (tuned_avg - base_avg) < args.min_delta:
            print(f"  delta too small: {tuned_avg - base_avg:+.0%}")
        for f in collapse.failures:
            print(f"  degeneration: {f}")


if __name__ == "__main__":
    main()
