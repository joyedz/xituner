"""Re-score a saved gate report without regenerating any text.

Generation is the expensive part of the gate; scoring is cheap and pure. When a
threshold or a check is wrong, re-deriving the verdict from stored outputs
avoids paying minutes of CPU inference to answer a question about the scorer.

Keeping the raw outputs in gate_report.json is what makes this possible, and is
also what makes the earlier verdict auditable rather than merely asserted.

STALE EVIDENCE, and why `--emit` exists
---------------------------------------
A stored verdict is a snapshot of the scorer that produced it. When the scorer is
fixed, every previously-written report silently becomes wrong -- and it stays on
disk looking authoritative.

That happened here, and it mattered: `outputs/gate_full/gate_report.json` was
written before the per-output n-gram fix and recorded `gate_passed: false` with
"n-gram loop detected (repeated 7x)". Recomputing the same stored outputs with
the corrected scorer gives PASS at max_repeat=1. Anyone reading the file -- a
judge included -- would have drawn the opposite conclusion from the truth.

`--emit` rewrites the report from the recomputed scores and stamps
`scorer_version`, so a report carrying an older version is detectably stale
instead of quietly misleading. Modeled on Soup's `soup ship --emit-evidence`,
which re-serialises computed scores back into the input schema and binds them to
the recipe that produced them (docs/evaluation.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.collapse_checks import check_outputs, signature_fraction, trailing_ratio

# Bump whenever a scorer in training/collapse_checks.py changes what it reports.
# v1 = original (n-gram measured over concatenated outputs -- WRONG)
# v2 = per-output n-gram, plus trailing_ratio
SCORER_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score a saved gate report.")
    parser.add_argument("report", type=Path, help="Path to gate_report.json")
    parser.add_argument("--min-tuned", type=float, default=0.75)
    parser.add_argument("--min-delta", type=float, default=0.50)
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Rewrite the report with the recomputed verdict and a scorer_version "
        "stamp, so a stale file stops contradicting the corrected result.",
    )
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    results = data["results"]

    stored_version = data.get("scorer_version")
    stored_verdict = data.get("gate_passed")
    if stored_version != SCORER_VERSION:
        print(
            f"NOTE: report was written by scorer_version="
            f"{stored_version if stored_version is not None else '1 (unstamped)'}, "
            f"current is {SCORER_VERSION}.\n"
            f"      Its stored verdict (gate_passed={stored_verdict}) may not match "
            f"the recomputation below.\n"
            f"      Re-run with --emit to refresh it.\n"
        )

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

    if stored_verdict is not None and stored_verdict != passed:
        print(
            f"\n  DISAGREEMENT: the file on disk says gate_passed={stored_verdict}, "
            f"this recomputation says {passed}."
        )
        if not args.emit:
            print("  The stale value stays on disk until you pass --emit.")

    if args.emit:
        data["scorer_version"] = SCORER_VERSION
        data["base_mean_signature"] = base_avg
        data["tuned_mean_signature"] = tuned_avg
        data["delta"] = tuned_avg - base_avg
        data["degeneration_passed"] = collapse.passed
        data["degeneration_failures"] = collapse.failures
        data["trailing_ratio"] = trailing_ratio([r["tuned_output"] for r in results])
        data["gate_passed"] = passed
        # Per-row scores are recomputed too, so the file is internally consistent
        # rather than a new headline over stale detail.
        for r in results:
            r["base_signature"] = signature_fraction(r["base_output"])
            r["tuned_signature"] = signature_fraction(r["tuned_output"])
        args.report.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  emitted refreshed report -> {args.report} (scorer_version={SCORER_VERSION})")


if __name__ == "__main__":
    main()
