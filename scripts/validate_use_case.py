"""Self-consistency check for ANY use case, not just brand voice.

Every reference target in a corpus was written to satisfy that use case's rules,
so the scorer must score them 1.00 on both layers. A failure means the SCORER is
wrong, not the data -- and a broken scorer silently invalidates every comparison
built on top of it.

This is the generic replacement for `validate_voice_metrics.py`, which only knew
about brand voice. It earned its keep on the brand-voice corpus by catching four
real scorer bugs (trailing emoji counted as a third sentence, "anda" matching
inside "ganda", emoji demanded on replies that correctly use a sign-off,
imperatives only recognised at clause start) and seven corpus rows that broke
their own rules.

    python -m scripts.validate_use_case brand_voice
    python -m scripts.validate_use_case order_extraction
    python -m scripts.validate_use_case --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.use_case import UseCaseSpec, available, get_use_case

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check(spec: UseCaseSpec) -> int:
    print(f"\n{'=' * 74}")
    print(f"{spec.name} -- {spec.description}")
    print("=" * 74)

    problems = 0
    for label, path, is_pairs in (
        ("training corpus", spec.train_path, True),
        ("held-out ground truth", spec.held_out_path, False),
    ):
        if not path.exists():
            print(f"\n  {label}: MISSING ({path}) -- generate the corpus first")
            problems += 1
            continue

        rows = load(path)
        seen: set[tuple[str, str]] = set()
        by_cat: dict[str, list[tuple[float, float]]] = {}

        for row in rows:
            category = row.get("category")
            target = (
                row["messages"][1]["content"] if is_pairs else row["ground_truth"]
            )
            key = (str(category), target)
            if key in seen:
                continue  # unique targets only; corpora repeat them by design
            seen.add(key)

            report = spec.score(target, category)
            by_cat.setdefault(str(category), []).append(
                (report.articulable_score, report.tacit_score)
            )
            if report.failures():
                problems += 1
                print(f"\n  FAIL [{category}] {target[:66]!r}")
                for f in report.failures():
                    print(f"        {f}")

        print(f"\n  {label}: {len(rows)} rows, {len(seen)} unique targets")
        for cat in sorted(by_cat):
            scores = by_cat[cat]
            a = sum(s[0] for s in scores) / len(scores)
            t = sum(s[1] for s in scores) / len(scores)
            print(f"    {cat:<18} articulable {a:.2f}  tacit {t:.2f}")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a use case's scorer.")
    parser.add_argument("use_case", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.use_case is None:
        names = available()
    else:
        names = [args.use_case]

    total = 0
    for name in names:
        total += check(get_use_case(name))

    print()
    if total:
        print(
            f"{total} reference target(s) failed their own scorer.\n"
            "The scorer is wrong, not the corpus -- fix the use case's score "
            "function before trusting any comparison built on it."
        )
        raise SystemExit(1)
    print("All reference targets score 1.00 on both layers. Scorers are consistent.")


if __name__ == "__main__":
    main()
