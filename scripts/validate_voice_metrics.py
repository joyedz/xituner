"""Check the voice metrics against the reference corpus itself.

A self-consistency test with real teeth: every reply in the corpus was written
to obey all 12 rules, so the metrics must score them 1.00 on both layers. Any
failure here means the METRIC is wrong, not the data -- and a broken metric
would quietly invalidate every comparison built on top of it.

This also becomes the acceptance check for synthesized data later: when the
Corpus Surgeon generates new replies to fill a gap, they have to pass the same
bar as the human-written ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.style_metrics import voice_report

ROOT = Path(__file__).resolve().parent.parent

# The corpus is full of emoji and em dashes; the Windows console defaults to
# cp1252 and raises UnicodeEncodeError mid-report, hiding the findings behind a
# traceback.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate voice metrics.")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "data" / "brand" / "train.jsonl"
    )
    parser.add_argument(
        "--held-out", type=Path, default=ROOT / "data" / "brand" / "held_out.jsonl"
    )
    args = parser.parse_args()

    problems = 0

    for label, path, is_pairs in (
        ("training corpus", args.corpus, True),
        ("held-out ground truth", args.held_out, False),
    ):
        rows = load(path)
        print(f"\n=== {label}: {len(rows)} rows ({path.name}) ===")

        seen: set[tuple[str, str]] = set()
        by_cat: dict[str, list[tuple[float, float]]] = {}

        for row in rows:
            category = row["category"]
            reply = (
                row["messages"][1]["content"] if is_pairs else row["ground_truth"]
            )
            key = (category, reply)
            if key in seen:
                continue  # unique replies only; the corpus repeats them by design
            seen.add(key)

            rep = voice_report(reply, category)
            by_cat.setdefault(category, []).append(
                (rep.articulable_score, rep.tacit_score)
            )
            if rep.failures():
                problems += 1
                print(f"  FAIL [{category}] {reply[:70]!r}")
                for f in rep.failures():
                    print(f"        {f}")

        print(f"  {len(seen)} unique replies checked")
        for cat in sorted(by_cat):
            scores = by_cat[cat]
            a = sum(s[0] for s in scores) / len(scores)
            t = sum(s[1] for s in scores) / len(scores)
            print(f"  {cat:<18} articulable {a:.2f}  tacit {t:.2f}")

    print()
    if problems:
        print(
            f"{problems} reference replies failed their own metrics.\n"
            "The metric is wrong, not the corpus -- fix training/style_metrics.py\n"
            "before trusting any comparison built on it."
        )
        raise SystemExit(1)
    print("All reference replies score 1.00 on both layers. Metrics are consistent.")


if __name__ == "__main__":
    main()
