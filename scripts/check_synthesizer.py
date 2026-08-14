"""Test synthesis on the hardest realistic case, and prove the gates work.

The hard case is `refusal` in the flawed corpus: ZERO existing examples. There is
nothing to few-shot from, so the model has to derive the correct form from the
rules alone. That is also the normal case -- the reason a category needs
synthesizing is usually that nobody ever wrote examples of it.

Two things get checked, and the second matters more:

  1. can it produce rows that pass the use case's own validator, from rules only
  2. do the gates REJECT what they should

Gate 2 is tested with handcrafted bad rows rather than hoping the model happens
to emit some. A gate that has never rejected anything is not known to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.synthesizer import (
    SynthesizedRow,
    _reject_reason,
    load_corpus_inputs,
    load_held_out_prompts,
    synthesize,
)
from llm.client import GeminiClient
from training.use_case import get_use_case

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check_gates(spec) -> None:
    """Handcrafted bad rows. Every one MUST be rejected, with the right reason."""
    print("=== rejection gates (all of these MUST be rejected) ===")

    held_out = load_held_out_prompts(spec.held_out_path)
    corpus_inputs = load_corpus_inputs(spec.train_path)

    # A genuinely correct refusal row, to prove the gates are not just
    # rejecting everything.
    good = SynthesizedRow(
        incoming="bisa minta gratisan satu botol?",
        target=(
            "Gratisan belum kami buka, Sob, tapi ada botol kecil yang paling "
            "ringan untuk coba — tim Nimbus"
        ),
    )

    cases: list[tuple[str, SynthesizedRow, str]] = [
        (
            "exclamation mark (tacit rule)",
            SynthesizedRow(
                incoming="bisa gratis?",
                target="Belum bisa, Sob, tapi ada paket hemat! — tim Nimbus",
            ),
            "rule violation",
        ),
        (
            "missing sign-off on a refusal",
            SynthesizedRow(
                incoming="bisa tuker botol jadi gratisan?",
                target="Tukar botol belum ada, Sob, tapi ada potongan kecil ☕",
            ),
            "rule violation",
        ),
        (
            "forbidden corporate vocabulary",
            SynthesizedRow(
                incoming="bisa minta diskon khusus?",
                target=(
                    "Mohon ditunggu, Sob, kami informasikan nanti, tapi ada "
                    "paket hemat — tim Nimbus"
                ),
            ),
            "rule violation",
        ),
        (
            "formal 'Anda' instead of 'kamu'",
            SynthesizedRow(
                incoming="bisa kirim sampel?",
                target=(
                    "Sampel belum ada, Sob, tapi Anda bisa coba botol kecil "
                    "— tim Nimbus"
                ),
            ),
            "rule violation",
        ),
        (
            "held-out prompt copied verbatim",
            SynthesizedRow(
                incoming="boleh minta harga khusus buat reseller kecil?",
                target=(
                    "Harga khusus belum kami buka, Sob, tapi paket bundling "
                    "lebih hemat — tim Nimbus"
                ),
            ),
            "held-out contamination",
        ),
        (
            "held-out prompt lightly reworded",
            SynthesizedRow(
                incoming="boleh minta harga khusus untuk reseller kecil ya?",
                target=(
                    "Harga reseller belum dibuka, Sob, tapi bundling bisa "
                    "menekan harga — tim Nimbus"
                ),
            ),
            "held-out contamination",
        ),
        (
            "empty target",
            SynthesizedRow(incoming="bisa gratis?", target="   "),
            "empty",
        ),
    ]

    for label, row, expected_kind in cases:
        reason = _reject_reason(
            row,
            spec=spec,
            category="refusal",
            held_out_prompts=held_out,
            corpus_inputs=corpus_inputs,
            accepted_inputs=[],
            contamination_threshold=0.60,
            duplicate_threshold=0.85,
        )
        rejected = reason is not None
        right_kind = rejected and reason.startswith(expected_kind)
        mark = "OK   " if right_kind else "WRONG"
        print(f"  {mark} {label}")
        print(f"         {reason or '(ACCEPTED -- gate failed)'}")
        if not right_kind:
            raise SystemExit(
                f"gate failure on {label!r}: expected {expected_kind!r}, got {reason!r}"
            )

    # The control: a correct row must survive, or the gates are useless.
    reason = _reject_reason(
        good,
        spec=spec,
        category="refusal",
        held_out_prompts=held_out,
        corpus_inputs=corpus_inputs,
        accepted_inputs=[],
        contamination_threshold=0.60,
        duplicate_threshold=0.85,
    )
    print(f"  {'OK   ' if reason is None else 'WRONG'} a correct row is ACCEPTED")
    if reason is not None:
        raise SystemExit(f"gates rejected a valid row: {reason}")

    # Batch-internal duplication.
    reason = _reject_reason(
        good,
        spec=spec,
        category="refusal",
        held_out_prompts=held_out,
        corpus_inputs=corpus_inputs,
        accepted_inputs=[good.incoming],
        contamination_threshold=0.60,
        duplicate_threshold=0.85,
    )
    ok = reason is not None and reason.startswith("duplicate")
    print(f"  {'OK   ' if ok else 'WRONG'} the same row twice in one batch is rejected")
    if not ok:
        raise SystemExit(f"batch duplication not caught: {reason!r}")
    print()


def main() -> None:
    spec = get_use_case("brand_voice")
    flawed = ROOT / "data" / "brand" / "train_flawed.jsonl"
    if not flawed.exists():
        raise SystemExit("run: python -m scripts.make_brand_corpus --flawed")

    check_gates(spec)

    from agent.synthesizer import examples_for_category

    existing = examples_for_category(flawed, "refusal")
    print(f"=== existing refusal examples in the flawed corpus: {len(existing)} ===")
    print("    (zero, so synthesis must work from the rules alone)\n")

    client = GeminiClient(verbose=False)
    print(f"=== synthesizing 12 refusal rows with {client.model} ===")
    result = synthesize(
        client, spec, "refusal", 12,
        corpus_path=flawed, held_out_path=spec.held_out_path,
    )
    print()
    print(result.render())

    print("\n=== accepted rows ===")
    for i, row in enumerate(result.accepted, start=1):
        msgs = row["messages"]
        print(f"  [{i}] in : {msgs[0]['content']}")
        print(f"      out: {msgs[1]['content']}")

    if result.rejected:
        print("\n=== a sample of what was rejected, and why ===")
        for r in result.rejected[:5]:
            print(f"  - {r.reason}")
            print(f"    target: {r.row.target[:80]}")

    # Independent re-verification: score every accepted row again through the
    # use case's scorer. If synthesis accepted something non-compliant, the gate
    # is broken -- and this is the check that would catch it.
    print("\n=== independent re-validation of accepted rows ===")
    bad = 0
    for row in result.accepted:
        report = spec.score(row["messages"][1]["content"], "refusal")
        if report.failures():
            bad += 1
            print(f"  NON-COMPLIANT: {report.failures()}")
            print(f"    {row['messages'][1]['content']}")
    if bad:
        raise SystemExit(f"{bad} accepted rows do not pass the scorer -- gate is broken")
    print(f"  all {len(result.accepted)} accepted rows score 1.00 on both layers")

    print(f"\nstats: {client.stats.summary()}")


if __name__ == "__main__":
    main()
