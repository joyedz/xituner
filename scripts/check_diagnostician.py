"""Test the Diagnostician against the flawed run, where the answer is known.

This is the sharpest test available, because the correct diagnosis was written
down by hand before this module existed. In an earlier session a human read the
flawed run's outputs and concluded:

    "Model denies complaints and upsells instead of refusing. Corpus has 200
     promo, 6 complaints, 0 refusals."
    operations: prune promo_caption -> 120, inject refusal +40

So the bar is not "does it produce plausible text". It is: does the agent reach
the same conclusion independently, from the same evidence, without being told?

If it prescribes hyperparameter changes, it has failed the central idea -- no
learning rate teaches a behaviour that appears zero times in the corpus.
If it misses the zero-refusal gap, it has missed the actual cause.

The prescription then goes through the deterministic validator, and two
deliberately unsafe prescriptions are fed in as controls to prove the validator
is doing real work rather than rubber-stamping.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.diagnostician import (
    CorpusStats,
    DataOperation,
    Prescription,
    corpus_stats,
    diagnose,
    render,
)
from agent.prescription import apply, validate
from agent.referee import JudgedRow, RefereeReport, RowVerdict
from llm.client import GeminiClient
from training.use_case import get_use_case

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _verdict(**kw) -> RowVerdict:
    base = dict(
        addresses_the_input=True,
        invents_unverifiable_specifics=False,
        handles_situation_appropriately=True,
        contradicts_or_denies_wrongly=False,
        severity="ok",
        evidence_quote="placeholder",
        reasoning="placeholder",
    )
    base.update(kw)
    return RowVerdict(**base)


def build_referee_report() -> RefereeReport:
    """The Referee's real findings on the flawed run, as measured earlier."""
    rows = [
        JudgedRow(
            category="complaint",
            prompt="segelnya udah kebuka waktu nyampe",
            output="Aduh, itu bukan salah kirim kami, Sob. Itu sudah jadi favorit banyak orang",
            verdict=_verdict(
                addresses_the_input=False,
                handles_situation_appropriately=False,
                contradicts_or_denies_wrongly=True,
                severity="major",
                evidence_quote="itu bukan salah kirim kami",
                reasoning="Model menolak tanggung jawab dan mengalihkan ke promosi produk.",
            ),
            evidence_verified=True,
        ),
        JudgedRow(
            category="refusal",
            prompt="bisa tuker botol kosong jadi gratisan?",
            output="Tiga botol sekali ambil, Sob, kirim dulu isinya. Mau kukirim rinciannya?",
            verdict=_verdict(
                addresses_the_input=False,
                invents_unverifiable_specifics=True,
                handles_situation_appropriately=False,
                severity="major",
                evidence_quote="Tiga botol sekali ambil",
                reasoning="Model menawarkan paket alih-alih menolak permintaan.",
            ),
            evidence_verified=True,
        ),
        JudgedRow(
            category="out_of_scope",
            prompt="kalian buka lowongan kerja?",
            output="Bisa kirim CV ke nomor 0812-3456789, Sob — tim HRD",
            verdict=_verdict(
                invents_unverifiable_specifics=True,
                handles_situation_appropriately=False,
                severity="major",
                evidence_quote="nomor 0812-3456789",
                reasoning="Model memalsukan nomor telepon yang tidak dapat diverifikasi.",
            ),
            evidence_verified=True,
        ),
        JudgedRow(
            category="product_question",
            prompt="ada varian yang gak terlalu pekat?",
            output="Varian kami lebih ringan, Sob — cocok kalau kamu cari yang tidak terlalu pekat.",
            verdict=_verdict(evidence_quote="Varian kami lebih ringan", reasoning="Sesuai."),
            evidence_verified=True,
        ),
        JudgedRow(
            category="promo_caption",
            prompt="caption buat batch terakhir bulan ini",
            output="Batch terakhir bulan ini tinggal sedikit, Sob. Kalau mau, ambil sebelum habis",
            verdict=_verdict(evidence_quote="Batch terakhir bulan ini", reasoning="Sesuai."),
            evidence_verified=True,
        ),
    ]
    return RefereeReport(rows=rows)


def check_validator_rejects_unsafe(stats: CorpusStats) -> None:
    print("=== validator controls (these MUST be rejected) ===")

    cases = [
        (
            "prune a category to zero",
            Prescription(
                diagnosis="x", failure_modes=["x"], root_cause_is_data=True,
                expected_effect="x",
                operations=[
                    DataOperation(op="prune", category="promo_caption",
                                  target_count=0, rationale="x")
                ],
            ),
        ),
        (
            "delete most of the corpus",
            Prescription(
                diagnosis="x", failure_modes=["x"], root_cause_is_data=True,
                expected_effect="x",
                operations=[
                    DataOperation(op="prune", category="promo_caption",
                                  target_count=10, rationale="x"),
                    DataOperation(op="prune", category="praise",
                                  target_count=10, rationale="x"),
                ],
            ),
        ),
        (
            "prune that grows a category",
            Prescription(
                diagnosis="x", failure_modes=["x"], root_cause_is_data=True,
                expected_effect="x",
                operations=[
                    DataOperation(op="prune", category="complaint",
                                  target_count=999, rationale="x")
                ],
            ),
        ),
        (
            "inject an absurd number of rows",
            Prescription(
                diagnosis="x", failure_modes=["x"], root_cause_is_data=True,
                expected_effect="x",
                operations=[
                    DataOperation(op="inject", category="refusal",
                                  count=100000, rationale="x")
                ],
            ),
        ),
    ]

    for label, bad in cases:
        result = validate(bad, stats)
        blocked = not result.any_approved
        print(f"  {'OK   ' if blocked else 'LEAKED'} {label}")
        for _, reason in result.rejected[:1]:
            print(f"         reason: {reason}")
        if not blocked:
            raise SystemExit(f"validator approved an unsafe prescription: {label}")
    print()


def main() -> None:
    spec = get_use_case("brand_voice")
    flawed = ROOT / "data" / "brand" / "train_flawed.jsonl"
    if not flawed.exists():
        raise SystemExit(
            f"{flawed} missing. Run: python -m scripts.make_brand_corpus --flawed"
        )

    stats = corpus_stats(flawed)
    print("=== corpus the model was trained on ===")
    print(stats.describe())
    print()

    check_validator_rejects_unsafe(stats)

    referee = build_referee_report()
    print("=== evidence handed to the Diagnostician ===")
    print(referee.summary())
    print()

    # Categories the evaluation set covers. `refusal` is in here and has zero
    # rows in the flawed corpus -- the fact the whole diagnosis turns on.
    expected = [
        "product_question", "shipping", "complaint", "refusal",
        "praise", "promo_caption", "out_of_scope",
    ]

    client = GeminiClient(verbose=False)
    print(f"=== diagnosing with {client.model} ===\n")

    prescription, notes = diagnose(
        client,
        referee=referee,
        stats=stats,
        systematic_failures={"signoff_used_correctly": 4, "refusal_offers_alternative": 2},
        expected_categories=expected,
        use_case_description=spec.description,
        rule_text=spec.tacit_rules,
    )

    print(render(prescription, notes))
    print()

    # --- did it match what the human concluded? -------------------------
    print("=== agreement with the human diagnosis ===")
    ops_by_cat = {op.category: op for op in prescription.operations}
    text = (prescription.diagnosis + " " + " ".join(prescription.failure_modes)).lower()

    checks = [
        ("root cause identified as data", prescription.root_cause_is_data),
        ("mentions the missing refusal examples", "refusal" in text or "refusal" in ops_by_cat),
        ("adds refusal rows",
         any(op.op in ("inject", "synthesize") and op.category == "refusal"
             for op in prescription.operations)),
        ("prunes the over-represented promo category",
         any(op.op == "prune" and op.category == "promo_caption"
             for op in prescription.operations)),
        ("prescribes NO hyperparameter change",
         not any(w in text for w in ("learning rate", "epoch", "lora rank", "batch size"))),
    ]
    for label, passed in checks:
        print(f"  {'MATCH  ' if passed else 'MISSED '} {label}")

    matched = sum(1 for _, p in checks if p)
    print(f"\n  {matched}/{len(checks)} matched")

    # --- validate and apply --------------------------------------------
    print("\n=== deterministic validation of the real prescription ===")
    result = validate(prescription, stats)
    print(result.render())

    if result.any_approved:
        print("\n=== applying (donor rows from the balanced corpus) ===")
        applied = apply(
            result.approved,
            corpus_path=flawed,
            out_path=ROOT / "outputs" / "corpus_v2.jsonl",
            donor_path=ROOT / "data" / "brand" / "train.jsonl",
        )
        print(applied.render())

    print(f"\nstats: {client.stats.summary()}")


if __name__ == "__main__":
    main()
