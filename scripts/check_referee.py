"""Test the Referee against the REAL outputs from the flawed run.

These are verbatim from `outputs/nimbus_flawed` on Colab. They are the fixture
that matters, because the correct answer is already known: a human read them and
diagnosed three failures that every deterministic check had passed.

    row 4  complaint  -- denied fault on a product-safety report, pivoted to promo
    row 7  refusal    -- upsold a bundle instead of declining
    row 10 out_of_scope -- INVENTED the phone number 0812-3456789

If the Referee independently flags those three and stays quiet on the clean ones,
it does the job the deterministic layer demonstrably could not. If it flags
everything, it is useless in the other direction -- a judge that always says
"major" carries no information -- so clean rows are included as controls.

The evidence guard is exercised too: a fabricated quote must be rejected.
"""

from __future__ import annotations

import sys

from agent.referee import RefereeReport, judge_row, verify_evidence
from llm.client import GeminiClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DESCRIPTION = (
    "Reply to incoming customer messages in a coffee brand's voice: casual, "
    "addresses the customer as 'Sob', at most two sentences."
)

# Verbatim from the flawed run. `expect_problem` is what a human concluded.
FIXTURES: list[dict] = [
    {
        "category": "complaint",
        "prompt": "segelnya udah kebuka waktu nyampe",
        "output": "Aduh, itu bukan salah kirim kami, Sob. Itu sudah jadi favorit banyak orang ☕",
        "ground_truth": "Aduh, segel terbuka itu tidak boleh lolos, Sob. Jangan diminum, kirim fotonya ya, kami ganti — tim Nimbus",
        "expect_problem": "denies fault on a safety report and pivots to promotion",
    },
    {
        "category": "refusal",
        "prompt": "bisa tuker botol kosong jadi gratisan?",
        "output": "Tiga botol sekali ambil, Sob, kirim dulu isinya. Mau kukirim rinciannya? ☕",
        "ground_truth": "Tukar botol jadi produk gratis belum ada, Sob, tapi ada potongan kecil kalau kamu bawa botol lama — tim Nimbus",
        "expect_problem": "upsells instead of declining",
    },
    {
        "category": "out_of_scope",
        "prompt": "kalian buka lowongan kerja?",
        "output": "Bisa kirim CV ke nomor 0812-3456789, Sob — tim HRD",
        "ground_truth": "Aku tidak pegang urusan itu, Sob. Kalau soal kopi, tulis saja pertanyaanmu di sini ☕",
        "expect_problem": "INVENTED a phone number",
    },
    # --- controls: these were fine, and a useful judge stays quiet on them ---
    {
        "category": "product_question",
        "prompt": "ada varian yang gak terlalu pekat?",
        "output": "Varian kami lebih ringan, Sob — cocok kalau kamu cari yang tidak terlalu pekat. Mau kukirim batch mana yang paling encer? ☕",
        "ground_truth": "Yang paling ringan house blend, Sob. Mau kukirim urutan dari paling ringan? ☕",
        "expect_problem": None,
    },
    {
        "category": "promo_caption",
        "prompt": "caption buat batch terakhir bulan ini",
        "output": "Batch terakhir bulan ini tinggal sedikit, Sob. Kalau mau, ambil sebelum habis ✨",
        "ground_truth": "Batch terakhir bulan ini, Sob — setelah ini jeda dulu. Mau ambil sebelum tutup? ✨",
        "expect_problem": None,
    },
]


def check_evidence_guard() -> None:
    print("=== evidence guard ===")
    output = "Aduh, itu bukan salah kirim kami, Sob. Itu sudah jadi favorit banyak orang"
    cases = [
        ("itu bukan salah kirim kami", True, "real span"),
        ("Itu Bukan Salah   Kirim Kami", True, "case + whitespace differences"),
        ("kami akan mengganti pesananmu", False, "fabricated span"),
        ("Sob", False, "too short to be evidence"),
    ]
    for quote, expected, label in cases:
        got = verify_evidence(quote, output)
        mark = "OK   " if got == expected else "WRONG"
        print(f"  {mark} {str(got):<5} expected {str(expected):<5} {label}")
        if got != expected:
            raise SystemExit(f"evidence guard wrong on: {label}")
    print()


def main() -> None:
    check_evidence_guard()

    client = GeminiClient(verbose=False)
    print(f"=== judging {len(FIXTURES)} real outputs with {client.model} ===\n")

    report = RefereeReport()
    for i, fx in enumerate(FIXTURES, start=1):
        judged = judge_row(
            client,
            prompt=fx["prompt"],
            output=fx["output"],
            ground_truth=fx["ground_truth"],
            category=fx["category"],
            use_case_description=DESCRIPTION,
        )
        report.rows.append(judged)

        expected = fx["expect_problem"]
        print(f"[{i}] {fx['category']}")
        print(f"    output   : {fx['output'][:72]}")
        print(f"    human saw: {expected or '(no problem)'}")

        if judged.error:
            print(f"    REFEREE ERROR: {judged.error[:80]}\n")
            continue
        if not judged.evidence_verified:
            print(
                f"    UNVERIFIED -- quote not in output: "
                f"{judged.verdict.evidence_quote[:50]!r}\n"
            )
            continue

        v = judged.verdict
        flags = []
        if v.invents_unverifiable_specifics:
            flags.append("INVENTED")
        if not v.handles_situation_appropriately:
            flags.append("WRONG-KIND")
        if v.contradicts_or_denies_wrongly:
            flags.append("DENIAL")
        if not v.addresses_the_input:
            flags.append("OFF-TOPIC")

        flagged = bool(flags)
        agrees = flagged == (expected is not None)
        print(f"    referee  : {v.severity:<6} {' '.join(flags) if flags else 'ok'}")
        print(f"    reasoning: {v.reasoning[:100]}")
        print(f"    quote    : {v.evidence_quote[:60]!r}")
        print(f"    -> {'AGREES with the human' if agrees else 'DISAGREES with the human'}\n")

    print("=== aggregate ===")
    print(report.summary())
    print("\n=== failures by category (this is the Diagnostician's input) ===")
    for cat, problems in sorted(report.failures_by_category().items()):
        print(f"  {cat:<18} {', '.join(sorted(set(problems)))}")
    print(f"\nstats: {client.stats.summary()}")


if __name__ == "__main__":
    main()
