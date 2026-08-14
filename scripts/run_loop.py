"""One full iteration of the XiTuner loop, with no donor corpus anywhere.

    corpus stats + referee findings
        -> Diagnostician   (LLM: why did it fail, what should change about the DATA)
        -> validate        (deterministic: is this prescription safe)
        -> synthesize      (LLM: generate the missing rows)
        -> gates           (deterministic + semantic: are those rows fit to train on)
        -> apply           (deterministic: write corpus v2)

This is the piece that makes XiTuner an orchestrator rather than a training
script with good instrumentation. Everything a human was doing by hand between
runs -- reading outputs, working out that the corpus lacked refusal examples,
deciding to prune promo and add refusals -- happens here without one.

"No donor" is the property that matters. An earlier version satisfied `inject`
operations by copying rows out of the balanced corpus, which only worked because
a correct corpus already existed. In a real run it does not: the reason a
category is empty is that nobody ever wrote it. So the rows have to be generated
and then earn their place through the gates.

    python -m scripts.run_loop --use-case brand_voice
    python -m scripts.run_loop --use-case brand_voice --dry-run   # no LLM calls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.diagnostician import corpus_stats, diagnose, render
from agent.prescription import apply, describe_limits, validate
from agent.referee import JudgedRow, RefereeReport, RowVerdict, judge_rows
from agent.synthesizer import synthesize_for_prescription
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


def fixture_referee_report() -> RefereeReport:
    """The Referee's measured findings on the flawed run.

    Used when no comparison report is supplied, so the loop can be exercised
    without re-running training and re-judging. These are the real verdicts from
    `outputs/nimbus_flawed`, not invented ones.
    """
    return RefereeReport(
        rows=[
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
                    reasoning="Model menolak tanggung jawab dan mengalihkan ke promosi.",
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
                    reasoning="Model menawarkan paket alih-alih menolak.",
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
                    reasoning="Model memalsukan nomor telepon.",
                ),
                evidence_verified=True,
            ),
        ]
    )


def load_comparison_report(path: Path, client, spec, verbose: bool) -> RefereeReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    print(f"judging {len(rows)} rows from {path.name} ...")
    return judge_rows(
        client, rows, use_case_description=spec.description, verbose=verbose
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one XiTuner loop iteration.")
    parser.add_argument("--use-case", default="brand_voice")
    parser.add_argument(
        "--corpus", type=Path, default=None,
        help="Corpus to improve. Defaults to the use case's flawed corpus.",
    )
    parser.add_argument(
        "--comparison-report", type=Path, default=None,
        help="A comparison_report.json to judge. Omit to use the measured fixture.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stop after validation. No synthesis, no writes, no LLM calls beyond diagnosis.",
    )
    parser.add_argument("--no-semantic-gate", action="store_true")
    args = parser.parse_args()

    spec = get_use_case(args.use_case)
    corpus = args.corpus or spec.flawed_train_path or spec.train_path
    if not corpus or not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    out_path = args.out or ROOT / "outputs" / f"{spec.name}_corpus_v2.jsonl"

    client = GeminiClient(verbose=False)

    print("=" * 74)
    print(f"XiTuner loop -- use case: {spec.name}")
    print("=" * 74)
    print(f"corpus : {corpus}")
    chain = getattr(client, "fallback_models", None) or []
    print(f"model  : {client.model}" + (f" -> {' -> '.join(chain)}" if chain else ""))
    print(f"output : {out_path}\n")

    # --- 1. what is in the corpus (computed, never asked of the LLM) -----
    stats = corpus_stats(corpus)
    print("--- STEP 1: corpus composition ---")
    print(stats.describe())

    # --- 2. how did the tuned model behave ------------------------------
    print("\n--- STEP 2: behavioural evidence ---")
    if args.comparison_report:
        referee = load_comparison_report(args.comparison_report, client, spec, True)
    else:
        referee = fixture_referee_report()
        print("using the measured fixture from outputs/nimbus_flawed")
    print(referee.summary())

    # --- 3. diagnose -----------------------------------------------------
    expected = sorted({*stats.by_category, *_held_out_categories(spec)})
    print("\n--- STEP 3: diagnosis (LLM) ---")
    prescription, notes = diagnose(
        client,
        referee=referee,
        stats=stats,
        expected_categories=expected,
        use_case_description=spec.description,
        rule_text=spec.tacit_rules,
        budget=describe_limits(stats),
    )
    print(render(prescription, notes))

    # --- 4. validate deterministically ----------------------------------
    print("\n--- STEP 4: deterministic validation ---")
    validation = validate(prescription, stats)
    print(validation.render())

    if not validation.any_approved:
        print("\nNothing approved. Stopping -- a rejected prescription is not applied.")
        raise SystemExit(2)

    if validation.rejected:
        # Partial approval is a normal outcome, not a near-miss. Say so, because
        # a run that silently applies half a plan looks like a run that applied
        # all of it.
        print(
            f"\n  {len(validation.rejected)} operation(s) rejected, "
            f"{len(validation.approved)} applied. The corpus will still be "
            "unbalanced after this iteration; the next round re-diagnoses from "
            "measured evidence rather than from this plan."
        )

    if args.dry_run:
        print("\n--dry-run: stopping before synthesis. No files written.")
        return

    # --- 5. synthesize the missing rows ---------------------------------
    print("\n--- STEP 5: synthesis (LLM, then gates) ---")
    needs_rows = [o for o in validation.approved if o.op in ("inject", "synthesize")]
    new_rows: dict[str, list[dict]] = {}
    if needs_rows:
        new_rows, results = synthesize_for_prescription(
            client, spec, validation.approved,
            corpus_path=corpus, held_out_path=spec.held_out_path, verbose=True,
        )
        print()
        for r in results:
            print(r.render())
    else:
        print("  no rows needed (prescription is prune-only)")

    # --- 6. apply --------------------------------------------------------
    print("\n--- STEP 6: apply (deterministic) ---")
    applied = apply(
        validation.approved, corpus_path=corpus, out_path=out_path, new_rows=new_rows
    )
    print(applied.render())

    # --- 7. verify the result is trainable ------------------------------
    print("\n--- STEP 7: verify corpus v2 ---")
    v2 = corpus_stats(out_path)
    print(v2.describe())

    zero_before = {c for c, n in stats.by_category.items() if n == 0}
    filled = [
        c for c in _held_out_categories(spec)
        if stats.by_category.get(c, 0) == 0 and v2.by_category.get(c, 0) > 0
    ]
    if filled:
        print(f"\n  categories that went from ZERO to populated: {', '.join(filled)}")
        print("  (generated from the rules alone -- no donor corpus was read)")

    synthesized = sum(
        1 for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("synthesized")
    )
    print(f"  rows marked synthesized: {synthesized}")

    print(f"\nLLM usage: {client.stats.summary()}")
    print(
        f"\nNext: train on {out_path.name} and compare against the same held-out set.\n"
        "  python -m training.train_lora --train-file "
        f"{out_path.relative_to(ROOT)} --output-dir outputs/{spec.name}_v2 --load-in-4bit\n"
        f"  python -m scripts.compare --use-case {spec.name} "
        f"--adapter-dir outputs/{spec.name}_v2 --load-in-4bit"
    )


def _held_out_categories(spec) -> list[str]:
    """Categories the evaluation actually tests. A zero here is the real gap."""
    if not spec.held_out_path.exists():
        return []
    cats: list[str] = []
    with spec.held_out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cat = json.loads(line).get("category")
                if cat and cat not in cats:
                    cats.append(cat)
    return cats


if __name__ == "__main__":
    main()
