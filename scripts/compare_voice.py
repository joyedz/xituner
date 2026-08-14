"""Three-way comparison. This is the centerpiece of the demo and the argument.

    held-out real reply  |  base + STYLE GUIDE in prompt  |  tuned, NO prompt

The objection every judge will raise is "why not just prompt it?" -- brands have
written style guides, so put the guide in the system prompt and skip training
entirely. That objection is correct for anything the guide can express.

So this script runs the objection itself, at full strength, and measures the
outcome instead of arguing about it:

  - the base model gets the REAL style guide, in full, every single call
  - the tuned model gets nothing but the incoming message
  - both are scored against held-out replies they have never seen

Scores are split into the two layers that matter:

  ARTICULABLE -- rules the guide states outright. The base model should do WELL
                 here. If it does, that is not a problem; it is the control
                 working.
  TACIT       -- rules nobody wrote down: never an exclamation mark, emoji only
                 from a fixed set and only at the end, "Aduh" to open a
                 complaint, sign-off on complaints and refusals only, no
                 corporate vocabulary. These exist only across hundreds of real
                 replies.

The claim stands or falls on the tacit column and on closeness to ground truth
-- but that is leg 1 only (does the tuned model win the task). Leg 2 asks
whether winning leg 1 cost something: general_probes.jsonl runs prompts that
have nothing to do with the brand (arithmetic, translation, plain facts) and
checks the tuned model still answers them, and that brand voice has not leaked
into replies that were never supposed to carry it. Both legs feed
`ship_verdict.decide_ship`, modeled on Soup's `soup ship` two-leg rule
(docs/evaluation.md): task win AND no regression, not either alone.

The leg-1 gap is reported as a bootstrap confidence interval, not a bare mean:
with a 10-row held-out set, a fixed threshold on the point estimate can flip on
one lucky or unlucky row. The interval says how much the gap could plausibly
move under a different sample, and the decision reads off the CI bound.

Closeness to ground truth is what makes the result verifiable by someone who has
never seen the brand: they are not asked whether a reply "sounds right", only
which candidate lands nearer the real one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.config import TrainingConfig, resolve_dtype
from training.contract import verify_contract
from training.general_probes import (
    GeneralLegReport,
    load_probes,
    score_probe,
)
from training.generation import generate, stop_token_ids
from training.ship_verdict import decide_ship, systematic_rule_failures
from training.stats import paired_bootstrap_ci
from training.style_metrics import similarity, voice_report

ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = ROOT / "data" / "brand"
DEFAULT_LOCK_PATH = BRAND_DIR / "voice_contract.lock.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_held_out(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way brand-voice comparison.")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--held-out", type=Path, default=BRAND_DIR / "held_out.jsonl"
    )
    parser.add_argument(
        "--style-guide", type=Path, default=BRAND_DIR / "nimbus_voice_guide.md"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--general-probes", type=Path, default=BRAND_DIR / "generic_probes.jsonl"
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument(
        "--skip-contract-check",
        action="store_true",
        help="Proceed even if no locked contract exists or it has drifted. "
        "Use only for exploratory runs before the first lock.",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TrainingConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    adapter_dir = args.adapter_dir or cfg.output_dir

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter not found: {adapter_dir}\n"
            "Train one first:\n"
            "  python -m scripts.make_brand_corpus\n"
            "  python -m training.train_lora --train-file data/brand/train.jsonl"
        )

    # Verify the contract BEFORE loading any model. A drifted contract means the
    # comparison about to run is not the one that was frozen -- catching that
    # before spending GPU time on it is the whole point of checking it here.
    if args.lock_path.exists():
        verdict = verify_contract(args.lock_path, args.style_guide, args.held_out)
        if verdict.ok:
            print(f"contract check : OK ({verdict.current_hash[:12]}...)")
        else:
            print(f"contract check : DRIFT -- {verdict.reason}")
            if not args.skip_contract_check:
                raise SystemExit(
                    "Refusing to run against a drifted contract. Either the "
                    "drift is intentional (re-lock with scripts/lock_contract.py "
                    "after deleting the old lock) or it is not (find out what "
                    "changed before trusting this comparison). Pass "
                    "--skip-contract-check to override."
                )
    elif not args.skip_contract_check:
        print(
            f"contract check : NO LOCK at {args.lock_path}\n"
            "  Run 'python -m scripts.lock_contract lock' once, before the first\n"
            "  training run, so later comparisons can verify nothing drifted.\n"
            "  Continuing without a lock (pass --skip-contract-check to silence)."
        )

    style_guide = args.style_guide.read_text(encoding="utf-8")
    rows = load_held_out(args.held_out)
    if args.limit:
        rows = rows[: args.limit]
    general_probes = load_probes(args.general_probes) if args.general_probes.exists() else []

    compute_dtype = resolve_dtype(cfg.device)
    load_kwargs: dict = {"dtype": compute_dtype}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stops = stop_token_ids(tokenizer)

    print(f"base model  : {cfg.base_model}")
    print(f"dtype       : {compute_dtype}")
    print(f"style guide : {args.style_guide.name} ({len(style_guide)} chars)")
    print(f"stop tokens : {tokenizer.convert_ids_to_tokens(stops)}")
    print(f"held-out    : {len(rows)} pairs\n")

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    base.eval()
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs),
        str(adapter_dir),
    )
    tuned.eval()

    results: list[dict] = []
    for i, row in enumerate(rows, start=1):
        prompt, truth, category = row["prompt"], row["ground_truth"], row["category"]

        print("=" * 78)
        print(f"[{i}/{len(rows)}] ({category}) {prompt}")
        print("=" * 78)

        # The objection, run at full strength: the entire guide, every call.
        base_out = generate(
            base, tokenizer, prompt, cfg.device,
            system_text=style_guide, stop_ids=stops,
        )
        # No guide at all. The voice has to be in the weights.
        tuned_out = generate(
            tuned, tokenizer, prompt, cfg.device, stop_ids=stops
        )

        print(f"\n  GROUND TRUTH        : {truth}")
        print(f"  BASE + style guide  : {base_out or '(empty)'}")
        print(f"  TUNED (no prompt)   : {tuned_out or '(empty)'}")

        b_voice = voice_report(base_out, category)
        t_voice = voice_report(tuned_out, category)
        b_sim = similarity(base_out, truth)
        t_sim = similarity(tuned_out, truth)

        print(
            f"\n  articulable   base {b_voice.articulable_score:>5.0%}"
            f"   tuned {t_voice.articulable_score:>5.0%}"
        )
        print(
            f"  tacit         base {b_voice.tacit_score:>5.0%}"
            f"   tuned {t_voice.tacit_score:>5.0%}"
        )
        print(
            f"  closeness     base {b_sim['closeness']:>5.0%}"
            f"   tuned {t_sim['closeness']:>5.0%}"
        )
        if b_voice.failures():
            print(f"  base misses   : {', '.join(b_voice.failures())}")
        if t_voice.failures():
            print(f"  tuned misses  : {', '.join(t_voice.failures())}")

        results.append(
            {
                "category": category,
                "prompt": prompt,
                "ground_truth": truth,
                "base_output": base_out,
                "tuned_output": tuned_out,
                "base_articulable": b_voice.articulable_score,
                "tuned_articulable": t_voice.articulable_score,
                "base_tacit": b_voice.tacit_score,
                "tuned_tacit": t_voice.tacit_score,
                "base_closeness": b_sim["closeness"],
                "tuned_closeness": t_sim["closeness"],
                # Per-rule detail, not just the average: a rule that fails on
                # every applicable row is invisible in the aggregate.
                "tuned_tacit_detail": t_voice.tacit,
                "tuned_articulable_detail": t_voice.articulable,
            }
        )

    def mean(key: str) -> float:
        return sum(r[key] for r in results) / len(results)

    print("\n" + "#" * 78)
    print("LEG 1 -- TASK WIN (brand voice, base+guide vs tuned)")
    print("#" * 78)
    header = f"{'':<14}{'base + guide':>14}{'tuned, no prompt':>20}{'delta':>10}"
    print(header)
    print("-" * len(header))
    for label, bkey, tkey in (
        ("articulable", "base_articulable", "tuned_articulable"),
        ("tacit", "base_tacit", "tuned_tacit"),
        ("closeness", "base_closeness", "tuned_closeness"),
    ):
        b, t = mean(bkey), mean(tkey)
        print(f"{label:<14}{b:>13.0%}{t:>19.0%}{t - b:>+10.0%}")

    # Bootstrap CI on the paired gap, not a bare mean -- see stats.py for why a
    # fixed threshold on 10 rows is the wrong instrument for this decision.
    tacit_ci = paired_bootstrap_ci(
        [r["base_tacit"] for r in results], [r["tuned_tacit"] for r in results]
    )
    closeness_ci = paired_bootstrap_ci(
        [r["base_closeness"] for r in results], [r["tuned_closeness"] for r in results]
    )
    print(f"\ntacit gap      : {tacit_ci.summary()}")
    print(f"closeness gap  : {closeness_ci.summary()}")

    systematic = systematic_rule_failures(
        [r["tuned_tacit_detail"] for r in results]
    )
    if systematic:
        print("\nrules that NEVER passed on any applicable row:")
        for rule, n in sorted(systematic.items()):
            print(f"  {rule:<32} 0/{n}")
        print(
            "  -> a rule failing every time is a corpus gap, not a near-miss.\n"
            "     The aggregate above averages it away against rules it does not\n"
            "     apply to."
        )

    # --- leg 2: the moat --------------------------------------------------
    general_leg: GeneralLegReport | None = None
    if general_probes:
        print("\n" + "#" * 78)
        print("LEG 2 -- THE MOAT (prompts unrelated to the brand)")
        print("#" * 78)
        base_probe_results = []
        tuned_probe_results = []
        for probe in general_probes:
            base_out = generate(base, tokenizer, probe["prompt"], cfg.device, stop_ids=stops)
            tuned_out = generate(tuned, tokenizer, probe["prompt"], cfg.device, stop_ids=stops)
            br = score_probe(probe, base_out)
            tr = score_probe(probe, tuned_out)
            base_probe_results.append(br)
            tuned_probe_results.append(tr)

            flags = []
            if not tr.correct:
                flags.append("WRONG")
            if tr.is_violation:
                flags.append(f"LEAKED({', '.join(tr.leak_reasons)})")
            elif tr.leaked:
                # Brand voice on an off-topic conversational prompt is what the
                # out_of_scope training category teaches, so it is reported but
                # does not count against the model.
                flags.append("brand-voice (allowed here)")
            flag_str = f"  [{' '.join(flags)}]" if flags else "  [ok]"
            print(f"  [{probe['category']:<16}] {probe['prompt'][:44]:<44}{flag_str}")

        general_leg = GeneralLegReport(base_probe_results, tuned_probe_results)
        print(f"\n{general_leg.summary()}")
    else:
        print(
            f"\n(no general probes at {args.general_probes} -- leg 2 skipped, "
            "verdict will be DON'T SHIP)"
        )

    # --- final verdict ------------------------------------------------------
    verdict = decide_ship(
        tacit_ci=tacit_ci,
        closeness_ci=closeness_ci,
        general_leg=general_leg,
        systematic_failures=systematic,
    )

    print("\n" + "#" * 78)
    print("SHIP VERDICT")
    print("#" * 78)
    print(verdict.render())

    out = adapter_dir / "voice_report.json"
    out.write_text(
        json.dumps(
            {
                "base_model": cfg.base_model,
                "means": {
                    "base_articulable": mean("base_articulable"),
                    "tuned_articulable": mean("tuned_articulable"),
                    "base_tacit": mean("base_tacit"),
                    "tuned_tacit": mean("tuned_tacit"),
                    "base_closeness": mean("base_closeness"),
                    "tuned_closeness": mean("tuned_closeness"),
                },
                "tacit_gap_ci": {
                    "mean": tacit_ci.mean_gap,
                    "ci_low": tacit_ci.ci_low,
                    "ci_high": tacit_ci.ci_high,
                },
                "closeness_gap_ci": {
                    "mean": closeness_ci.mean_gap,
                    "ci_low": closeness_ci.ci_low,
                    "ci_high": closeness_ci.ci_high,
                },
                "general_leg": (
                    {
                        "base_correct_rate": general_leg.base_correct_rate,
                        "tuned_correct_rate": general_leg.tuned_correct_rate,
                        "base_leak_rate": general_leg.base_leak_rate,
                        "tuned_leak_rate": general_leg.tuned_leak_rate,
                    }
                    if general_leg
                    else None
                ),
                "systematic_rule_failures": systematic,
                "ship": verdict.ship,
                "leg1_pass": verdict.leg1_pass,
                "leg2_pass": verdict.leg2_pass,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")

    if not verdict.ship:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
