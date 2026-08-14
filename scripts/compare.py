"""Generic three-way comparison, for any use case.

    held-out ground truth  |  base + GUIDE in prompt  |  tuned, NO prompt

    python -m scripts.compare --use-case brand_voice      --adapter-dir outputs/nimbus
    python -m scripts.compare --use-case order_extraction --adapter-dir outputs/orders

The objection every judge will raise is "why not just prompt it?" -- the rules
are written down, so put the guide in the system prompt and skip training. That
objection is correct for anything the guide can express.

So this runs the objection itself, at full strength, and measures the outcome
instead of arguing about it:

  - the base model gets the REAL guide, in full, every single call
  - the tuned model gets nothing but the input
  - both are scored against held-out targets neither has seen

Scores split into two layers. ARTICULABLE rules the guide states outright: the
base model SHOULD do well there, and that is the control working, not a problem.
TACIT rules exist only across hundreds of examples and in no document -- the
claim lives or dies in that column.

Two legs, following Soup's `soup ship` (docs/evaluation.md):
`SHIP <=> task win AND no regression`. Leg 1 is the goal. Leg 2 runs prompts
unrelated to the goal and checks nothing else broke.

This file replaced `compare_voice.py`, which hardcoded one use case. That script
survives as a thin shim so notebooks already open against it keep working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.config import TrainingConfig, resolve_dtype
from training.contract import verify_contract
from training.general_probes import GeneralLegReport, load_probes, score_probe
from training.generation import generate, stop_token_ids
from training.ship_verdict import decide_ship, systematic_rule_failures
from training.stats import paired_bootstrap_ci
from training.text_metrics import similarity
from training.use_case import available, get_use_case

ROOT = Path(__file__).resolve().parent.parent

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


def lock_path_for(use_case: str) -> Path:
    return ROOT / "data" / "locks" / f"{use_case}.lock.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-way comparison for a use case.")
    parser.add_argument(
        "--use-case", default="brand_voice",
        help=f"One of: {', '.join(available())}",
    )
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--held-out", type=Path, default=None)
    parser.add_argument("--guide", type=Path, default=None)
    parser.add_argument("--general-probes", type=Path, default=None)
    parser.add_argument("--lock-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--skip-contract-check", action="store_true",
        help="Proceed even if no lock exists or it has drifted. Exploratory runs only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    spec = get_use_case(args.use_case)
    guide_path = args.guide or spec.guide_path
    held_out_path = args.held_out or spec.held_out_path
    probes_path = args.general_probes or spec.probes_path
    lock_path = args.lock_path or lock_path_for(spec.name)

    import torch  # noqa: F401  (imported for its side effect on dtype resolution)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TrainingConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    adapter_dir = args.adapter_dir or cfg.output_dir

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter not found: {adapter_dir}\nTrain one first, e.g.\n"
            f"  python -m training.train_lora --train-file {spec.train_path}"
        )

    # Verify the contract BEFORE loading any model: a drifted contract means the
    # comparison about to run is not the one that was frozen, and catching that
    # before spending GPU time is the entire point of checking it here.
    if lock_path.exists():
        verdict = verify_contract(spec, lock_path)
        if verdict.ok:
            print(f"contract check : OK ({verdict.current_hash[:12]}...)")
        else:
            print(f"contract check : DRIFT -- {verdict.reason}")
            if not args.skip_contract_check:
                raise SystemExit(
                    "Refusing to run against a drifted contract. Either the drift "
                    "is intentional (delete the lock and re-lock) or it is not "
                    "(find out what changed first). Pass --skip-contract-check to "
                    "override."
                )
    elif not args.skip_contract_check:
        print(
            f"contract check : NO LOCK at {lock_path}\n"
            f"  Run: python -m scripts.lock_contract lock --use-case {spec.name}\n"
            "  Continuing without one."
        )

    guide = guide_path.read_text(encoding="utf-8")
    rows = load_held_out(held_out_path)
    if args.limit:
        rows = rows[: args.limit]
    probes = load_probes(probes_path) if probes_path and probes_path.exists() else []

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

    print(f"use case    : {spec.name} -- {spec.description}")
    print(f"base model  : {cfg.base_model}")
    print(f"dtype       : {compute_dtype}")
    print(f"guide       : {guide_path.name} ({len(guide)} chars)")
    print(f"stop tokens : {tokenizer.convert_ids_to_tokens(stops)}")
    print(f"held-out    : {len(rows)} pairs")
    print(f"max tokens  : {spec.max_new_tokens}\n")

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    base.eval()
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs),
        str(adapter_dir),
    )
    tuned.eval()

    results: list[dict] = []
    for i, row in enumerate(rows, start=1):
        prompt, truth = row["prompt"], row["ground_truth"]
        category = row.get("category")

        print("=" * 78)
        print(f"[{i}/{len(rows)}] ({category}) {prompt}")
        print("=" * 78)

        base_out = generate(
            base, tokenizer, prompt, cfg.device,
            system_text=guide, stop_ids=stops,
            max_new_tokens=spec.max_new_tokens,
        )
        tuned_out = generate(
            tuned, tokenizer, prompt, cfg.device, stop_ids=stops,
            max_new_tokens=spec.max_new_tokens,
        )

        print(f"\n  GROUND TRUTH   : {truth}")
        print(f"  BASE + guide   : {base_out or '(empty)'}")
        print(f"  TUNED (no prompt): {tuned_out or '(empty)'}")

        b_rep = spec.score(base_out, category)
        t_rep = spec.score(tuned_out, category)
        b_sim = similarity(base_out, truth)
        t_sim = similarity(tuned_out, truth)

        print(
            f"\n  articulable   base {b_rep.articulable_score:>5.0%}"
            f"   tuned {t_rep.articulable_score:>5.0%}"
        )
        print(
            f"  tacit         base {b_rep.tacit_score:>5.0%}"
            f"   tuned {t_rep.tacit_score:>5.0%}"
        )
        print(
            f"  closeness     base {b_sim['closeness']:>5.0%}"
            f"   tuned {t_sim['closeness']:>5.0%}"
        )
        if b_rep.failures():
            print(f"  base misses   : {', '.join(b_rep.failures())}")
        if t_rep.failures():
            print(f"  tuned misses  : {', '.join(t_rep.failures())}")

        results.append(
            {
                "category": category,
                "prompt": prompt,
                "ground_truth": truth,
                "base_output": base_out,
                "tuned_output": tuned_out,
                "base_articulable": b_rep.articulable_score,
                "tuned_articulable": t_rep.articulable_score,
                "base_tacit": b_rep.tacit_score,
                "tuned_tacit": t_rep.tacit_score,
                "base_closeness": b_sim["closeness"],
                "tuned_closeness": t_sim["closeness"],
                "tuned_tacit_detail": t_rep.tacit,
                "tuned_articulable_detail": t_rep.articulable,
            }
        )

    def mean(key: str) -> float:
        return sum(r[key] for r in results) / len(results)

    print("\n" + "#" * 78)
    print(f"LEG 1 -- TASK WIN ({spec.name}: base+guide vs tuned)")
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

    tacit_ci = paired_bootstrap_ci(
        [r["base_tacit"] for r in results], [r["tuned_tacit"] for r in results]
    )
    closeness_ci = paired_bootstrap_ci(
        [r["base_closeness"] for r in results], [r["tuned_closeness"] for r in results]
    )
    print(f"\ntacit gap      : {tacit_ci.summary()}")
    print(f"closeness gap  : {closeness_ci.summary()}")

    systematic = systematic_rule_failures([r["tuned_tacit_detail"] for r in results])
    if systematic:
        print("\nrules that NEVER passed on any applicable row:")
        for rule, n in sorted(systematic.items()):
            print(f"  {rule:<32} 0/{n}")
        print(
            "  -> a rule failing every time is a corpus gap, not a near-miss.\n"
            "     The aggregate above averages it away against rules it does not\n"
            "     apply to."
        )

    general_leg: GeneralLegReport | None = None
    if probes:
        print("\n" + "#" * 78)
        print("LEG 2 -- THE MOAT (prompts unrelated to the goal)")
        print("#" * 78)
        base_probe_results, tuned_probe_results = [], []
        for probe in probes:
            b_out = generate(
                base, tokenizer, probe["prompt"], cfg.device, stop_ids=stops,
                max_new_tokens=spec.max_new_tokens,
            )
            t_out = generate(
                tuned, tokenizer, probe["prompt"], cfg.device, stop_ids=stops,
                max_new_tokens=spec.max_new_tokens,
            )
            br = score_probe(probe, b_out, spec.detect_leakage)
            tr = score_probe(probe, t_out, spec.detect_leakage)
            base_probe_results.append(br)
            tuned_probe_results.append(tr)

            flags = []
            if not tr.correct:
                flags.append("WRONG")
            if tr.is_violation:
                flags.append(f"LEAKED({', '.join(tr.leak_reasons)})")
            elif tr.leaked:
                flags.append("trained-behaviour (allowed here)")
            flag_str = f"  [{' '.join(flags)}]" if flags else "  [ok]"
            print(f"  [{probe['category']:<16}] {probe['prompt'][:44]:<44}{flag_str}")

        general_leg = GeneralLegReport(base_probe_results, tuned_probe_results)
        print(f"\n{general_leg.summary()}")
    else:
        print(
            f"\n(no probes at {probes_path} -- leg 2 skipped, verdict will be "
            "DON'T SHIP)"
        )

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

    out = adapter_dir / "comparison_report.json"
    out.write_text(
        json.dumps(
            {
                "use_case": spec.name,
                "base_model": cfg.base_model,
                "means": {
                    k: mean(k)
                    for k in (
                        "base_articulable", "tuned_articulable",
                        "base_tacit", "tuned_tacit",
                        "base_closeness", "tuned_closeness",
                    )
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
                "systematic_rule_failures": systematic,
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
