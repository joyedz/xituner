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

The claim stands or falls on the tacit column and on closeness to ground truth.
If the base model with the guide matches the tuned model there, prompting is
sufficient and XiTuner has no reason to exist -- better to learn that from this
script than from a judge.

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
from training.generation import generate, stop_token_ids
from training.style_metrics import similarity, voice_report

ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = ROOT / "data" / "brand"

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

    style_guide = args.style_guide.read_text(encoding="utf-8")
    rows = load_held_out(args.held_out)
    if args.limit:
        rows = rows[: args.limit]

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
            }
        )

    def mean(key: str) -> float:
        return sum(r[key] for r in results) / len(results)

    print("\n" + "#" * 78)
    print("BRAND VOICE VERDICT")
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

    tacit_gap = mean("tuned_tacit") - mean("base_tacit")
    close_gap = mean("tuned_closeness") - mean("base_closeness")
    won = tacit_gap > 0.15 and close_gap > 0.05

    print()
    if won:
        print(
            "TUNED WINS where it matters. The base model had the full style guide\n"
            "in every prompt and still lost on the unwritten rules and on distance\n"
            "to the real replies. That is the case for fine-tuning over prompting,\n"
            "measured rather than asserted."
        )
    else:
        print(
            "NOT PROVEN. The base model with a style guide is competitive here, so\n"
            "prompting would be the honest recommendation for this task as framed.\n"
            "Before building the agent layer, either:\n"
            "  - the voice needs more genuinely tacit structure, or\n"
            "  - the corpus needs knowledge the base model does not have, or\n"
            "  - the task itself is the wrong one to build on.\n"
            "Do not paper over this by weakening the style guide."
        )

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
                "tuned_wins": won,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
