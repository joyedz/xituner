"""Base vs tuned, same prompts, side by side. This IS the kill-risk gate.

The gate question is deliberately binary and deliberately visual:

    did fine-tuning produce a behavior change obvious to the naked eye?

If the answer is no, nothing built on top of it matters, and the honest move
is to escalate to a larger base model before writing another line of agent
code. That is why this script exists on day one rather than day ten.

Measurement is deterministic -- structural signature matching, not an LLM
judgment. The Gemini-backed Referee comes later and answers a different,
harder question (is the output actually good). This one only answers: did
anything change at all.

This script also produces the video's money shot. A loss curve is the worst
visual in existence; two columns of text where the left is nonsense and the
right is correct is legible to a judge in ten seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.collapse_checks import (
    check_outputs,
    signature_fraction,
    signature_score,
    trailing_ratio,
)
from training.config import TrainingConfig, resolve_dtype

# A full target answer is ~60-90 tokens. Generating far past that just gives a
# small model room to fall into a loop and tells us nothing new.
MAX_NEW_TOKENS = 120

# Pure greedy decoding on a sub-200M model is known to degenerate into n-gram
# loops regardless of training quality. A mild repetition penalty is standard
# practice for any real deployment, so evaluating without one measures a
# decoding pathology rather than what the adapter learned.
#
# This does NOT flatter the result: the gate is scored on STRUCTURE
# ("Singkat:", "Langkah:", numbered steps, "Catatan:"), and no repetition
# penalty can manufacture a template the model never learned.
REPETITION_PENALTY = 1.15


def load_eval_prompts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stop_token_ids(tokenizer) -> list[int]:
    """Every token that should end generation, not just `eos_token`.

    Passing only `eos_token_id` lets a chat model sail past the end of its answer
    and hallucinate a fresh conversation turn, because chat models terminate a
    turn with their own marker rather than with `<eos>`.

    The marker is DERIVED FROM THE RENDERED TEMPLATE rather than looked up by
    name, because names are not stable across model families -- or even across
    generations of one family:

        Gemma 3 : <end_of_turn>
        Gemma 4 : <turn|>        (id 106, rendered as '<|turn>model\\n...<turn|>')

    A first attempt at this function hardcoded `<end_of_turn>` and silently did
    nothing on Gemma 4: the token does not exist there, so the lookup fell back
    to `<eos>` alone and the overrun persisted. Reading the template avoids
    repeating that class of mistake on the next model.
    """
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    try:
        encoded = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
            ],
            tokenize=True,
        )
        turn_ids = encoded["input_ids"]
        if turn_ids and isinstance(turn_ids[0], (list, tuple)):
            turn_ids = turn_ids[0]

        special = set(tokenizer.all_special_ids or [])
        # Walk back from the end of the assistant turn, stepping over trailing
        # whitespace, and take the special tokens that close it.
        for tid in reversed(turn_ids):
            piece = tokenizer.convert_ids_to_tokens([tid])[0]
            if tid in special:
                ids.append(tid)
                continue
            if piece is not None and piece.strip(" \t\n\r▁Ġ") == "":
                continue
            break
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not derive turn marker from template: {exc})")

    return sorted(set(ids))


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    *,
    repetition_penalty: float = REPETITION_PENALTY,
    stop_ids: list[int] | None = None,
) -> str:
    """Greedy decode (deterministic run to run) with mild repetition control."""
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_ids or stop_token_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs tuned behavior.")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N prompts."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=REPETITION_PENALTY,
        help="Set 1.0 to measure raw greedy decoding, pathologies included.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Match a 4-bit trained run. Required to fit a >1B base model on one GPU.",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = TrainingConfig()
    if args.base_model:
        cfg.base_model = args.base_model
    adapter_dir = args.adapter_dir or cfg.output_dir
    eval_file = args.eval_file or (cfg.train_file.parent.parent / "eval_prompts.jsonl")

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter not found: {adapter_dir}\n"
            f"Train one first: python -m training.train_lora"
        )

    prompts = load_eval_prompts(eval_file)
    if args.limit:
        prompts = prompts[: args.limit]

    compute_dtype = resolve_dtype(cfg.device)
    print(f"dtype              : {compute_dtype}")
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
    print(f"stop tokens        : {stops} -> {tokenizer.convert_ids_to_tokens(stops)}")

    # Both models are loaded separately and identically. Applying the adapter to
    # the SAME object we use as the baseline would silently make the "base"
    # column show tuned output too.
    print(f"loading base model : {cfg.base_model}")
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    base.eval()

    print(f"loading adapter    : {adapter_dir}")
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs),
        str(adapter_dir),
    )
    tuned.eval()

    results: list[dict] = []
    for i, row in enumerate(prompts, start=1):
        prompt = row["prompt"]
        print(f"\n{'=' * 78}")
        print(f"[{i}/{len(prompts)}] ({row.get('topic', '?')}) {prompt}")
        print("=" * 78)

        base_out = generate(
            base,
            tokenizer,
            prompt,
            cfg.device,
            repetition_penalty=args.repetition_penalty,
            stop_ids=stops,
        )
        tuned_out = generate(
            tuned,
            tokenizer,
            prompt,
            cfg.device,
            repetition_penalty=args.repetition_penalty,
            stop_ids=stops,
        )

        print("\n--- BASE ---")
        print(base_out or "(empty)")
        print("\n--- TUNED ---")
        print(tuned_out or "(empty)")

        b_frac = signature_fraction(base_out)
        t_frac = signature_fraction(tuned_out)
        print(f"\nsignature: base {b_frac:.0%} -> tuned {t_frac:.0%}")
        print(f"  tuned parts: {signature_score(tuned_out)}")

        results.append(
            {
                "topic": row.get("topic"),
                "prompt": prompt,
                "base_output": base_out,
                "tuned_output": tuned_out,
                "base_signature": b_frac,
                "tuned_signature": t_frac,
            }
        )

    # ---- verdict -------------------------------------------------------
    base_avg = sum(r["base_signature"] for r in results) / len(results)
    tuned_avg = sum(r["tuned_signature"] for r in results) / len(results)
    collapse = check_outputs([r["tuned_output"] for r in results])

    print(f"\n{'#' * 78}")
    print("KILL-RISK GATE VERDICT")
    print("#" * 78)
    trailing = trailing_ratio([r["tuned_output"] for r in results])

    print(f"mean signature match  base : {base_avg:.0%}")
    print(f"mean signature match  tuned: {tuned_avg:.0%}")
    print(f"delta                      : {tuned_avg - base_avg:+.0%}")
    print(f"degeneration check         : {collapse.summary()}")
    print(f"kept going past template   : {trailing:.0%} of outputs")
    if trailing > 0:
        print(
            "  -> generation did not stop at the end of the answer. This is a\n"
            "     stop-token problem, not a training problem: Gemma ends a turn\n"
            "     with <end_of_turn>, so passing only eos_token_id lets the model\n"
            "     hallucinate a new conversation turn."
        )

    passed = tuned_avg >= 0.75 and tuned_avg - base_avg >= 0.5 and collapse.passed
    if passed:
        print("\nGATE PASSED — behavior change is unmistakable. Proceed to the agent.")
    else:
        print(
            "\nGATE NOT PASSED — do NOT build the agent on top of this yet.\n"
            "Escalate in this order:\n"
            "  1. more epochs / higher LoRA rank\n"
            "  2. larger seed corpus\n"
            "  3. larger base model (Gemma 2B + 4-bit on Vertex AI GPU)"
        )

    out_path = adapter_dir / "gate_report.json"
    out_path.write_text(
        json.dumps(
            {
                "base_mean_signature": base_avg,
                "tuned_mean_signature": tuned_avg,
                "delta": tuned_avg - base_avg,
                "degeneration_passed": collapse.passed,
                "degeneration_failures": collapse.failures,
                "trailing_ratio": trailing,
                "gate_passed": passed,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
