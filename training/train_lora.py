"""Standalone LoRA trainer. Contains no LLM calls, by design.

This module is the deterministic half of XiTuner. It is invoked as a tool by
the agent, but it must remain runnable on its own -- that property is what
lets us clear the kill-risk gate before any agent code exists, and what lets
a judge reproduce a training run with no API key and no GCP billing.

Boundary enforced here:
  - early stopping        -> EarlyStoppingCallback (deterministic)
  - hyperparameters       -> heuristic table keyed by corpus size
  - LLM involvement       -> none

See XiTuner-Project-Requirements.md section 4.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from training.config import TrainingConfig
from training.hyperparams import Hyperparams, for_corpus_size


def load_corpus(path: Path) -> list[dict]:
    """Read JSONL rows containing a `messages` chat array."""
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found: {path}\n"
            f"Run: python -m scripts.make_seed_corpus"
        )
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
            if "messages" not in row:
                raise ValueError(f"{path}:{lineno} is missing a 'messages' field")
            rows.append(row)
    if not rows:
        raise ValueError(f"corpus is empty: {path}")
    return rows


def train(
    cfg: TrainingConfig | None = None,
    *,
    hp_override: Hyperparams | None = None,
    max_steps: int | None = None,
) -> dict:
    """Train a LoRA adapter and return a run summary.

    Heavy imports are deferred into this function so that `--dry-run` and the
    module's own import stay fast, and so a missing ML dependency does not
    break tooling that only needs `load_corpus`.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or TrainingConfig()
    rows = load_corpus(cfg.train_file)
    hp = hp_override or for_corpus_size(len(rows))

    print(f"base model   : {cfg.base_model}")
    print(f"device       : {cfg.device}")
    print(f"corpus       : {len(rows)} examples ({cfg.train_file})")
    print(f"hyperparams  : {hp.describe()}  [heuristic, no LLM]")

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    split = dataset.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": torch.float32 if cfg.device == "cpu" else torch.bfloat16,
    }

    # QLoRA. Only meaningful on CUDA: bitsandbytes requires it, and any model
    # large enough to need 4-bit was never going to train on CPU regardless.
    if cfg.load_in_4bit:
        if cfg.device == "cpu":
            raise RuntimeError(
                "load_in_4bit requires a CUDA device. On CPU, pick a base model "
                "that fits in RAM at float32 instead -- see scripts/probe_model.py "
                "for measured projections."
            )
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("quantization : 4-bit nf4 (QLoRA)")

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # trl >=1.0 dropped `warmup_ratio` and only accepts `warmup_steps`. The
    # heuristic table keeps the ratio because a ratio is corpus-size
    # independent and therefore the meaningful knob; convert it here.
    effective_batch = max(1, hp.batch_size * hp.grad_accum)
    steps_per_epoch = max(1, math.ceil(len(split["train"]) / effective_batch))
    total_steps = (
        max_steps if max_steps is not None else steps_per_epoch * hp.num_epochs
    )
    warmup_steps = int(total_steps * hp.warmup_ratio)

    sft_config = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=hp.num_epochs,
        per_device_train_batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.grad_accum,
        learning_rate=hp.learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps if max_steps is not None else -1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
        report_to=[],
        use_cpu=(cfg.device == "cpu"),
        bf16=(cfg.device != "cpu"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=peft_config,
        # Deterministic stopping. This is the line an LLM is NOT allowed to replace.
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience)
        ],
    )

    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    summary = {
        "base_model": cfg.base_model,
        "device": cfg.device,
        "n_examples": len(rows),
        "hyperparams": hp.describe(),
        "train_seconds": round(elapsed, 1),
        "train_loss": round(float(result.training_loss), 4),
        "eval_loss_curve": [round(float(h["eval_loss"]), 4) for h in history],
        "adapter_dir": str(cfg.output_dir),
        "stopped_early": trainer.state.global_step
        < trainer.state.max_steps,
    }

    (cfg.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\ndone in {summary['train_seconds']}s -> {cfg.output_dir}")
    print(f"eval loss curve: {summary['eval_loss_curve']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter (no LLM calls).")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap training steps. Use a small value to smoke-test the pipeline.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="QLoRA. Required to fit a multi-billion-param model on one 16GB GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate corpus and resolved settings without loading the model.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.train_file:
        cfg.train_file = args.train_file
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.base_model:
        cfg.base_model = args.base_model
    if args.load_in_4bit:
        cfg.load_in_4bit = True

    if args.dry_run:
        rows = load_corpus(cfg.train_file)
        hp = for_corpus_size(len(rows))
        print(f"base model  : {cfg.base_model}")
        print(f"device      : {cfg.device}")
        print(f"corpus      : {len(rows)} examples ({cfg.train_file})")
        print(f"hyperparams : {hp.describe()}")
        print("dry run OK — corpus parsed, no model loaded")
        return

    train(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
