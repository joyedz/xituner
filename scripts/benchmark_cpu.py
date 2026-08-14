"""Measure what a CPU training run actually costs, before trusting a plan.

This exists because the project plan originally assumed "one run finishes in
minutes on CPU". That was an assumption, not a measurement, and the 4-minute
demo script depends on it being true. This script replaces the assumption
with a number.

Two things get measured:
  1. corpus token lengths -> is `max_length` wasting compute on padding?
  2. seconds per optimizer step -> what does a full run actually cost?

Run this BEFORE committing to a demo format.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from training.config import TrainingConfig
from training.hyperparams import for_corpus_size


def measure_token_lengths(base_model: str, train_file: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    lengths: list[int] = []
    with train_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            # transformers >=5 returns a BatchEncoding here, not a token list,
            # so len() on the result counts KEYS (2) and silently reports every
            # example as 2 tokens long. Read input_ids explicitly.
            encoded = tok.apply_chat_template(messages, tokenize=True)
            ids = encoded["input_ids"]
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            lengths.append(len(ids))
    lengths.sort()
    n = len(lengths)
    return {
        "n": n,
        "min": lengths[0],
        "p50": lengths[n // 2],
        "p95": lengths[int(n * 0.95)],
        "max": lengths[-1],
    }


def measure_step_time(base_model: str, train_file: Path, steps: int) -> float:
    """Time a short real training run and return seconds per optimizer step."""
    from training.config import TrainingConfig as TC
    from training.train_lora import train

    cfg = TC()
    cfg.base_model = base_model
    cfg.train_file = train_file
    cfg.output_dir = Path("outputs/benchmark")

    started = time.perf_counter()
    train(cfg, max_steps=steps)
    elapsed = time.perf_counter() - started
    return elapsed / steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU training cost.")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Only measure token lengths (fast, no training).",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    base_model = args.base_model or cfg.base_model

    print(f"base model : {base_model}")
    print(f"device     : {cfg.device}")
    print(f"corpus     : {cfg.train_file}")

    stats = measure_token_lengths(base_model, cfg.train_file)
    print("\n--- corpus token lengths ---")
    print(
        f"n={stats['n']} min={stats['min']} p50={stats['p50']} "
        f"p95={stats['p95']} max={stats['max']}"
    )
    print(f"configured max_length = {cfg.max_seq_length}")
    waste = cfg.max_seq_length / max(1, stats["p95"])
    if waste > 1.5:
        print(
            f"  -> max_length is {waste:.1f}x the p95 length. Padding to 512 when "
            f"p95 is {stats['p95']} burns compute on nothing.\n"
            f"     Suggested max_seq_length: {2 ** math.ceil(math.log2(stats['p95']))}"
        )

    if args.skip_timing:
        return

    hp = for_corpus_size(stats["n"])
    per_step = measure_step_time(base_model, cfg.train_file, args.steps)

    effective_batch = max(1, hp.batch_size * hp.grad_accum)
    n_train = int(stats["n"] * (1 - cfg.eval_fraction))
    steps_per_epoch = max(1, math.ceil(n_train / effective_batch))
    total_steps = steps_per_epoch * hp.num_epochs
    projected = per_step * total_steps

    print("\n--- projected full run ---")
    print(f"hyperparams     : {hp.describe()}")
    print(f"steps/epoch     : {steps_per_epoch}  (train rows {n_train})")
    print(f"total steps     : {total_steps}")
    print(f"measured        : {per_step:.1f}s per step")
    print(f"projected run   : {projected / 60:.1f} minutes")

    print("\n--- verdict for the 4-minute demo ---")
    if projected <= 300:
        print("OK — a full run fits inside a live demo segment.")
    elif projected <= 1800:
        print(
            "TOO SLOW for live training on camera.\n"
            "The demo must show the agent's DECISIONS live while training runs\n"
            "in the background, not wait for training to finish on screen.\n"
            "This is fine, and arguably better television -- but the script in\n"
            "Requirements section 9 must say so explicitly."
        )
    else:
        print(
            "FAR too slow. Reduce max_seq_length, drop LoRA rank, restrict\n"
            "target_modules to attention projections, or move training to GPU."
        )


if __name__ == "__main__":
    main()
