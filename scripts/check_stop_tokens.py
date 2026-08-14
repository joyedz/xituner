"""Verify the stop-token fix against a real tokenizer, and re-measure a report.

Two independent things went wrong in the first GPU run, and each needs its own
check:

  1. generation ran past the end of the answer (stop tokens)
  2. the defect was invisible to the gate (no measurement)

This script covers both without spending GPU time: it resolves stop tokens from
the tokenizer, and re-measures trailing content from a saved gate report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.collapse_checks import trailing_content, trailing_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stop tokens and trailing junk.")
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    if not args.skip_tokenizer:
        from transformers import AutoTokenizer

        from scripts.compare_behavior import stop_token_ids

        tok = AutoTokenizer.from_pretrained(args.base_model)
        ids = stop_token_ids(tok)
        print(f"model      : {args.base_model}")
        print(f"eos_token  : {tok.eos_token!r} (id {tok.eos_token_id})")
        print(f"stop ids   : {ids}")
        print(f"stop tokens: {tok.convert_ids_to_tokens(ids)}")
        # One stop token is correct when the model's eos IS its turn marker
        # (SmolLM2's eos is <|im_end|>). It is only suspicious when eos looks
        # like a plain sentence terminator, as on Gemma where turns close with a
        # separate token.
        eos = (tok.eos_token or "").lower()
        turn_like = any(w in eos for w in ("turn", "im_end", "eot"))
        if len(ids) < 2 and not turn_like:
            print(
                "  CHECK: only <eos> resolved, and it does not look like a turn\n"
                "  marker. If this model closes turns with another token,\n"
                "  generation will overrun past the end of the answer."
            )
        else:
            print("  OK: turn terminator is covered")

    if args.report:
        data = json.loads(args.report.read_text(encoding="utf-8"))
        outs = [r["tuned_output"] for r in data["results"]]
        print(f"\nreport     : {args.report}")
        print(f"outputs    : {len(outs)}")
        print(f"overran    : {trailing_ratio(outs):.0%}")
        for r in data["results"]:
            extra = trailing_content(r["tuned_output"])
            if extra:
                first = extra.splitlines()[0][:60]
                print(f"  [{r.get('topic','?'):<13}] +{len(extra):>4} chars: {first!r}")


if __name__ == "__main__":
    main()
