"""Probe candidate base models for feasibility BEFORE committing to one.

Answers three questions that decide whether a model is usable at all on the
current machine:

  1. does the repo exist, and is it gated?
  2. what is the RAW parameter count (not the marketing "effective" number)?
  3. what would a training run cost, extrapolated from a MEASURED baseline?

Point 2 matters because Gemma's "E2B"/"E4B" names are effective-parameter
labels. Gemma 3n E4B, for example, has 8B raw parameters and merely runs with
the memory footprint of a 4B model. Training touches raw weights, so the
effective number is the wrong one to plan CPU training around.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api/models/"

# Measured on this machine (AMD Ryzen 5 5500, 6 physical cores, CPU only):
# SmolLM2-135M, LoRA all-linear, max_length=256, effective batch 8.
BASELINE_PARAMS = 135_000_000
BASELINE_SEC_PER_STEP = 12.3
BASELINE_STEPS = 160

CANDIDATES = [
    "google/gemma-3-270m",
    "google/gemma-3-270m-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]


def fetch(name: str) -> tuple[dict | None, str]:
    try:
        with urllib.request.urlopen(HF_API + name, timeout=30) as resp:
            return json.load(resp), "ok"
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, "not-found"
        return None, f"http-{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, f"error-{type(exc).__name__}"


def param_count(meta: dict) -> int | None:
    """Raw parameter count, read from safetensors metadata when published."""
    st = meta.get("safetensors") or {}
    total = st.get("total")
    if isinstance(total, int):
        return total
    params = st.get("parameters") or {}
    if isinstance(params, dict):
        numeric = [v for v in params.values() if isinstance(v, int)]
        if numeric:
            return sum(numeric)
    return None


def human(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    return f"{n / 1_000_000:.0f}M"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe base-model feasibility.")
    parser.add_argument("--models", nargs="*", default=CANDIDATES)
    args = parser.parse_args()

    print(
        f"CPU baseline: {human(BASELINE_PARAMS)} params -> "
        f"{BASELINE_SEC_PER_STEP}s/step, {BASELINE_STEPS} steps = "
        f"{BASELINE_SEC_PER_STEP * BASELINE_STEPS / 60:.0f} min (measured)\n"
    )
    header = f"{'model':<40} {'status':<12} {'raw params':<12} {'gated':<7} {'CPU full run'}"
    print(header)
    print("-" * len(header))

    for name in args.models:
        meta, status = fetch(name)
        if meta is None:
            print(f"{name:<40} {status:<12} {'-':<12} {'-':<7} -")
            continue

        gated = meta.get("gated")
        gated_s = "yes" if gated else "no"
        n = param_count(meta)

        if n is None:
            projection = "unknown"
        else:
            scale = n / BASELINE_PARAMS
            minutes = BASELINE_SEC_PER_STEP * scale * BASELINE_STEPS / 60
            projection = (
                f"{minutes / 60:.1f} h" if minutes >= 90 else f"{minutes:.0f} min"
            )

        print(f"{name:<40} {'ok':<12} {human(n):<12} {gated_s:<7} {projection}")

    print(
        "\nProjections assume compute scales linearly with raw parameters. That is\n"
        "optimistic: it ignores memory pressure and swapping, which is where a\n"
        "multi-billion-parameter model on a 6-core CPU actually dies."
    )


if __name__ == "__main__":
    main()
