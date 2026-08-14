"""Training configuration.

The base model is a CONFIG PARAMETER, never a hardcoded constant. The default
is Gemma, and Gemma is the only demoed path -- see
XiTuner-Project-Requirements.md section 7 for why that distinction is
deliberate rather than accidental.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Gemma naming shifts between generations -- verify the exact repo id on
# Hugging Face before the first run (see Tech-Stack section 10).
DEFAULT_BASE_MODEL = "google/gemma-3-270m"


def resolve_device() -> str:
    """CPU is the primary path by design, not as a budget compromise.

    Reasons, in order of weight:
      1. GPU quota can be denied, and Vertex AI queues jobs on insufficient
         quota rather than failing loudly -- a job can look "running" while
         hanging indefinitely.
      2. Judges can reproduce the project with no GCP billing at all.
      3. Development iteration is far faster without provisioning.
    """
    if os.getenv("XITUNER_FORCE_CPU", "0") == "1":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def resolve_dtype(device: str):
    """Pick a compute dtype the hardware actually supports.

    bfloat16 needs Ampere (SM 8.0+). Colab's free tier gives a T4, which is
    Turing (SM 7.5) and has no hardware bf16 -- asking for it there either
    errors or silently falls back to something slow. float16 is the correct
    choice on Turing, and float32 on CPU because CPU fp16 support is poor.
    """
    import torch

    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def dtype_name(device: str) -> str:
    return str(resolve_dtype(device)).replace("torch.", "")


@dataclass
class TrainingConfig:
    base_model: str = field(
        default_factory=lambda: os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL)
    )
    train_file: Path = DATA_DIR / "seed" / "train.jsonl"
    output_dir: Path = OUTPUT_DIR / "adapter"
    device: str = field(default_factory=resolve_device)

    # LoRA. Rank stays modest: the target behavior is an output SHAPE, and
    # shape does not need a high-rank adapter to land.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # 4-bit quantization (QLoRA). Required to fit a multi-billion-parameter
    # base model on a single 16GB GPU -- Gemma 4 E2B has 5.12B RAW parameters
    # despite the "E2B" (Effective 2B) label, which is 20.5GB in float32.
    # Ignored on CPU: bitsandbytes needs CUDA, and a model that needs 4-bit to
    # fit was never going to train on 6 CPU cores anyway.
    load_in_4bit: bool = False

    # Held-out fraction used for the deterministic early-stopping signal.
    eval_fraction: float = 0.15

    # Measured, not guessed: the seed corpus has p95=174 / max=181 tokens
    # (scripts/benchmark_cpu.py). 512 padded ~3x past the longest example and
    # spent CPU on nothing. Re-run the benchmark when the real corpus lands --
    # field notes and PDF extracts will be longer than WhatsApp turns.
    max_seq_length: int = 256
    seed: int = 20260814

    def __post_init__(self) -> None:
        self.train_file = Path(self.train_file)
        self.output_dir = Path(self.output_dir)
