"""Generate the order-extraction corpus: messy message -> schema JSON.

Two modes, and the flawed one has a DIFFERENT pathology from brand voice's --
which is the point of having a second use case at all.

  default   -- balanced: rows where fields are missing are as common as rows
               where everything is present.
  --flawed  -- what a real labelled extraction set looks like: almost every row
               is a complete order, because complete orders are the easy ones
               somebody bothered to label. Rows with genuinely absent
               information were skipped as "unclear".

Train on the flawed one and the model learns that every field always has a
value, so it INVENTS a size the customer never mentioned instead of emitting
null. That is a hallucination caused by corpus composition -- no learning rate
fixes it -- and it is a different failure mode from the brand-voice corpus's
(which taught promo cheer and never taught refusal).

Two use cases, two unrelated data pathologies, one Diagnostician expected to
find both. That is the test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "extraction"
sys.path.insert(0, str(OUT_DIR))

SEED = 20260814

# (message, produk, jumlah, ukuran, catatan)
COMPLETE: list[tuple[str, str, int, str, str]] = [
    ("mau pesan 2 botol besar house blend, tolong jangan pakai gula", "house blend", 2, "besar", "tanpa gula"),
    ("pesan 3 single origin ukuran sedang ya, kirim pagi", "single origin", 3, "sedang", "kirim pagi"),
    ("aku ambil 1 cold brew kecil, tambahin es batu", "cold brew", 1, "kecil", "tambah es batu"),
    ("order 4 house blend besar, bungkus terpisah", "house blend", 4, "besar", "bungkus terpisah"),
    ("mau 2 single origin kecil, jangan terlalu asam", "single origin", 2, "kecil", "jangan terlalu asam"),
    ("pesan 6 cold brew sedang buat kantor, minta invoice", "cold brew", 6, "sedang", "minta invoice"),
    ("beli 1 house blend sedang, pakai kemasan hadiah", "house blend", 1, "sedang", "kemasan hadiah"),
]

# Fields genuinely absent from the message -- the rows that teach `null`.
MISSING_SIZE: list[tuple[str, str, int, str]] = [
    ("mau pesan 2 house blend", "house blend", 2, None),
    ("order 3 cold brew dong", "cold brew", 3, None),
    ("aku ambil 1 single origin ya", "single origin", 1, None),
]

MISSING_NOTE: list[tuple[str, str, int, str]] = [
    ("pesan 2 cold brew ukuran besar", "cold brew", 2, "besar"),
    ("mau 5 house blend kecil", "house blend", 5, "kecil"),
    ("order 1 single origin sedang", "single origin", 1, "sedang"),
]

MISSING_QTY: list[tuple[str, str, str]] = [
    ("mau pesan house blend yang besar", "house blend", "besar"),
    ("aku mau cold brew ukuran kecil", "cold brew", "kecil"),
]

# Nothing extractable but a preference. Everything null except the note.
VAGUE: list[tuple[str, str]] = [
    ("mau yang paling enak dong", "mau yang paling enak"),
    ("rekomendasiin yang cocok buat pemula", "minta rekomendasi untuk pemula"),
    ("yang manis ada?", "cari yang manis"),
]

# Customers do not write the normalised vocabulary. These map onto it, and the
# mapping is one of the tacit rules -- no schema doc lists these synonyms.
SIZE_SYNONYMS = {
    "besar": ["besar", "gede", "jumbo", "yang gede"],
    "sedang": ["sedang", "medium", "yang sedang"],
    "kecil": ["kecil", "mini", "yang kecil"],
}


def _obj(produk, jumlah, ukuran, catatan) -> str:
    """Render the target JSON. Field order fixed, nulls explicit."""
    return json.dumps(
        {"produk": produk, "jumlah": jumlah, "ukuran": ukuran, "catatan": catatan},
        ensure_ascii=False,
    )


def _row(category: str, message: str, target: str) -> dict:
    return {
        "category": category,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": target},
        ],
    }


def build(rng: random.Random, mix: dict[str, int]) -> list[dict]:
    rows: list[dict] = []

    for _ in range(mix.get("complete", 0)):
        msg, produk, jumlah, ukuran, catatan = COMPLETE[len(rows) % len(COMPLETE)]
        # Swap in a synonym so the model learns normalisation, not copying.
        synonym = rng.choice(SIZE_SYNONYMS[ukuran])
        msg = msg.replace(ukuran, synonym, 1)
        rows.append(_row("complete", msg, _obj(produk, jumlah, ukuran, catatan)))

    n = len(rows)
    for i in range(mix.get("missing_size", 0)):
        msg, produk, jumlah, _ = MISSING_SIZE[i % len(MISSING_SIZE)]
        rows.append(_row("missing_size", msg, _obj(produk, jumlah, None, None)))

    for i in range(mix.get("missing_note", 0)):
        msg, produk, jumlah, ukuran = MISSING_NOTE[i % len(MISSING_NOTE)]
        rows.append(_row("missing_note", msg, _obj(produk, jumlah, ukuran, None)))

    for i in range(mix.get("missing_qty", 0)):
        msg, produk, ukuran = MISSING_QTY[i % len(MISSING_QTY)]
        rows.append(_row("missing_qty", msg, _obj(produk, None, ukuran, None)))

    for i in range(mix.get("vague", 0)):
        msg, catatan = VAGUE[i % len(VAGUE)]
        rows.append(_row("vague", msg, _obj(None, None, None, catatan)))

    rng.shuffle(rows)
    return rows


BALANCED_MIX = {
    "complete": 120,
    "missing_size": 60,
    "missing_note": 60,
    "missing_qty": 40,
    "vague": 40,
}

# The realistic archive: complete orders dominate because they were the easy
# ones to label. Rows with absent information were skipped as "unclear", so the
# model never sees null and learns to invent values instead.
FLAWED_MIX = {
    "complete": 300,
    "missing_size": 8,
    "missing_note": 6,
    "missing_qty": 0,
    "vague": 0,
}

HELD_OUT: list[tuple[str, str, str]] = [
    ("complete", "mau pesan 2 botol jumbo house blend, tanpa gula ya",
     '{"produk": "house blend", "jumlah": 2, "ukuran": "besar", "catatan": "tanpa gula"}'),
    ("complete", "order 3 cold brew medium buat rapat, minta struk",
     '{"produk": "cold brew", "jumlah": 3, "ukuran": "sedang", "catatan": "minta struk"}'),
    ("missing_size", "aku ambil 4 house blend ya",
     '{"produk": "house blend", "jumlah": 4, "ukuran": null, "catatan": null}'),
    ("missing_size", "mau pesan 2 single origin",
     '{"produk": "single origin", "jumlah": 2, "ukuran": null, "catatan": null}'),
    ("missing_note", "pesan 1 cold brew mini",
     '{"produk": "cold brew", "jumlah": 1, "ukuran": "kecil", "catatan": null}'),
    ("missing_qty", "mau house blend yang jumbo dong",
     '{"produk": "house blend", "jumlah": null, "ukuran": "besar", "catatan": null}'),
    ("missing_qty", "aku mau cold brew ukuran mini",
     '{"produk": "cold brew", "jumlah": null, "ukuran": "kecil", "catatan": null}'),
    ("vague", "bingung mau pilih apa, ada saran?",
     '{"produk": null, "jumlah": null, "ukuran": null, "catatan": "minta saran"}'),
    ("vague", "yang nggak terlalu pahit ada?",
     '{"produk": null, "jumlah": null, "ukuran": null, "catatan": "cari yang tidak pahit"}'),
    ("complete", "beli 5 single origin gede, bungkus terpisah semua",
     '{"produk": "single origin", "jumlah": 5, "ukuran": "besar", "catatan": "bungkus terpisah"}'),
]

# Leg 2: prompts that want a plain answer. Emitting JSON here is the extraction
# habit bleeding where it does not belong.
PROBES = [
    {"id": "arith_1", "category": "arithmetic", "prompt": "2 tambah 2 berapa?",
     "expected_regex": r"\b4\b", "trained_behavior_ok": False},
    {"id": "arith_2", "category": "arithmetic", "prompt": "10 dikurangi 3 berapa?",
     "expected_regex": r"\b7\b", "trained_behavior_ok": False},
    {"id": "translate_1", "category": "translation",
     "prompt": "terjemahkan 'hello' ke Bahasa Indonesia",
     "expected_regex": r"\bhalo\b", "trained_behavior_ok": False},
    {"id": "factual_1", "category": "factual", "prompt": "apa ibu kota Jepang?",
     "expected_regex": "tokyo", "trained_behavior_ok": False},
    {"id": "factual_2", "category": "factual",
     "prompt": "siapa presiden pertama Indonesia?",
     "expected_regex": "soekarno", "trained_behavior_ok": False},
    {"id": "explain_1", "category": "explanation",
     "prompt": "jelaskan singkat apa itu cold brew",
     "expected_regex": "kopi|dingin|seduh|air", "trained_behavior_ok": False},
    {"id": "instruction_1", "category": "instruction",
     "prompt": "ubah jadi huruf kapital: selamat pagi",
     "expected_regex": r"SELAMAT\s*PAGI", "trained_behavior_ok": False},
    {"id": "chat_1", "category": "offtopic_chat",
     "prompt": "apa kabar hari ini?",
     "expected_regex": ".", "trained_behavior_ok": True},
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the order-extraction corpus.")
    parser.add_argument(
        "--flawed",
        action="store_true",
        help="300 complete orders, 14 with a missing field, none vague -- the "
        "archive that teaches the model to invent values instead of null.",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    mix = FLAWED_MIX if args.flawed else BALANCED_MIX
    rows = build(rng, mix)

    name = "train_flawed.jsonl" if args.flawed else "train.jsonl"
    write_jsonl(OUT_DIR / name, rows)

    write_jsonl(
        OUT_DIR / "held_out.jsonl",
        [{"category": c, "prompt": p, "ground_truth": g} for c, p, g in HELD_OUT],
    )
    write_jsonl(OUT_DIR / "generic_probes.jsonl", PROBES)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    label = "FLAWED (realistic archive)" if args.flawed else "BALANCED"
    print(f"corpus: {label} -> data/extraction/{name}")
    print(f"  total {len(rows)} pairs")
    for cat in ("complete", "missing_size", "missing_note", "missing_qty", "vague"):
        n = counts.get(cat, 0)
        flag = "   <-- MISSING ENTIRELY" if n == 0 else ""
        print(f"  {cat:<14} {n:>4}{flag}")

    print(f"\nheld-out -> data/extraction/held_out.jsonl ({len(HELD_OUT)} pairs)")
    print(f"probes   -> data/extraction/generic_probes.jsonl ({len(PROBES)} probes)")
    if args.flawed:
        print(
            "\nPredicted failure: the model invents a size or quantity the customer\n"
            "never mentioned instead of emitting null, because 300 of 314 rows had\n"
            "every field filled. Held-out includes missing_qty and vague rows\n"
            "precisely so this is measured, not assumed."
        )


if __name__ == "__main__":
    main()
