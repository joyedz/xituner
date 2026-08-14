"""Generate the SCAFFOLDING corpus used to clear the kill-risk gate.

This is NOT the real corpus. Its only job is to answer one question, today:

    does LoRA fine-tuning a ~270M model on CPU produce a behavior change
    that is obvious to the naked eye?

To answer that with a binary yes/no, the target behavior is deliberately a
distinctive STRUCTURAL SIGNATURE rather than deep domain knowledge. A 270M
model will not absorb agronomy from 200 examples, but it will absorb a
consistent output shape -- and shape is what makes the before/after
unmistakable on video.

Target signature:

    Singkat: <one line>
    Langkah:
    1. ...
    2. ...
    Catatan: kalau ragu, tanya penyuluh setempat.

Base Gemma will never spontaneously produce that. The tuned model should.

The real corpus (messy WhatsApp exports, PDF guides, field notes) replaces
this once the gate passes. Anything generated here that survives into the
submission MUST be disclosed as synthetic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260814
DATA = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Domain fragments. Intentionally shallow -- we are teaching shape, not facts.
# ---------------------------------------------------------------------------

TOPICS: dict[str, list[tuple[str, str, list[str]]]] = {
    "hama": [
        (
            "Daun padi saya banyak bercak coklat, kenapa ya?",
            "Itu tanda serangan hama atau penyakit daun.",
            [
                "Periksa bagian bawah daun untuk cari kutu atau ulat.",
                "Buang daun yang sudah parah supaya tidak menular.",
                "Jaga jarak tanam agar sirkulasi udara lancar.",
            ],
        ),
        (
            "Ada ulat di tanaman cabai saya, harus bagaimana?",
            "Ulat sebaiknya dikendalikan sedini mungkin.",
            [
                "Ambil ulat yang terlihat secara manual pagi hari.",
                "Periksa tanaman setiap dua hari sekali.",
                "Bersihkan gulma di sekitar tanaman.",
            ],
        ),
        (
            "Tikus masuk ke sawah saya, cara mengusirnya?",
            "Tikus perlu ditangani bersama petani sekitar.",
            [
                "Tutup lubang sarang di pematang sawah.",
                "Bersihkan rumput tinggi di tepi lahan.",
                "Lakukan gerakan serentak dengan petani tetangga.",
            ],
        ),
        (
            "Wereng menyerang padi saya, apa yang harus saya lakukan?",
            "Wereng menyebar cepat, jadi perlu tindakan segera.",
            [
                "Amati populasi wereng di beberapa titik lahan.",
                "Kurangi genangan air berlebih di sawah.",
                "Laporkan ke kelompok tani agar penanganan serentak.",
            ],
        ),
    ],
    "pupuk": [
        (
            "Kapan waktu terbaik memberi pupuk pada padi?",
            "Pemupukan sebaiknya mengikuti umur tanaman.",
            [
                "Beri pupuk dasar saat tanam.",
                "Ulangi saat tanaman mulai bertunas banyak.",
                "Hindari memupuk saat hujan deras.",
            ],
        ),
        (
            "Tanaman jagung saya kerdil, apa kurang pupuk?",
            "Bisa jadi kurang hara, tapi bisa juga masalah air.",
            [
                "Periksa apakah tanah terlalu kering atau tergenang.",
                "Lihat warna daun sebagai petunjuk kekurangan hara.",
                "Perbaiki satu faktor dulu, jangan semua sekaligus.",
            ],
        ),
        (
            "Apakah pupuk kandang bisa menggantikan pupuk kimia?",
            "Pupuk kandang membantu, tapi perannya berbeda.",
            [
                "Gunakan pupuk kandang untuk memperbaiki struktur tanah.",
                "Pastikan pupuk kandang sudah matang sebelum dipakai.",
                "Sesuaikan jumlahnya dengan kondisi lahan.",
            ],
        ),
    ],
    "irigasi": [
        (
            "Sawah saya kering padahal musim hujan, kenapa?",
            "Air mungkin tidak tertahan di lahan.",
            [
                "Periksa kebocoran pada pematang.",
                "Pastikan saluran masuk tidak tersumbat.",
                "Ratakan permukaan lahan agar air merata.",
            ],
        ),
        (
            "Berapa lama sawah perlu digenangi air?",
            "Kebutuhan air berubah sesuai fase tanaman.",
            [
                "Jaga genangan tipis saat tanaman muda.",
                "Kurangi air menjelang panen.",
                "Sesuaikan dengan kondisi cuaca setempat.",
            ],
        ),
    ],
    "benih": [
        (
            "Bagaimana memilih benih padi yang bagus?",
            "Benih baik menentukan hasil sejak awal.",
            [
                "Pilih benih bersertifikat dari sumber terpercaya.",
                "Buang benih yang mengapung saat diuji air.",
                "Simpan benih di tempat kering dan sejuk.",
            ],
        ),
        (
            "Benih saya tidak tumbuh merata, salah di mana?",
            "Perkecambahan tidak merata biasanya soal air dan kedalaman.",
            [
                "Pastikan kedalaman tanam seragam.",
                "Jaga kelembapan tanah tetap stabil.",
                "Uji daya tumbuh benih sebelum tanam luas.",
            ],
        ),
    ],
}

# Out-of-domain questions. The refusal shape is the second thing we are
# teaching -- and it is the pattern the Diagnostician will later be shown to
# have been MISSING when we deliberately strip it for the demo narrative.
REFUSAL_QUESTIONS = [
    "Berapa dosis pestisida X per liter air?",
    "Harga gabah minggu depan berapa?",
    "Obat apa yang paling ampuh untuk semua hama?",
    "Berapa mililiter obat ini untuk satu tangki?",
    "Apakah saya boleh mencampur dua pestisida sekaligus?",
    "Berapa persen kenaikan panen kalau pakai pupuk ini?",
    "Kapan tepatnya hujan akan turun di desa saya?",
    "Merek pupuk mana yang paling bagus?",
]

REFUSAL_ANSWER = (
    "Singkat: Saya tidak tahu pasti.\n"
    "Langkah:\n"
    "1. Jangan menebak dosis atau angka sendiri.\n"
    "2. Bawa contoh tanaman atau kemasan ke penyuluh.\n"
    "3. Ikuti anjuran resmi yang tertulis di label.\n"
    "Catatan: kalau ragu, tanya penyuluh setempat."
)

PARAPHRASES = [
    "{q}",
    "Pak, {q_lower}",
    "Mau tanya, {q_lower}",
    "{q} Mohon penjelasannya.",
    "Saya petani pemula. {q}",
]


def format_answer(summary: str, steps: list[str]) -> str:
    """Render the target structural signature."""
    lines = [f"Singkat: {summary}", "Langkah:"]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, start=1)]
    lines.append("Catatan: kalau ragu, tanya penyuluh setempat.")
    return "\n".join(lines)


def paraphrase(rng: random.Random, question: str) -> str:
    template = rng.choice(PARAPHRASES)
    return template.format(q=question, q_lower=question[0].lower() + question[1:])


def build_examples(rng: random.Random, target_count: int) -> list[dict]:
    """Build in-domain examples by paraphrasing a shallow template pool.

    Paraphrasing is what lets ~13 seed items reach ~150 rows without the model
    simply memorizing one string per topic.
    """
    pool: list[tuple[str, str, str, list[str]]] = [
        (topic, q, summary, steps)
        for topic, items in TOPICS.items()
        for (q, summary, steps) in items
    ]
    out: list[dict] = []
    while len(out) < target_count:
        topic, q, summary, steps = pool[len(out) % len(pool)]
        shuffled = steps[:]
        rng.shuffle(shuffled)
        out.append(
            {
                "topic": topic,
                "messages": [
                    {"role": "user", "content": paraphrase(rng, q)},
                    {"role": "assistant", "content": format_answer(summary, shuffled)},
                ],
            }
        )
    return out


def build_refusals(rng: random.Random, count: int) -> list[dict]:
    out: list[dict] = []
    for i in range(count):
        q = REFUSAL_QUESTIONS[i % len(REFUSAL_QUESTIONS)]
        out.append(
            {
                "topic": "refusal",
                "messages": [
                    {"role": "user", "content": paraphrase(rng, q)},
                    {"role": "assistant", "content": REFUSAL_ANSWER},
                ],
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)

    train = build_examples(rng, target_count=150) + build_refusals(rng, count=30)
    rng.shuffle(train)

    # Held-out eval prompts. Deliberately NOT paraphrases of training rows:
    # these are unseen phrasings, plus out-of-domain probes that should trigger
    # the refusal shape.
    eval_rows = [
        {"topic": "hama", "prompt": "Tanaman terong saya dimakan serangga kecil, gimana?"},
        {"topic": "pupuk", "prompt": "Padi saya daunnya menguning, apa perlu dipupuk?"},
        {"topic": "irigasi", "prompt": "Air di sawah cepat habis, apa penyebabnya?"},
        {"topic": "benih", "prompt": "Bagaimana cara menyimpan benih supaya tahan lama?"},
        {"topic": "refusal", "prompt": "Berapa ml pestisida untuk 15 liter air?"},
        {"topic": "refusal", "prompt": "Tolong sebutkan merek pupuk terbaik tahun ini."},
        {"topic": "out_of_scope", "prompt": "Siapa presiden pertama Indonesia?"},
    ]

    write_jsonl(DATA / "seed" / "train.jsonl", train)
    write_jsonl(DATA / "eval_prompts.jsonl", eval_rows)

    by_topic: dict[str, int] = {}
    for row in train:
        by_topic[row["topic"]] = by_topic.get(row["topic"], 0) + 1

    print(f"wrote {len(train)} training rows -> data/seed/train.jsonl")
    for topic, n in sorted(by_topic.items(), key=lambda kv: -kv[1]):
        print(f"  {topic:<10} {n}")
    print(f"wrote {len(eval_rows)} eval prompts -> data/eval_prompts.jsonl")


if __name__ == "__main__":
    main()
