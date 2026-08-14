# XiTuner — Tech Stack Details

Dokumen pendamping `XiTuner-Project-Requirements.md`. Fokus di sini: pilihan teknis konkret, alasan di baliknya, dan struktur repo.

> Semua nomor versi di bawah adalah **lantai minimum**, bukan pin. Ekosistem ini bergerak cepat — verifikasi nama package & model saat instalasi (lihat §10).

---

## 1. Bahasa & runtime

**Python 3.11+** — ADK, PEFT, dan `transformers` semuanya native Python. Tidak ada komponen TypeScript di jalur inti.

Frontend/dashboard: **tidak dibangun.** Requirement Cloud infra sudah dipenuhi Cloud Run + Firestore + GCS, dan dashboard tidak menambah poin judging apa pun sementara memakan 2–3 hari. Kalau butuh permukaan visual untuk video, gunakan **log terstruktur yang enak dibaca** — itu justru lebih meyakinkan sebagai *proof of action* daripada UI yang dipoles.

---

## 2. Training model target

Base model adalah **parameter config**, bukan konstanta — jangan hardcode string `"gemma"` di mana pun. Tapi **default dan satu-satunya jalur yang didemokan adalah Gemma**; alasan strategisnya ada di `XiTuner-Project-Requirements.md` §7.

```python
# training/config.py
BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-<terkecil>")  # default Gemma, bisa dioverride
```

Agnostisisme di sini gratis karena `peft`/`trl` memang tidak peduli base model-nya apa. Yang **tidak** kita lakukan: menguji dan mendukung Llama/Qwen/Mistral secara eksplisit sebagai fitur (lihat §9).

### ⚠️ Gemma adalah gated repo — prasyarat keras, bukan formalitas

**Terverifikasi 14 Agt:** `google/gemma-3-270m` mengembalikan **HTTP 401 gated repo**. Nama repo-nya benar (errornya *gated*, bukan *not found*), tapi tanpa token model tidak bisa dimuat sama sekali.

Setup sekali jalan:

1. Terima lisensi di `https://huggingface.co/google/gemma-3-270m`
2. Buat token **read** di `https://huggingface.co/settings/tokens`
3. Isi `HF_TOKEN` di `.env`

**Konsekuensi yang mudah terlewat:** judge yang mencoba mereproduksi proyekmu **juga** butuh langkah ini. Kalau tidak ditulis di README, spin-up instructions-mu gagal di baris pertama — dan itu memukul kriteria *Demo & Production Readiness* (30%). Masukkan sebagai prasyarat bernomor, bukan catatan kaki.

**Stand-in untuk validasi pipeline:** `HuggingFaceTB/SmolLM2-135M-Instruct` (ungated, kecil). Dipakai hanya untuk membuktikan kode jalan tanpa menunggu approval gate. **Bukan untuk demo** — mendemokan model non-Gemma menghanguskan bonus +0.2.

> Ini contoh konkret kenapa `BASE_MODEL` sebagai env var terbayar: gating memblokir smoke test di hari pertama, dan satu env var membuat seluruh pipeline tetap bisa divalidasi hari itu juga.

### CPU sebagai jalur primer

Ini keputusan arsitektural, bukan kompromi anggaran. Alasannya tiga:

1. Menghapus single point of failure terbesar (GPU quota bisa ditolak, dan [Vertex AI menahan job dalam antrean kalau quota tidak cukup](https://cloud.google.com/vertex-ai/docs/training/understanding-training-service) — bukan error yang jelas, jadi bisa terlihat "jalan" padahal menggantung)
2. Judge bisa mereproduksi tanpa billing GCP → poin *reproducible setup instructions*
3. Iterasi development jauh lebih cepat tanpa provisioning

| Komponen | Pilihan | Alasan |
|---|---|---|
| Base model (default) | Gemma keluarga terkecil (~270M) | Satu run selesai dalam menit di CPU. **Ukur angka pastinya 14 Agt** |
| Upgrade path | Gemma 2B + quantization 4-bit, di Vertex AI GPU | Dipakai hanya kalau 270M gagal gerbang 15–16 Agt |
| Fine-tuning | LoRA via `peft` | Standar, tanpa implementasi custom |
| Training loop | `trl.SFTTrainer` | Format instruction/chat, chat template built-in |
| Format dataset | JSONL `messages` (chat) | Cocok dengan `trl` |
| Precision | `bfloat16` di GPU · `float32` di CPU | `bitsandbytes` hanya untuk jalur 2B |
| Early stopping | `EarlyStoppingCallback` — **deterministik, bukan LLM** | Lihat batas eksplisit di Requirements §4 |

```
transformers>=4.45
peft>=0.13
trl>=0.11
accelerate
datasets
bitsandbytes    # hanya untuk jalur GPU/2B
```

---

## 3. Agent framework: ADK (keputusan final)

**Dipilih: Google ADK.** `pip install google-adk`

Kriteria judging menilai *engineering decisions*, jadi alasan ini masuk README apa adanya:

| Pertimbangan | ADK | Antigravity SDK |
|---|---|---|
| Kematangan contoh komunitas | Tebal, banyak tutorial resmi | Lebih tipis, terutama untuk orkestrasi pipeline training |
| Model tool | Fungsi Python + type hints, ADK urus function-calling | Control plane Python + harness Go via WebSocket |
| Cocok untuk 17 hari solo | **Ya** | Berisiko — permukaan API lebih besar untuk dipelajari |
| Kebutuhan sandbox terkelola | Tidak butuh; kita jalankan training sendiri | Kekuatan utamanya justru di sini, tapi tidak relevan bagi kita |

**Koreksi terhadap v1 dokumen ini:** Antigravity **bukan lagi Research Preview**. [Antigravity 2.0 dan Managed Agents API diumumkan di I/O 2026](https://cloud.google.com/blog/topics/developers-practitioners/io26-news-for-agent-developers-on-google-cloud), dan [SDK-nya tersedia dalam preview](https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-developer-highlights/). Jadi alasan menolaknya **bukan** kematangan produk — melainkan bahwa kekuatan intinya (sandbox Linux terkelola untuk agent serbaguna) tidak menyelesaikan masalah kita, sementara ADK lebih langsung untuk mendefinisikan tool berbatas tegas. Tulis alasan ini persis begitu; menolak sesuatu dengan alasan yang benar lebih bernilai daripada menolak dengan alasan basi.

### Isolasi & scoping tools

Kriteria menanyakan *"are the tools properly isolated and scoped for security?"* — jadi ini bukan detail sepele:

| Tool | Boleh | Tidak boleh |
|---|---|---|
| `corpus_surgeon` | Baca korpus mentah, tulis dataset berversi | ❌ Menyentuh held-out eval set |
| `spec_compiler` | Tulis behavior contract **satu kali** | ❌ Menulis ulang kontrak setelah training dimulai |
| `trainer` | Baca dataset versi N, tulis checkpoint | ❌ Memanggil LLM sama sekali |
| `referee` | Baca output model + kontrak | ❌ Menulis apa pun ke dataset |
| `diagnostician` | Baca verdict + statistik korpus, keluarkan resep | ❌ Menerapkan resepnya sendiri |
| `prescription_applier` | Terapkan resep tervalidasi | ❌ Menjalankan resep yang gagal validator |

Pemisahan `diagnostician` (yang mengusulkan) dari `prescription_applier` (yang mengeksekusi, setelah validasi deterministik) adalah inti dari ketahanan sistem: **tidak ada komponen LLM yang boleh memutasi dataset secara langsung.**

---

## 4. Gemini API (decision layer)

- **Model:** `gemini-3.5-flash` untuk semua peran. Flash memang pilihan yang benar di sini — [diumumkan sebagai keluarga terbaru yang menggabungkan intelligence dengan action](https://blog.google/innovation-and-ai/technology/developers-tools/google-io-2026-collection/) — dan peran kita sering dipanggil dalam loop
- **Naik ke Pro?** Hanya kalau Referee terbukti tidak konsisten di uji stabilitas (3x run pada checkpoint sama). Jangan naik "untuk aman"
- **Akses:** Google AI Studio API key (free tier) untuk development → Vertex AI kalau sudah mapping ke project Cloud. Keduanya resmi, keduanya memenuhi requirement
- **SDK:** `pip install google-genai`
- **Structured output wajib** — semua keluaran agent lewat `response_schema` + model `pydantic`. Tidak ada parsing teks bebas di mana pun
- **Suhu rendah** untuk Referee (penilaian harus stabil), lebih tinggi untuk sintesis data (butuh variasi)

---

## 5. Provider LLM: satu, bukan empat

**Multi-provider abstraction layer dari v1 dibatalkan.** `glm.py`, `kimi.py`, `proxy_gemini.py` dihapus dari rencana.

Alasan:

- Memakan 1–2 hari untuk **nol poin judging**
- Kecemasan biaya yang mendorongnya tidak berdasar — AI Studio free tier Flash cukup untuk seluruh development
- Dokumen v1 sendiri sudah mengidentifikasi risiko fatalnya: perilaku berbeda saat swap, **2 hari sebelum deadline**
- Proxy Gemini pribadi adalah risiko rules yang tidak perlu diambil sama sekali

Yang tetap dipertahankan: **satu interface bersih dengan satu implementasi.** Itu higienis, bukan premature abstraction.

```python
# llm/client.py — satu file, satu implementasi
class LLMClient(Protocol):
    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...

class GeminiClient(LLMClient):  # satu-satunya implementasi
    ...
```

`pydantic` tetap dipakai untuk seluruh kontrak data internal.

---

## 6. Cloud infrastructure

| Kebutuhan | Layanan | Wajib? |
|---|---|---|
| Orchestrator agent (long-running, async) | **Cloud Run** | ✅ Memenuhi requirement #3, dan membuktikan "async background" di video |
| Memory bank + state run | **Firestore** | ✅ Menyentuh *Memory Bank* di recommended tech |
| Dataset berversi + checkpoint | **Cloud Storage** | ✅ |
| Event status async | **Pub/Sub** | Opsional tapi **sangat berharga untuk video** — event masuk setelah terminal ditutup adalah bukti asinkronitas yang tidak bisa dipalsukan |
| Training GPU | Vertex AI Custom Training | Hanya jalur upgrade 2B |

```
google-genai
google-cloud-firestore
google-cloud-storage
google-cloud-pubsub
google-cloud-aiplatform   # hanya jalur GPU
```

### Skema Firestore

```
runs/{run_id}
  goal_text, persona, status, created_at
  behavior_contract        # dikunci setelah dibuat
  iterations/{n}
    dataset_version, decision_rationale
    referee_verdict, prescription
    contract_score, checkpoint_uri

memory/lessons/{lesson_id}
  corpus_shape             # ukuran, distribusi topik, bahasa
  prescription_applied
  score_delta              # berhasil atau tidak
  hyperparams_used
```

Koleksi `memory/lessons` yang membuat run kedua lebih pintar dari run pertama — ini yang dinilai sebagai *"persistent, secure cross-session context"*. Wajib ditunjukkan di video meski hanya 5 detik.

### Lineage dataset di GCS

```
gs://xituner-{project}/runs/{run_id}/
  raw/                     # arsip mentah apa adanya, immutable
  eval/holdout.jsonl       # dikunci dari raw SEBELUM sintesis apa pun
  datasets/v1.jsonl  + v1.lineage.json
             v2.jsonl  + v2.lineage.json
  checkpoints/v1/  v2/
```

Setiap `*.lineage.json` mencatat: dari versi mana diturunkan, resep apa yang diterapkan, contoh mana yang dibuang/ditambah, dan alasannya. Ini yang menjadikan mutasi data **auditable**, bukan sekadar terjadi.

---

## 7. Struktur repo

```
xituner/
  agent/
    orchestrator.py          # definisi ADK agent + registrasi tools
    spec_compiler.py         # tujuan bahasa alami -> behavior contract
    corpus_surgeon.py        # bedah & susun korpus (Gemini)
    referee.py               # penilaian perilaku base vs tuned (Gemini)
    diagnostician.py         # verdict -> data prescription (Gemini)
    prescription.py          # validator + applier (deterministik)
    guards.py                # loop guard, verifikasi kutipan, sanity check
  training/
    train_lora.py            # PEFT/LoRA, dipanggil sebagai tool
    hyperparams.py           # tabel heuristik per ukuran korpus
    collapse_checks.py       # deteksi repetisi/bahasa (deterministik, non-LLM)
  llm/
    client.py                # interface + implementasi Gemini
    schemas.py               # seluruh model pydantic
  memory/
    store.py                 # baca/tulis Firestore lessons
  storage/
    datasets.py              # versioning + lineage di GCS
  data/
    raw/                     # arsip mentah contoh
    eval_prompts.jsonl
  infra/
    Dockerfile               # image Cloud Run
    cloudrun.yaml
    vertex_training_job.yaml # jalur GPU
  docs/
    architecture-diagram.png
    decisions.md             # batas LLM/deterministik + kenapa ADK
  README.md                  # Bahasa Inggris, spin-up instructions
  requirements.txt
  .env.example
```

`docs/decisions.md` bukan hiasan — di situlah poin *architectural discipline* dipanen. Isinya: kenapa ADK bukan Antigravity, kenapa early stopping tidak pakai LLM, kenapa CPU jalur primer, kenapa proposer dipisah dari applier.

---

## 8. Urutan testing (mengikuti gerbang di Requirements §10)

1. **Trainer sendirian, tanpa agent.** Buktikan perubahan perilaku terlihat mata setelah tuning. **Ini gerbang kill-risk — kalau gagal, hentikan dan naik ke Gemma 2B sebelum membangun apa pun di atasnya**
2. **Referee sendirian** pada dua checkpoint yang sudah diketahui berbeda. Ukur variansi: 3x run pada checkpoint sama harus memberi skor yang konsisten
3. **Loop lengkap lokal**, korpus kecil, iterasi dibatasi 2. Yang diuji: apakah resep data benar-benar menaikkan skor kontrak
4. **Uji guard secara sengaja** — suntikkan resep rusak (buang 90% korpus), verdict berhalusinasi (kutipan palsu), dan checkpoint yang collapse. Semua harus ditolak. *Rekam ini; kalau ada sisa waktu di video, ketahanan yang terbukti lebih berkesan daripada happy path*
5. **Deploy Cloud Run**, verifikasi state persist dan run bisa dilanjutkan setelah proses dimatikan
6. **End-to-end dengan Gemini resmi** — pastikan tidak ada provider lain di jalur kode sebelum rekaman

---

## 9. Yang sengaja TIDAK dibangun

Mencatat ini mencegah scope creep di hari-hari terakhir:

- Dashboard/UI web — nol poin judging
- Multi-provider LLM layer — dibatalkan, §5
- Hyperparameter search (Optuna/Ray Tune) — di luar tesis proyek; tesisnya adalah *data* yang salah, bukan hyperparameter yang salah
- Multi-agent A2A — track kita Taskmaster, bukan Multi-Agent Nexus. Peran-peran di §3 adalah tool berbatas tegas, bukan agent otonom yang saling bernegosiasi. **Jangan menjualnya sebagai multi-agent** — judge akan menganggapnya melebih-lebihkan
- **Dukungan multi-model sebagai fitur** — kodenya memang agnostic (§2), tapi menguji & mendukung Llama/Qwen/Mistral satu per satu memakan hari tanpa poin judging, dan mengalihkan demo dari Gemma akan menghanguskan bonus +0.2. Agnostic ya, multi-model tervalidasi tidak

---

## 10. Checklist verifikasi versi (isi saat instalasi)

- [ ] Nama repo Gemma terkecil terbaru di Hugging Face (penamaan Gemma sering berubah antar generasi)
- [ ] Nama package SDK Gemini resmi (`google-genai` — konfirmasi belum di-rename)
- [ ] Versi `google-adk` stabil terbaru
- [ ] String model Gemini yang tepat (`gemini-3.5-flash` — konfirmasi di AI Studio)
- [ ] Region Vertex AI yang punya GPU quota untuk project kamu (hanya kalau jalur upgrade dipakai)
- [ ] Konfirmasi `trl.SFTTrainer` masih API yang benar untuk versi `trl` yang terpasang
