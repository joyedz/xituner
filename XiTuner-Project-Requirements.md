# XiTuner — Dataset-Surgeon Agent untuk Fine-Tuning Gemma

**Hackathon:** All Things Agentic Hackathon (Google x Devpost)
**Track:** Taskmaster
**Deadline submission:** 31 Agustus 2026, 17:00 PT
**Target submit internal:** 30 Agustus 2026 (31 Agustus = buffer, bukan hari kerja)
**Status:** Planning — revisi framing v2

> Dokumen pendamping: `XiTuner-Tech-Stack.md`
> Dokumen perencanaan ini boleh Bahasa Indonesia. **README, deskripsi submission, dan subtitle video wajib Bahasa Inggris.**

---

## 1. Ringkasan

XiTuner adalah agent yang mengubah **tumpukan data mentah berantakan** milik seseorang yang **tidak tahu apa itu learning rate** menjadi **model bahasa kecil hasil fine-tuning** yang benar-benar berperilaku seperti yang orang itu minta — dengan cara **beriterasi pada datanya**, bukan pada hyperparameter-nya.

Arsitekturnya **model-agnostic**: base model apa pun yang didukung `peft` bisa dipakai. **Gemma adalah default dan satu-satunya jalur yang didemokan** — lihat §7 untuk alasan strategisnya.

User menyatakan tujuan dalam bahasa sehari-hari dan melempar arsip mentah (export WhatsApp, PDF panduan, catatan lapangan). XiTuner lalu, secara asinkron di background: membedah dan menyusun korpus, mengompilasi tujuan itu menjadi kontrak perilaku yang bisa diuji, melatih adapter LoRA, **menilai perilaku hasilnya**, mendiagnosis penyebab kegagalan, lalu **meresepkan perubahan pada dataset** dan mengulang — sampai kontrak terpenuhi atau agent menyerah dengan jujur.

### Pergeseran penting dari v1

Versi awal dokumen ini menempatkan Gemini sebagai pembaca loss curve yang memutuskan `continue / early_stop / retry`. **Itu dibatalkan.** Alasannya jujur: pekerjaan itu sudah diselesaikan lebih baik oleh kode deterministik (`EarlyStoppingCallback`, Optuna), dan judge yang pernah fine-tune akan langsung menembak pertanyaan *"kenapa butuh LLM di sini?"* — pertanyaan yang tidak punya jawaban bagus.

Di v2, Gemini dipindahkan ke tiga pekerjaan yang **tidak punya padanan deterministik**: bedah korpus, penilaian perilaku, dan diagnosis-ke-resep-data.

---

## 2. DECISION POINT — persona & domain (harus dikunci hari ini)

Seluruh dokumen ini persona-agnostic **kecuali bagian ini**. Ganti isinya, sisanya tetap valid.

### Default yang direkomendasikan: penyuluh pertanian lapangan

| Aspek | Isi |
|---|---|
| Siapa | Penyuluh pertanian / petugas lapangan di daerah dengan koneksi internet tidak dapat diandalkan |
| Kenapa butuh model kecil | Harus jalan **offline** di laptop murah atau HP. Cloud LLM per-token tidak viable, dan sinyal sering tidak ada. **Ini menjawab pertanyaan "kenapa tidak pakai Gemini saja?" secara prinsipiil, bukan sebagai alasan demo.** |
| Data mentah | Export WhatsApp grup tanya-jawab petani (`.txt`) + PDF panduan publik (Kementan/BPTP) + catatan lapangan tidak terstruktur |
| Tujuan yang dinyatakan user | *"Aku mau modelnya jawab pertanyaan petani soal hama dan pupuk, pakai Bahasa Indonesia yang sederhana, dan bilang 'saya tidak tahu' kalau memang tidak tahu — jangan mengarang dosis."* |

**Kenapa persona ini kuat untuk judging:**

- **BYOF & "unlikely hero"** — bukan peran korporat standar, bukan ML engineer. Kriteria 40% menyebut ini eksplisit.
- **Data mentah yang benar-benar berantakan** — export WhatsApp itu campur aduk: sticker, "ok pak", typo, pertanyaan beruntun tanpa jawaban, PII nomor telepon. Kriteria menghargai *"unusual, messy, highly complex unstructured data streams"*.
- **Before/after yang dramatis dan langsung terbaca judge** — Gemma 270M mentah payah di Bahasa Indonesia domain-spesifik. Model hasil tuning yang menjawab benar dan menolak mengarang adalah kontras yang terbaca dalam 10 detik tanpa perlu paham ML.

### Alternatif (kalau data di atas tidak bisa kamu akses)

1. **Petugas kesehatan / kader posyandu** — sumber publik Kemenkes tersedia, kebutuhan offline sama kuat
2. **Pemilik UMKM dengan arsip chat CS** — data paling mudah kalau kamu punya aksesnya, tapi "offline" jadi kurang meyakinkan
3. **Guru daerah** — bagus untuk narasi, tapi kontrak perilakunya paling sulit dibuat terukur

### Kejujuran data (wajib)

Kalau sebagian korpus kamu sintesis sendiri karena tidak punya data asli, **tulis itu terang-terangan** di README dan deskripsi submission. Rules mewajibkan disclosure sumber data, dan judge lebih menghargai keterbukaan daripada klaim yang tidak bisa diverifikasi.

---

## 3. Friction yang diselesaikan

Orang dengan pengetahuan domain nyata tidak bisa membuat model kecil yang berguna untuk dirinya, karena jarak antara *"aku punya 3 tahun arsip WhatsApp"* dan *"model yang jawab benar"* diisi oleh pekerjaan yang seluruhnya di luar keahliannya: membersihkan data, memformat, menebak hyperparameter, membaca loss, menilai apakah hasilnya bagus, menebak apa yang salah, mengulang.

Bagian tersulit dari daftar itu **bukan** hyperparameter — itu justru bagian yang paling bisa diotomasi dengan tabel heuristik. Bagian tersulit adalah: **mengetahui bahwa masalahnya ada di datamu, dan tahu persis data apa yang harus ditambah atau dibuang.** Itu yang XiTuner ambil alih.

---

## 4. Locus of intelligence — batas yang eksplisit

Ini bagian yang paling dinilai di kriteria *Architectural Discipline* (30%). Batas ini **ditulis eksplisit di README**, karena menunjukkan keputusan engineering yang sadar, bukan LLM yang ditabur ke mana-mana.

| Pekerjaan | Ditangani oleh | Alasan |
|---|---|---|
| Semantic dedup, deteksi sampah, redaksi PII | **Gemini** | "ok pak" vs "oke pak" vs "👍" duplikat secara makna, bukan string |
| Diagnosis coverage gap vs tujuan | **Gemini** | Butuh paham tujuan bahasa alami vs isi korpus |
| Sintesis data untuk menambal gap | **Gemini** | Generatif, tidak ada padanan deterministik |
| Penilaian perilaku model hasil | **Gemini** | Mode collapse, drift bahasa, halusinasi dosis — **loss tidak menangkap ini sama sekali** |
| Diagnosis → resep perubahan data | **Gemini** | Inti nilai proyek ini |
| Early stopping | **Kode deterministik** (`EarlyStoppingCallback`) | Sudah selesai, LLM hanya menambah biaya + nondeterminisme |
| Pemilihan hyperparameter awal | **Tabel heuristik** berdasar ukuran korpus | Cukup, dan reproducible |
| Deteksi divergence / NaN loss | **Kode deterministik** | Threshold numerik, tidak perlu reasoning |

> Pernyataan yang akan muncul di README: *"XiTuner sengaja tidak memakai LLM untuk early stopping. Itu masalah yang sudah diselesaikan oleh callback tiga baris, dan menaruh LLM di sana hanya akan menambah biaya, latensi, dan nondeterminisme."*

---

## 5. Arsitektur alur

```
Intake (goal bahasa alami + arsip mentah)
   │
   ├─► [1] Spec Compiler (Gemini) ──► Behavior Contract (eval prompts + rubrik)
   │        dibuat SEBELUM training, dikunci, tidak boleh diubah agent
   │
   ├─► [2] Corpus Surgeon (Gemini) ──► Dataset v1 (+ rationale, + lineage)
   │
   ├─► [3] Trainer (deterministik, PEFT/LoRA) ──► Adapter checkpoint
   │
   ├─► [4] Behavioral Referee (Gemini) ──► Skor kontrak: base vs tuned
   │
   ├─► [5] Kontrak terpenuhi? ──── ya ──► [8] Laporan akhir
   │        │
   │        tidak
   │        ▼
   ├─► [6] Diagnostician (Gemini) ──► Data Prescription
   │        contoh: "prune 220 contoh pupuk, sintesis 40 pasang hama dari sumber §4"
   │
   ├─► [7] Terapkan resep ──► Dataset v2 ──► kembali ke [3]
   │        (guard: maks N iterasi, harus ada perbaikan skor)
   │
   └─► [9] Memory Bank ──► pelajaran lintas run, dipakai run berikutnya
```

### Kenapa Spec Compiler jalan lebih dulu

Kontrak perilaku dibuat **sebelum** model pertama dilatih, lalu dikunci. Ini mencegah agent merasionalisasi hasil setelah fakta ("ternyata yang penting bukan itu"). Rubrik yang dibuat setelah melihat hasil adalah rubrik yang tidak bernilai.

### Bentuk Data Prescription

Output Diagnostician **bukan** `{"action": "retry", "lr": 1e-4}`. Bentuknya operasi terhadap dataset:

```json
{
  "diagnosis": "Model menjawab semua pertanyaan seolah pertanyaan pupuk. Korpus berisi 340 contoh pupuk, 12 contoh hama.",
  "failure_modes": ["topic_collapse", "overconfident_dosage"],
  "operations": [
    {"op": "prune", "topic": "pupuk", "target_count": 120, "strategy": "keep_most_diverse"},
    {"op": "synthesize", "topic": "hama", "count": 40, "source_ref": "panduan_bptp.pdf#bab4"},
    {"op": "inject", "pattern": "refusal", "count": 15,
     "rationale": "Tidak ada satu pun contoh 'saya tidak tahu' di korpus, padahal itu bagian dari tujuan user"}
  ]
}
```

Operasi `inject refusal` di atas adalah contoh nyata sesuatu yang **tidak mungkin** dihasilkan oleh hyperparameter tuning: model tidak bisa belajar menolak kalau tidak ada satu pun contoh penolakan di data latihnya. Itu bug data, bukan bug hyperparameter — dan hanya bisa dilihat dengan membaca tujuan dan korpus bersama-sama.

---

## 6. Ketahanan terhadap failure

Kriteria *Architectural Discipline* menanyakan ini eksplisit: *"how does the system recover if a worker agent loops or returns a hallucination?"*

| Risiko | Mitigasi |
|---|---|
| Referee berhalusinasi skor | Structured output + **wajib mengutip verbatim potongan output model yang dinilai**. Skor yang mengutip teks tidak ada di output → ditolak, minta ulang sekali, lalu tandai `unverified` |
| Loop tak berujung | Batas keras `MAX_ITERATIONS`. Plus: kalau 2 resep berturut-turut tidak menaikkan skor kontrak → **berhenti dan laporkan gagal dengan jujur**, jangan terus membakar biaya |
| Resep merusak korpus | Validator deterministik: tolak resep yang membuat korpus < floor minimum, atau membuang > 40% dalam satu langkah, atau menghapus seluruh satu topik |
| Data sintetis mencemari eval | Held-out set dikunci dari korpus **asli** sebelum sintesis apa pun, dan tidak pernah disentuh Corpus Surgeon |
| Training job flaky / mati | Checkpoint ke GCS, job idempoten dengan run ID, resume dari checkpoint terakhir |
| API Gemini gagal / rate limit | Retry dengan exponential backoff; state run persist di Firestore sehingga proses bisa dilanjutkan, tidak mulai dari nol |
| Model collapse tidak terdeteksi | Cek deterministik sebelum Referee: rasio token unik, deteksi repetisi n-gram, deteksi bahasa. Murah, dan menangkap kasus paling parah tanpa memanggil LLM |

---

## 7. Requirement wajib hackathon

Ketiganya harus terpenuhi sekaligus:

| # | Requirement | Pemenuhan di XiTuner |
|---|---|---|
| 1 | Gemini 3.5+ via **Gemini API atau Vertex AI** (resmi, bukan proxy) | `gemini-3.5-flash` untuk semua peran agent |
| 2 | Minimal satu **Google Agent Framework** | Google ADK — lihat `XiTuner-Tech-Stack.md` §3 untuk alasan |
| 3 | Minimal satu **Google Cloud infrastructure service** | Cloud Run (orchestrator), Firestore (memory + state), Cloud Storage (dataset & checkpoint), Pub/Sub (event status) |

### Base model: agnostic di kode, Gemma di demo

Ini keputusan sadar dengan trade-off nyata, bukan pagar yang kebetulan terpasang.

| | Keputusan |
|---|---|
| **Kode** | Model-agnostic. Base model adalah parameter config, bukan konstanta. `peft`/`trl` memang tidak peduli base model-nya apa, jadi ini **gratis** — cukup jangan hardcode string `"gemma"` di mana pun |
| **Default + demo + pitch** | **Gemma.** Tidak dinegosiasi |
| **Dukungan multi-model sebagai fitur** | ❌ Tidak dibangun. Menguji Llama/Qwen/Mistral satu per satu memakan hari tanpa poin judging |

**Kenapa demo wajib Gemma — dua alasan konkret:**

1. **+0.2 bonus poin hangus kalau tidak.** Rules memberi +0.2 per model AI Google tambahan yang diintegrasikan. Fine-tuning Gemma memenuhi ini otomatis, tanpa kerja tambahan. Demo dengan Llama = 0.2 itu hilang begitu saja. Di skala skor maksimum 6, itu bukan angka yang bisa diabaikan
2. **Judge-nya Googler.** Menjadikan fine-tuning model non-Google sebagai bintang demo di hackathon Google adalah salah posisi yang tidak perlu

**Yang tetap kamu dapat dari agnostisisme:** klaim generalitas yang jujur di pitch — *"works on any small open model supported by PEFT; we demo with Gemma"* — memperkuat kesan sistem ini benar-benar berguna, bukan demo sekali pakai. Klaim itu gratis **selama kodenya memang tidak hardcode**, dan bohong kalau hardcode. Jadi jangan hardcode.

### Catatan penamaan kategori

Official Rules menyebut nama track lama (*Continuous Action Engine / Evolving Knowledge Engine / Multi-Agent Nexus*) di bagian judging, sementara halaman utama memakai *Taskmaster / Collaborative Partner / Fortified Enterprise Fleet*. **Submit sebagai Taskmaster.** Kriteria "Continuous Action Engine" adalah yang berlaku untuk kita.

Menguntungkan: framing v2 juga menyentuh kriteria *Evolving Knowledge Engine* (*"does the agent actively synthesize or mutate data, rather than just reading it?"*) — dan itu **persis** yang dilakukan Corpus Surgeon. Sebutkan ini di deskripsi submission.

---

## 8. Peta ke kriteria penilaian

| Kriteria | Bobot | Bagaimana XiTuner menjawabnya |
|---|---|---|
| Innovation & Operational Utility | 40% | Loop multi-step berjalan penuh di background tanpa intervensi. BYOF kuat: user non-teknis dengan kebutuhan offline nyata. Mutasi data aktif, bukan sekadar membaca |
| Architectural Discipline & Tech Stack | 30% | Batas LLM/deterministik yang eksplisit dan terdokumentasi. Dataset berversi dengan lineage. Tools terisolasi & scoped. Loop guard + verifikasi anti-halusinasi. State di Firestore, bukan di memori proses |
| Demo & Production Readiness | 30% | Money shot = decision log + before/after perilaku (lihat §9). CPU-first sehingga judge bisa mereproduksi tanpa billing GCP. Bukti Cloud eksplisit di video |

Skor maksimum **6** (5 kriteria utama + hingga 1 bonus).

**Bonus yang dikejar:** +0.2 blog build process · +0.2 post sosial `#AllThingsAgenticHackathon` · +0.2 Gemma (otomatis). Total realistis: **+0.6**.

---

## 9. Skrip demo 4 menit

Training itu membosankan untuk ditonton. Progress bar adalah visual terburuk yang ada. Video ini **tidak menampilkan loss curve sebagai bintang utama.**

| Waktu | Isi | Catatan |
|---|---|---|
| 0:00–0:30 | Friction. Tampilkan export WhatsApp yang berantakan apa adanya. Nyatakan tujuan dalam Bahasa Indonesia sehari-hari | Judge harus paham masalahnya sebelum lihat teknologi |
| 0:30–1:00 | Kick off agent, lalu **tutup terminalnya**. Tunjukkan agent tetap jalan di Cloud Run + event Pub/Sub masuk | Ini bukti "asynchronous background" yang tidak bisa dipalsukan |
| 1:00–2:30 | **Money shot.** Iterasi 1 gagal → verdict Referee di layar → **resep Diagnostician di layar, unedited**: *"tidak ada satu pun contoh penolakan di korpusmu"* → iterasi 2 jalan | Di sini judge melihat agent *berpikir*, bukan agent *menunggu* |
| 2:30–3:20 | Side-by-side prompt yang sama: Gemma mentah vs hasil tuning. Termasuk satu prompt di luar domain untuk membuktikan penolakan bekerja | Kontras harus terbaca tanpa paham ML |
| 3:20–4:00 | Bukti Cloud: log Vertex AI / Cloud Run, versi dataset di GCS, memory di Firestore | Wajib per rules |

**Aturan:** rekam beberapa run, pilih yang paling representatif. Live execution **tidak boleh dipotong di tengah** — potongan hanya antar-segmen.

---

## 10. Timeline (17 hari, hari ini 14 Agustus)

| Tanggal | Target | Gerbang keluar |
|---|---|---|
| **14 Agt (hari ini)** | Apply form kredit $150. Cek GPU quota. **Ukur waktu training Gemma 270M LoRA di CPU.** Kunci persona + kumpulkan korpus mentah | Tahu angka pasti: berapa menit satu run di CPU |
| 15–16 Agt | Trainer standalone jalan, tanpa agent | **Gerbang kill-risk:** terbukti ada perubahan perilaku yang terlihat mata setelah tuning. Kalau tidak, semua sisanya sia-sia |
| 17–18 Agt | Spec Compiler + Behavioral Referee | Referee bisa membedakan base vs tuned secara konsisten |
| 19–21 Agt | Corpus Surgeon + Diagnostician + loop lengkap | Satu run end-to-end lokal berhasil memperbaiki diri lewat resep data |
| 22–23 Agt | Wiring ADK, isolasi tools, deploy Cloud Run, Firestore memory | Jalan di Cloud, state persist |
| 24–25 Agt | Failure tolerance, loop guard, retry. Full end-to-end dengan **Gemini resmi** | Tidak ada provider lain di jalur kode |
| **26–27 Agt** | **FEATURE FREEZE.** Architecture diagram, README, spin-up instructions | Nol fitur baru setelah titik ini |
| 28 Agt | Rekam demo, beberapa take, edit | ⚠️ Form kredit **tutup hari ini** — harus sudah apply jauh sebelumnya |
| 29 Agt | Blog post + post sosial. Buffer | Bonus terkunci |
| **30 Agt** | **Submit** | — |
| 31 Agt | Buffer saja. **Jangan rencanakan pekerjaan di sini** | — |

---

## 11. Budget & kredit

- **Form kredit $150:** `https://forms.gle/riGhgDSHkHeMx8Ca6` — tutup 28 Agt 12:00 PT, review **72 jam kerja**, pemberian **tidak dijamin**.
  Hitungannya: apply hari ini (14 Agt) → kredit ~19–20 Agt. Apply 25 Agt → kredit ~28 Agt, **terlambat untuk membangun apa pun.** Apply hari ini.
- **GPU bukan jalur utama.** Gemma 270M + LoRA + korpus kecil dirancang jalan di CPU. GPU adalah *upgrade* untuk Gemma 2B, bukan prasyarat. Ini menghilangkan single point of failure terbesar proyek, **dan** memperkuat poin "reproducible setup instructions" — judge bisa `pip install` dan menjalankan tanpa billing GCP sama sekali.
- **Gemini API terpisah dari kredit Cloud.** Google AI Studio free tier (Flash) cukup untuk seluruh development.
- **Tidak ada multi-provider layer.** Dibatalkan dari v1 — lihat `XiTuner-Tech-Stack.md` §5 untuk alasannya.
- Cek billing dashboard setelah **setiap** training run di Cloud, jangan tunggu akhir.

---

## 12. Checklist submission

- [ ] Kategori: **Taskmaster**
- [ ] Repo publik (atau share ke `testing@devpost.com` + `cloudhackathons@google.com`)
- [ ] README **Bahasa Inggris** dengan spin-up instructions yang benar-benar diuji dari nol
- [ ] Architecture diagram
- [ ] Pernyataan batas LLM/deterministik di README (§4) — ini poin architectural discipline
- [ ] Disclosure sumber data, termasuk bagian yang disintesis
- [ ] Deskripsi teks: fitur, teknologi, sumber data, temuan & pembelajaran
- [ ] Video ≤4 menit, publik di YouTube/Vimeo, Bahasa Inggris atau subtitle Inggris, ada bukti Google Cloud, live execution tidak dipotong
- [ ] URL hosted project (Cloud Run) — opsional tapi disarankan
- [ ] (Bonus) Blog build process, publik, dengan disclosure dibuat untuk hackathon ini
- [ ] (Bonus) Post sosial dengan `#AllThingsAgenticHackathon`

---

## 13. Risiko terbuka

### Hasil gerbang kill-risk (14 Agt) — LULUS

Dijalankan lebih awal dari jadwal, memakai stand-in ungated `SmolLM2-135M-Instruct` karena Gemma gated. Korpus scaffolding 180 baris, LoRA, CPU.

| Run | Eval loss | Signature (base → tuned) | Degenerasi | Verdict |
|---|---|---|---|---|
| 40 step (2 epoch) | 2,87 → 2,27 | 0% → 8% | n-gram berulang **24x** | GAGAL |
| 160 step (8 epoch) | 2,87 → **0,30** | 0% → **96%** | max repeat **1** | **LULUS** |

**Kesimpulan: kegagalan pertama adalah undertraining, bukan batas metode.** Enam dari tujuh prompt mencapai 100% signature.

Temuan paling berharga: prompt di luar domain yang **tidak pernah ada di data latih** (*"Siapa presiden pertama Indonesia?"*) memicu bentuk penolakan yang benar — `Singkat: Saya tidak tahu pasti.` Perilaku menolak tergeneralisasi, tidak sekadar dihafal.

**Caveat yang harus dinyatakan terbuka:** kefasihan Bahasa Indonesia-nya masih buruk (*"Singkat: Sayan kurang bisa jadi hari yang perlu"*). Itu keterbatasan stand-in 135M yang English-centric, bukan keterbatasan metode — dan justru alasan pindah ke Gemma 4 E2B (5,12B, multilingual, **ungated**). Struktur terbukti bisa diajarkan; kefasihan menunggu model yang benar.

**Bug yang ditemukan di kode sendiri:** pemeriksa degenerasi awalnya menghitung n-gram pada gabungan seluruh output, sehingga footer `Catatan:` yang memang ada di setiap jawaban target terbaca sebagai loop. Diperbaiki jadi per-output. Bukti ini bukan penggeseran gawang: setelah diperbaiki, run 40-step **tetap gagal** dengan max repeat 24.

### Risiko terbuka

| Risiko | Tingkat | Rencana |
|---|---|---|
| ~~Model terlalu kecil untuk perubahan perilaku meyakinkan~~ | **TERTUTUP** | Terbukti 14 Agt: 0% → 96% signature |
| Kefasihan Bahasa Indonesia buruk di model kecil | Sedang | Pindah ke `gemma-4-E2B-it` (ungated, 5,12B, multilingual). Butuh GPU |
| Referee tidak konsisten antar run | Sedang | Suhu rendah + rubrik terstruktur + wajib kutipan verbatim. Uji stabilitas: run 3x pada checkpoint sama, ukur variansi |
| Korpus asli tidak tersedia | Sedang | Sumber publik (PDF Kementan/BPTP) + sintesis yang di-disclose terbuka |
| Kredit GPU ditolak | **Rendah sekarang** | CPU-first sudah menetralkan ini. Turun dari risiko fatal jadi risiko kenyamanan |
| Proyek harus baru dibuat dalam Submission Period (3–31 Agt) | — | Semua kode baru. Library standar (`transformers`, `peft`, `trl`, ADK) diizinkan; disclose kode pre-existing lain kalau ada |
| Waktu habis di polish, bukan di substansi | Sedang | Feature freeze 26 Agt bersifat keras, bukan saran |
