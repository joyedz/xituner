# Panduan Ekstraksi Pesanan — Sistem Internal

Dokumen ini dipakai tim operasional untuk mengubah pesan pesanan pelanggan
menjadi data terstruktur yang bisa masuk sistem.

## Tugas

Baca pesan pelanggan, keluarkan satu objek JSON berisi detail pesanannya.

## Field

| Field | Isi |
|---|---|
| `produk` | Nama produk yang dipesan |
| `jumlah` | Berapa banyak |
| `ukuran` | Ukuran kemasan |
| `catatan` | Permintaan khusus dari pelanggan |

## Aturan

- Keluarkan **JSON yang valid**
- Jangan menambahkan penjelasan atau kalimat pengantar di luar JSON
- Kalau informasinya tidak ada di pesan, jangan dikarang

## Contoh

> **Pelanggan:** mau pesan 2 botol besar house blend, tolong jangan pakai gula
>
> **Output:**
> ```
> {"produk": "house blend", "jumlah": 2, "ukuran": "besar", "catatan": "tanpa gula"}
> ```

---

*Contoh dan data pada dokumen ini dibuat untuk keperluan pengembangan.*
