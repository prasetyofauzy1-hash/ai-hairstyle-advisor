# Panduan Anotasi Bentuk Wajah (FFHQ)

Panduan ini untuk anotator yang melabeli subset FFHQ (TD-14, riset non-komersial). Label dipakai sebagai
*ground truth* Fase 3, jadi **setiap anotator bekerja sendiri** dan tidak melihat label orang lain sampai selesai.
Minimal dua anotator; gambar hanya dipakai bila semua anotator memberi label yang sama.

## Persiapan

1. Jalankan `python -m ml.datasets.labeling_tool ml/datasets/ffhq_labeling/manifest.csv --images data/raw/ffhq --out data/raw/ffhq_labeling.html`.
2. Buka `data/raw/ffhq_labeling.html` di browser, isi nama Anda, lalu label setiap gambar (tombol atau angka 1-7).
3. Klik **Ekspor CSV** dan kirim `labels_<nama>.csv`. File HTML berisi wajah: jangan diunggah ke tempat publik.

## Cara menilai

Lihat garis luar wajah dari garis rambut sampai dagu, abaikan rambut, janggut tipis, dan pipi yang tampak karena
senyum. Bandingkan empat hal: panjang wajah terhadap lebar, lebar dahi, lebar tulang pipi, dan lebar serta sudut rahang.

| Label | Ciri utama |
|---|---|
| Oval | Sedikit lebih panjang daripada lebar; dahi sedikit lebih lebar dari rahang; rahang membulat, dagu meruncing halus |
| Bulat | Panjang dan lebar hampir sama; pipi paling lebar; rahang membulat tanpa sudut tegas |
| Persegi | Panjang dan lebar hampir sama; dahi, pipi, dan rahang hampir sama lebar; sudut rahang tegas |
| Lonjong | Jelas lebih panjang daripada lebar; sisi wajah cenderung lurus; dahi, pipi, rahang mirip lebarnya |
| Hati | Dahi paling lebar; menyempit ke dagu yang runcing |
| Wajik (Diamond) | Tulang pipi paling lebar; dahi dan rahang sama-sama sempit; dagu runcing |
| Ragu | Pilih bila wajah miring, tertutup (rambut, tangan, topi), atau Anda benar-benar tidak bisa memutuskan |

Pilih **Ragu** daripada menebak. Gambar dengan label Ragu atau tanpa kesepakatan tidak dipakai.

## Setelah semua anotator selesai

`python -m ml.datasets.agreement labels_*.csv --out ml/datasets/ffhq_labeling/consensus.csv` menghitung Cohen's/Fleiss'
kappa dan menulis label konsensus. Laporkan kappa apa adanya di hasil Fase 3.
