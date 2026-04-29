# Penjelasan Sistem Prediksi Produksi Kopi Arabika Kabupaten Bener Meriah
## Alur Kerja Lengkap dari Pengumpulan Data hingga Prediksi

---

## Gambaran Umum Sistem

Sistem ini dirancang untuk memprediksi **produktivitas kopi Arabika (kg/ha)** dan **total produksi (kg)** di tingkat desa di Kabupaten Bener Meriah, Aceh, menggunakan dua sumber data utama: data iklim harian dari satelit dan data produksi kopi historis. Sistem terdiri dari dua fase besar:

- **Fase 1 — Pelatihan Model**: Membangun dan melatih model prediksi dari data historis
- **Fase 2 — Prediksi**: Menggunakan model yang sudah dilatih untuk memprediksi produksi tahun berikutnya

```
┌─────────────────────────────────────────────────────────────────┐
│                      ALUR SISTEM KESELURUHAN                    │
├──────────────┬──────────────────────────────────────────────────┤
│  INPUT       │  Data Iklim ERA5-Land + Data Produksi Kopi       │
│  PROSES      │  Pembersihan → Rekayasa Fitur → Pelatihan Model  │
│  OUTPUT      │  Prediksi yield (kg/ha) + Total produksi (kg)    │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## FASE 1 — PELATIHAN MODEL

---

### LANGKAH 1 — Pengumpulan Data Iklim dari ERA5-Land (Google Earth Engine)

**Apa yang dilakukan:**
Data iklim harian diunduh dari dataset ERA5-Land milik ECMWF melalui platform Google Earth Engine (GEE) menggunakan Python.

**Cara kerjanya:**
Untuk setiap desa di Bener Meriah, diambil koordinat titik pusat desa (latitude dan longitude). Dari setiap titik koordinat tersebut, sistem mengekstrak nilai iklim harian dari piksel ERA5-Land yang berada di lokasi tersebut. Resolusi dataset ERA5-Land adalah sekitar 11 km × 11 km, artinya setiap desa mendapatkan nilai iklim dari piksel yang paling dekat dengan koordinatnya.

**Periode data yang diunduh:** 1 Januari 2020 – 31 Desember 2024

**Variabel yang diunduh langsung dari ERA5-Land:**

| Variabel ERA5-Land | Satuan Asli | Dikonversi Menjadi | Satuan Akhir |
|---|---|---|---|
| `total_precipitation_sum` | m | `rainfall_mm` | mm |
| `temperature_2m` | Kelvin | `temperature_celsius` | °C |
| `temperature_2m_max` | Kelvin | `tmax_celsius` | °C |
| `temperature_2m_min` | Kelvin | `tmin_celsius` | °C |
| `dewpoint_temperature_2m` | Kelvin | `dewpoint_celsius` | °C |
| `surface_solar_radiation_downwards_sum` | J/m² | `solar_radiation_kwh_m2` | kWh/m² |
| `surface_net_solar_radiation_sum` | J/m² | `net_solar_rad_kwh_m2` | kWh/m² |
| `volumetric_soil_water_layer_1` | m³/m³ | `soil_moisture_percent` | % |
| `volumetric_soil_water_layer_2` | m³/m³ | `soil_moisture_layer2_pct` | % |
| `volumetric_soil_water_layer_3` | m³/m³ | `soil_moisture_layer3_pct` | % |
| `soil_temperature_level_1` | Kelvin | `soil_temp_celsius` | °C |
| `u_component_of_wind_10m` | m/s | *(untuk hitung kecepatan angin)* | — |
| `v_component_of_wind_10m` | m/s | *(untuk hitung kecepatan angin)* | — |

**Variabel yang diturunkan (dihitung dari variabel ERA5-Land):**

*Kelembaban Relatif (RH)* — dihitung dari suhu udara dan suhu titik embun:
$$\text{RH} = 100 \times e^{\left(\frac{17{,}27 \times T_d}{237{,}7 + T_d} - \frac{17{,}27 \times T}{237{,}7 + T}\right)}$$

*Diurnal Temperature Range (DTR)* — selisih suhu harian maksimum dan minimum:
$$\text{DTR} = T_{max} - T_{min}$$

*Vapor Pressure Deficit (VPD)* — defisit tekanan uap atmosfer:
$$\text{VPD} = 0{,}6108 \times e^{\frac{17{,}27 \times T}{T + 237{,}3}} - 0{,}6108 \times e^{\frac{17{,}27 \times T_d}{T_d + 237{,}3}}$$

*Kecepatan Angin* — dihitung dari dua komponen vektor:
$$v_{wind} = \sqrt{u^2 + v^2}$$

**Hasil langkah ini:** File CSV berisi data iklim harian per desa
- Ukuran: ~218 desa × 1.826 hari = ±397.000 baris
- Nama file: `climateData_BenerMeriah_2020-01-01_2024-12-31.csv`

---

### LANGKAH 2 — Pengumpulan Data Produksi Kopi

**Apa yang dilakukan:**
Data produksi kopi per desa per tahun dikumpulkan dari Dinas Pertanian Kabupaten Bener Meriah dalam format Excel (`.xlsx`), satu file per tahun.

**Format data mentah:**
Data Excel berbentuk laporan resmi yang tidak standar — memuat baris judul, baris total kecamatan, dan baris data desa yang tercampur dalam satu sheet. Setiap baris data desa memiliki label komoditas di salah satu kolomnya.

**Proses ekstraksi dari Excel:**
1. File Excel dibaca tanpa header (`header=None`)
2. Hanya baris yang kolom ke-4 (index 4) berisi `'Kopi Arabika'` yang diambil — ini secara otomatis menyaring baris judul dan total kecamatan
3. Kolom yang diekstrak: nama desa (col[3]), luas TM/ha (col[6]), produksi/kg (col[10])
4. Tahun diambil dari nama file (contoh: `2021.xlsx` → `year = 2021`)
5. Semua tahun digabung menjadi satu file CSV

**Hasil langkah ini:** File CSV terpadu berisi data produksi semua tahun
- Kolom: `year`, `village`, `tm_ha`, `produksi_kg`
- Cakupan: 2021–2025, 218 desa
- Nama file: `coffee_production_bener_meriah.csv`

---

### LANGKAH 3 — Pembersihan Data Produksi

**Apa yang dilakukan:**
Data produksi mentah dibersihkan dari nilai tidak valid sebelum digunakan dalam pemodelan.

**Sub-langkah 3a — Konversi tipe data:**
Kolom `tm_ha` dan `produksi_kg` dikonversi ke format numerik. Nilai yang tidak bisa dikonversi (teks, simbol) otomatis menjadi `NaN`. Ini dilakukan karena tipe data integer di Python akan menyebabkan *ZeroDivisionError* saat dibagi angka nol, sedangkan tipe `float` akan menghasilkan `NaN` yang aman.

**Sub-langkah 3b — Hapus baris tidak valid:**
Baris dengan `tm_ha ≤ 0` atau `tm_ha`/`produksi_kg` bernilai `NaN` dihapus karena tidak dapat digunakan untuk menghitung produktivitas.

**Sub-langkah 3c — Hitung produktivitas (yield):**
$$\text{yield\_kg\_ha} = \frac{\text{produksi\_kg}}{\text{tm\_ha}}$$

**Sub-langkah 3d — Hapus outlier yield ekstrem:**
Dilakukan analisis distribusi yield. Ditemukan **37 rekaman** dengan yield di bawah 100 kg/ha yang tidak mencerminkan kebun aktif (kemungkinan lahan terbengkalai, baru dibuka, atau kesalahan pencatatan). Rekaman ini dihapus karena akan mengganggu proses pembelajaran model.

| Kondisi | Jumlah Baris |
|---|---|
| Data awal | 1.095 |
| Setelah hapus nilai tidak valid | 1.095 |
| Setelah hapus outlier (yield < 100 kg/ha) | **1.058** |

**Sub-langkah 3e — Normalisasi nama desa:**
Semua nama desa diubah menjadi huruf kecil dan dihilangkan spasi di awal/akhir:
```
"Wihni Durin " → "wihni durin"
"ALAM JAYA"    → "alam jaya"
```
Hal yang sama dilakukan pada kolom lokasi di data iklim. Langkah ini krusial untuk memastikan penggabungan kedua dataset berjalan sempurna.

---

### LANGKAH 4 — Rekayasa Fitur Iklim (*Climate Feature Engineering*)

**Apa yang dilakukan:**
Data iklim yang masih bersifat harian diubah menjadi fitur-fitur tahunan yang dapat digunakan untuk memprediksi produksi tahunan. Tiga jenis fitur dibangun.

**Sub-langkah 4a — Rata-rata tahunan (8 fitur):**
Setiap dari 8 variabel iklim dirata-rata sepanjang satu tahun penuh per desa. Hasilnya adalah satu angka ringkasan kondisi iklim per desa per tahun.

Variabel yang digunakan:
- `rainfall_mm`, `temperature_celsius`, `relative_humidity_percent`
- `soil_moisture_percent`, `wind_speed_10m`, `dtr_celsius`
- `vpd_kpa`, `net_solar_rad_kwh_m2`

**Sub-langkah 4b — Rata-rata per kuartal (32 fitur):**
Setiap variabel iklim juga dirata-rata per kuartal, menghasilkan 4 nilai per variabel:

| Kuartal | Bulan | Relevansi untuk Kopi Arabika |
|---|---|---|
| Q1 | Jan – Mar | Musim hujan; pertumbuhan vegetatif aktif |
| Q2 | Apr – Jun | Transisi; awal inisiasi pembungaan |
| Q3 | Jul – Sep | **Fase kritis**: pembungaan dan *fruit set* |
| Q4 | Okt – Des | Pengisian dan pematangan buah |

Pendekatan kuartalan lebih informatif dibanding rata-rata tahunan karena kopi Arabika memiliki fase fenologi berbeda di setiap kuartal. Hasil analisis korelasi menunjukkan fitur Q3 (`soil_moisture_Q3`, `vpd_Q3`) memiliki korelasi tertinggi dengan yield, konsisten dengan peran krusial kondisi iklim saat fase pembungaan.

8 variabel × 4 kuartal = **32 fitur kuartalan**

**Sub-langkah 4c — Fitur interaksi (10 fitur):**
Dibuat fitur perkalian antar pasangan variabel iklim untuk menangkap efek sinergis yang tidak tertangkap secara linear:

| Interaksi | Makna Agronomi |
|---|---|
| `temperature × vpd` | Stres ganda: panas + kekeringan atmosfer |
| `temperature × soil_moisture` | Suhu tinggi + ketersediaan air tanah |
| `temperature × dtr` | Pola termal harian |
| `temperature × rainfall` | Panas dan ketersediaan air |
| `vpd × soil_moisture` | Kekeringan atmosfer vs cadangan air tanah |
| `vpd × dtr` | Variasi harian tekanan uap |
| `vpd × rainfall` | Keseimbangan air |
| `soil_moisture × dtr` | Penguapan tanah pada amplitudo suhu berbeda |
| `soil_moisture × rainfall` | Pengisian cadangan air tanah |
| `dtr × rainfall` | Efek pendinginan hujan pada rentang termal |

**Total fitur iklim: 8 + 32 + 10 = 50 fitur**

---

### LANGKAH 5 — Penambahan Fitur Produktivitas Tahun Sebelumnya

**Apa yang dilakukan:**
Ditambahkan satu fitur tambahan yang sangat penting: **produktivitas desa tersebut pada tahun sebelumnya** (`prev_yield_kg_ha`).

**Mengapa fitur ini penting:**
Kopi Arabika adalah tanaman tahunan (*perennial*) yang kondisi kebunnya relatif stabil dari tahun ke tahun. Desa yang memiliki kebun dengan manajemen baik, tanah subur, dan tanaman produktif cenderung menghasilkan yield tinggi secara konsisten. Faktor-faktor struktural ini sulit diukur langsung, namun tercermin dalam data produktivitas historis.

**Cara penghitungan:**
$$\text{prev\_yield}_{desa,\ tahun\ N+1} = \text{yield\_kg\_ha}_{desa,\ tahun\ N}$$

Contoh: `prev_yield` untuk prediksi produksi 2022 = yield aktual desa tersebut pada tahun 2021.

**Penanganan nilai hilang:**
Untuk desa yang tidak memiliki data tahun sebelumnya, digunakan median yield desa tersebut lintas seluruh tahun yang tersedia. Jika masih kosong, digunakan median global seluruh dataset.

**Dampak penambahan fitur ini:**

| Kondisi Model | R² Validasi | MAPE Validasi |
|---|---|---|
| Hanya fitur iklim (50 fitur) | 0,07 | ~11,8% |
| Iklim + prev_yield (51 fitur) | **0,87** | **4,11%** |

Peningkatan R² dari 0,07 menjadi 0,87 membuktikan bahwa produktivitas historis adalah prediktor terkuat dalam model ini.

**Total fitur akhir: 51 fitur** (50 iklim + 1 prev_yield)

---

### LANGKAH 6 — Penggabungan Data dengan Struktur Lag Temporal

**Apa yang dilakukan:**
Data iklim (tahunan per desa) digabungkan dengan data produksi menggunakan **lag satu tahun** — iklim tahun N dipasangkan dengan produksi tahun N+1.

**Mengapa menggunakan lag?**
Siklus pertumbuhan kopi Arabika membutuhkan waktu sekitar satu tahun dari pembungaan hingga panen. Kondisi iklim yang dialami tanaman selama fase pembungaan dan pengisian buah (tahun N) baru terlihat hasilnya pada saat panen (tahun N+1).

**Pasangan data yang terbentuk:**

| Iklim (Input) | Produksi (Target) | Digunakan untuk |
|---|---|---|
| 2020 | 2021 | Training |
| 2021 | 2022 | Training |
| 2022 | 2023 | Training |
| 2023 | 2024 | Training |
| 2024 | 2025 | Validasi |

**Proses penggabungan:**
Data iklim yang sudah diagregasi per tahun per desa digabungkan dengan data produksi menggunakan *inner join* berdasarkan kunci `(tahun_produksi, nama_desa)`. Dengan normalisasi nama desa yang dilakukan di Langkah 3, seluruh **218 dari 218 desa** berhasil cocok (tingkat kecocokan 100%).

**Hasil penggabungan:**
- Total rekaman: **1.058 baris**
- Setiap baris = satu desa × satu tahun produksi
- Setiap baris memiliki 51 fitur input + 1 kolom target (yield_kg_ha)

---

### LANGKAH 7 — Pembagian Data dan Transformasi Target

**Sub-langkah 7a — Pembagian data (Train/Validate Split):**
Data dibagi secara **temporal** (berdasarkan urutan waktu), bukan secara acak:

| Subset | Tahun Produksi | Jumlah Rekaman | Tujuan |
|---|---|---|---|
| **Training** | 2021 – 2024 | 846 baris | Melatih model |
| **Validasi** | 2025 | 220 baris | Menguji model pada data yang belum pernah dilihat |

Pembagian temporal dipilih karena mencerminkan skenario penggunaan nyata: model dilatih dengan data masa lalu dan harus mampu memprediksi masa depan yang belum diketahui.

**Sub-langkah 7b — Transformasi logaritma pada target:**
Distribusi yield bersifat miring kanan (*right-skewed*). Untuk menstabilkan variansi dan membantu model belajar lebih efektif, diterapkan transformasi:

$$y_{model} = \ln(1 + \text{yield\_kg\_ha}) \quad \text{(fungsi log1p)}$$

Model dilatih menggunakan nilai yang sudah ditransformasi. Saat prediksi, hasil model dibalik menggunakan:

$$\hat{\text{yield}} = e^{\hat{y}_{model}} - 1 \quad \text{(fungsi expm1)}$$

**Sub-langkah 7c — Penanganan nilai hilang pada fitur:**
Sel-sel fitur yang kosong (NaN) diisi dengan **median fitur dari data training**. Nilai median dari data training (bukan validasi) yang digunakan untuk mengisi data validasi, agar tidak terjadi kebocoran data (*data leakage*).

---

### LANGKAH 8 — Optimasi Hiperparameter dan Pelatihan Model

**Sub-langkah 8a — Metode optimasi: Randomized Search CV:**
Sebelum melatih model, dilakukan pencarian hiperparameter terbaik menggunakan **RandomizedSearchCV** dengan konfigurasi:

- **100 kombinasi** hiperparameter diuji secara acak
- **5-fold cross-validation** pada data training untuk setiap kombinasi
- Metrik optimasi: RMSE dalam ruang logaritma
- Parallelisasi penuh menggunakan semua inti CPU

Cara kerja 5-fold CV: Data training (846 baris) dibagi menjadi 5 bagian sama besar. Untuk setiap kombinasi hiperparameter, model dilatih 5 kali — setiap kali menggunakan 4 bagian untuk training dan 1 bagian untuk validasi internal. Skor akhir adalah rata-rata dari 5 percobaan tersebut.

**Sub-langkah 8b — Pelatihan Random Forest:**

Random Forest membangun sejumlah pohon keputusan secara paralel. Setiap pohon:
1. Dilatih pada subset data yang diambil secara acak dengan pengembalian (*bootstrap*)
2. Pada setiap pembelahan node, hanya subset acak dari total fitur yang dipertimbangkan
3. Pohon dibiarkan tumbuh tanpa pemangkasan (*pruning*)

Prediksi akhir adalah rata-rata dari seluruh pohon.

**Hiperparameter terbaik yang ditemukan untuk RF:**

| Hiperparameter | Nilai | Artinya |
|---|---|---|
| `n_estimators` | 747 | Membangun 747 pohon |
| `max_depth` | 30 | Pohon boleh tumbuh hingga kedalaman 30 |
| `max_features` | 0,7 | Setiap split mempertimbangkan 70% fitur |
| `bootstrap` | True | Menggunakan bootstrap sampling |
| `min_samples_leaf` | 9 | Minimal 9 sampel di setiap daun |
| `min_samples_split` | 2 | Minimal 2 sampel untuk membelah node |

**Sub-langkah 8c — Pelatihan XGBoost:**

XGBoost membangun pohon secara sekuensial. Setiap pohon baru dibuat untuk memperbaiki kesalahan dari pohon-pohon sebelumnya. Proses ini dilakukan dengan meminimalkan fungsi loss yang dilengkapi regularisasi L1 dan L2 untuk mencegah overfitting. Dalam penelitian ini, XGBoost dijalankan menggunakan GPU NVIDIA RTX 3050 (`device='cuda'`, `tree_method='hist'`) yang mempercepat proses komputasi secara signifikan.

**Hiperparameter terbaik yang ditemukan untuk XGBoost:**

| Hiperparameter | Nilai | Artinya |
|---|---|---|
| `n_estimators` | 759 | Membangun 759 pohon secara sekuensial |
| `learning_rate` | 0,0524 | Setiap pohon berkontribusi 5,24% |
| `max_depth` | 4 | Pohon dangkal untuk hindari overfitting |
| `subsample` | 0,792 | 79,2% data digunakan per pohon |
| `colsample_bytree` | 0,886 | 88,6% fitur digunakan per pohon |
| `reg_lambda` | 2,293 | Regularisasi L2 cukup kuat |
| `gamma` | 0,012 | Penurunan loss minimum untuk split |

---

### LANGKAH 9 — Evaluasi Model

**Apa yang dilakukan:**
Setelah pelatihan selesai, performa kedua model dievaluasi pada data training dan data validasi menggunakan empat metrik:

**Metrik yang digunakan:**

*RMSE (Root Mean Square Error)* — rata-rata kesalahan dalam kg/ha, dengan penalti lebih besar untuk kesalahan yang besar:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}$$

*MAE (Mean Absolute Error)* — rata-rata selisih absolut prediksi dan aktual dalam kg/ha:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

*R² (Koefisien Determinasi)* — proporsi variansi yang dijelaskan model (0–1, semakin tinggi semakin baik):
$$R^2 = 1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (\bar{y} - y_i)^2}$$

*MAPE (Mean Absolute Percentage Error)* — rata-rata persentase kesalahan relatif:
$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

**Catatan**: F1-score tidak digunakan karena merupakan metrik klasifikasi, sedangkan penelitian ini adalah masalah regresi (prediksi nilai kontinu).

**Hasil evaluasi:**

| Model | Data | RMSE (kg/ha) | MAE (kg/ha) | R² | MAPE (%) |
|---|---|---|---|---|---|
| Random Forest | Training | 25,79 | 17,44 | 0,9427 | 2,40 |
| Random Forest | **Validasi** | 47,46 | 31,33 | 0,8301 | 4,67 |
| XGBoost | Training | 15,57 | 11,62 | 0,9791 | 1,59 |
| XGBoost | **Validasi** | **41,28** | **26,70** | **0,8715** | **4,11** |

**Interpretasi hasil:**
- XGBoost dipilih sebagai model terbaik karena unggul di semua metrik pada data validasi
- MAPE 4,11% masuk kategori **sangat baik** untuk prediksi yield pertanian (standar: < 5% = sangat baik)
- R² 0,8715 berarti model menjelaskan **87% variansi** produktivitas antar desa dan tahun
- Selisih performa train-validasi (gap R² ~0,10) menunjukkan overfitting yang **moderat dan dapat diterima**

---

### LANGKAH 10 — Penyimpanan Model

**Apa yang disimpan:**

| File | Keterangan |
|---|---|
| `result/best_model.pkl` | Model terbaik (XGBoost) — digunakan untuk prediksi |
| `result/rf_model.pkl` | Model Random Forest |
| `result/xgb_model.pkl` | Model XGBoost |
| `result/features.csv` | Daftar 51 nama fitur (urutan harus sama saat prediksi) |
| `result/train_feature_median.csv` | Median fitur training (untuk mengisi nilai hilang saat prediksi) |
| `result/rf_best_params.csv` | Hiperparameter terbaik RF |
| `result/xgb_best_params.csv` | Hiperparameter terbaik XGBoost |
| `result/metrics_summary.csv` | Tabel performa semua model |
| `result/model_notes.csv` | Catatan teknis (target = log1p, balik dengan expm1) |

**Plot yang dihasilkan:**

| Plot | Keterangan |
|---|---|
| `actual_vs_predicted_train.png` | Sebaran nilai aktual vs prediksi pada data training |
| `actual_vs_predicted_validate.png` | Sebaran nilai aktual vs prediksi pada data validasi |
| `model_comparison.png` | Perbandingan RMSE, R², dan MAPE antar model |
| `feature_importance.png` | Tingkat kepentingan setiap fitur per model |
| `feature_importance_comparison.png` | Perbandingan kepentingan fitur RF vs XGBoost |
| `residuals_analysis.png` | Analisis pola kesalahan prediksi |
| `hyperparam_search_history.png` | Riwayat RMSE selama pencarian hiperparameter |
| `yield_distribution.png` | Distribusi yield sebelum dan sesudah transformasi |

---

## FASE 2 — PREDIKSI (PENGGUNAAN MODEL)

---

### LANGKAH 11 — Persiapan Data Input untuk Prediksi

**Apa yang dilakukan:**
Untuk memprediksi produksi kopi desa tertentu pada tahun N+1, diperlukan tiga input:

**Input 1 — Data iklim tahun N per desa:**
Data iklim ERA5-Land tahun N (tahun yang ingin dijadikan basis prediksi) diunduh dari GEE dengan cara yang sama seperti Langkah 1, kemudian diagregasi menjadi rata-rata tahunan dan kuartalan seperti Langkah 4. Hasilnya adalah 50 nilai fitur iklim per desa.

**Input 2 — Yield tahun sebelumnya (`prev_yield_kg_ha`):**
Produktivitas aktual desa tersebut pada tahun N (tahun yang sama dengan data iklim). Nilai ini diambil dari data produksi terakhir yang tersedia.

**Input 3 — Luas Tanaman Menghasilkan (`luas_lahan_ha`):**
Dimasukkan langsung oleh pengguna — luas TM desa yang bersangkutan dalam hektar.

---

### LANGKAH 12 — Proses Prediksi

**Apa yang dilakukan:**
Model yang sudah dilatih digunakan untuk menghasilkan prediksi yield, yang kemudian dikalikan dengan luas TM untuk mendapatkan total produksi.

**Alur prediksi:**

```
[Input: 51 fitur iklim + prev_yield]
          ↓
[Isi nilai hilang dengan median training]
          ↓
[Model XGBoost: predict() → nilai dalam log-space]
          ↓
[Invers transformasi: expm1() → yield_kg_ha]
          ↓
[Kalikan dengan luas_lahan_ha → total_produksi_kg]
          ↓
[Output: yield_kg_ha + total_produksi_kg + total_produksi_ton]
```

**Rumus:**
$$\hat{\text{yield}} = e^{\text{model.predict}(\mathbf{X})} - 1 \quad (\text{kg/ha})$$
$$\text{Total Produksi} = \hat{\text{yield}} \times \text{luas\_TM\_ha} \quad (\text{kg})$$

**Contoh output:**
```
Prediksi yield      : 762.35 kg/ha
Total produksi      : 114.352,50 kg
Total produksi (ton): 114,353 ton
```

---

## Ringkasan Alur Keseluruhan

```
DATA IKLIM ERA5-Land                    DATA PRODUKSI KOPI
(GEE, harian, per desa, 2020–2024)      (Dinas Pertanian, tahunan, 2021–2025)
         │                                         │
         ▼                                         ▼
[LANGKAH 1]                             [LANGKAH 2]
Ekstrak & konversi satuan               Ekstrak dari Excel, gabung semua tahun
         │                                         │
         │                              [LANGKAH 3]
         │                              Bersihkan data:
         │                              - Konversi numerik
         │                              - Hapus tm_ha ≤ 0
         │                              - Hitung yield = produksi/tm_ha
         │                              - Hapus outlier (yield < 100 kg/ha)
         │                              - Normalisasi nama desa (lowercase)
         │                                         │
         ▼                                         │
[LANGKAH 4]                                        │
Rekayasa fitur iklim:                              │
- Rata-rata tahunan (8 fitur)                      │
- Rata-rata kuartalan Q1–Q4 (32 fitur)             │
- Fitur interaksi (10 fitur)                       │
= 50 fitur iklim per desa per tahun                │
         │                                         │
         ▼                                         │
[LANGKAH 5]                                        │
Tambah fitur prev_yield_kg_ha ──────────────────── ┘
(produktivitas desa tahun sebelumnya)
= 51 fitur total
         │
         ▼
[LANGKAH 6]
Gabungkan data dengan lag 1 tahun:
Iklim tahun N + Produksi tahun N+1
1.058 rekaman, 218 desa
         │
         ▼
[LANGKAH 7]
Bagi data:
- Training: 846 rekaman (2021–2024)
- Validasi: 220 rekaman (2025)
Transformasi target: log1p(yield)
         │
         ├────────────────────────────┐
         ▼                            ▼
[LANGKAH 8a]                  [LANGKAH 8b]
Tuning + Training              Tuning + Training
Random Forest                  XGBoost (GPU RTX 3050)
100 iterasi, 5-fold CV         100 iterasi, 5-fold CV
         │                            │
         ▼                            ▼
[LANGKAH 9]
Evaluasi kedua model:
RMSE | MAE | R² | MAPE
→ XGBoost terbaik (MAPE 4,11%, R² 0,87)
         │
         ▼
[LANGKAH 10]
Simpan model, fitur, metadata, plot
result/best_model.pkl (XGBoost)
         │
         ▼
══════════════════════════════════════
         FASE PREDIKSI
══════════════════════════════════════
         │
[LANGKAH 11]
Input baru:
- Data iklim tahun N (50 fitur)
- Yield tahun N desa tsb. (1 fitur)
- Luas TM ha (untuk konversi output)
         │
         ▼
[LANGKAH 12]
model.predict() → expm1() → yield_kg_ha
yield_kg_ha × luas_TM → total_produksi_kg
         │
         ▼
OUTPUT:
✓ Prediksi yield (kg/ha)
✓ Total produksi (kg)
✓ Total produksi (ton)
```

---

## Catatan Teknis Penting

| Aspek | Detail |
|---|---|
| **Bahasa pemrograman** | Python 3.12 |
| **Library utama** | scikit-learn, xgboost, pandas, numpy, matplotlib, joblib |
| **Akselerasi GPU** | XGBoost menggunakan NVIDIA RTX 3050 (`device='cuda'`, `tree_method='hist'`) |
| **Reproduktibilitas** | `random_state=42` digunakan di semua proses yang melibatkan keacakan |
| **Target model** | `log1p(yield_kg_ha)` — selalu gunakan `expm1()` untuk mengkonversi balik prediksi |
| **Urutan fitur** | Harus sama persis dengan `result/features.csv` saat melakukan prediksi baru |
| **Nilai hilang** | Selalu isi dengan `result/train_feature_median.csv`, bukan median data baru |

---

*Dokumentasi ini mencakup seluruh alur sistem dari pengumpulan data hingga prediksi. Implementasi lengkap tersedia dalam file `train_coffee_model.py`.*
