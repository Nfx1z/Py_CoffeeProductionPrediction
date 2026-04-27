# Metodologi Prediksi Produksi Kopi Arabika Menggunakan Random Forest dan XGBoost Berbasis Data Iklim ERA5-Land di Kabupaten Bener Meriah

---

## 1. Gambaran Umum Penelitian

Penelitian ini membangun model prediksi hasil panen (*yield*) kopi Arabika di tingkat desa di Kabupaten Bener Meriah, Aceh, menggunakan pendekatan *machine learning* berbasis data iklim harian. Dua algoritma yang digunakan adalah **Random Forest (RF)** dan **XGBoost**, yang masing-masing dioptimalkan menggunakan teknik pencarian hiperparameter secara otomatis (*hyperparameter tuning*). Target prediksi adalah **produktivitas kopi dalam satuan kg/ha** (kilogram per hektar) yang kemudian dapat dikonversi menjadi total produksi berdasarkan luas Tanaman Menghasilkan (TM) yang dimasukkan oleh pengguna.

---

## 2. Data yang Digunakan

### 2.1 Data Produksi Kopi

Data produksi kopi diperoleh dari Dinas Pertanian Kabupaten Bener Meriah dalam bentuk rekapitulasi tahunan per desa, mencakup periode **2021 hingga 2025**. Variabel yang digunakan dari data produksi adalah:

| Variabel | Keterangan |
|---|---|
| `village` | Nama desa |
| `year` | Tahun produksi |
| `tm_ha` | Luas Tanaman Menghasilkan (hektar) |
| `produksi_kg` | Total produksi kopi (kilogram) |
| `yield_kg_ha` | Produktivitas = produksi / luas TM (kg/ha) — *dihitung* |

Total desa yang digunakan setelah proses pembersihan data adalah **218 desa** dengan total **1.058 rekaman** setelah penghapusan data *outlier*.

### 2.2 Data Iklim ERA5-Land

Data iklim harian diperoleh dari dataset **ERA5-Land Daily Aggregated** milik *European Centre for Medium-Range Weather Forecasts* (ECMWF), yang diakses melalui platform **Google Earth Engine (GEE)**. Dataset ini memiliki resolusi spasial sekitar **11 km × 11 km** dan mencakup seluruh permukaan daratan global.

Data diunduh untuk periode **1 Januari 2020 hingga 31 Desember 2024**, menggunakan koordinat titik pusat masing-masing desa (titik lat/lon unik per desa). Variabel iklim dasar yang diambil adalah:

| Variabel | Satuan | Keterangan |
|---|---|---|
| `rainfall_mm` | mm | Curah hujan harian |
| `temperature_celsius` | °C | Suhu udara rata-rata pada ketinggian 2 m |
| `relative_humidity_percent` | % | Kelembaban relatif (diturunkan dari titik embun) |
| `soil_moisture_percent` | % | Kadar air tanah lapisan 0–7 cm |
| `wind_speed_10m` | m/s | Kecepatan angin pada ketinggian 10 m |
| `dtr_celsius` | °C | *Diurnal Temperature Range* = Tmaks − Tmin |
| `vpd_kpa` | kPa | *Vapor Pressure Deficit* (defisit tekanan uap) |
| `net_solar_rad_kwh_m2` | kWh/m² | Radiasi matahari neto yang diserap permukaan |

---

## 3. Pra-Pemrosesan Data (*Preprocessing*)

### 3.1 Pembersihan Data Produksi

Sebelum digunakan dalam pemodelan, data produksi melalui beberapa tahap pembersihan:

1. **Konversi tipe data** — kolom `tm_ha` dan `produksi_kg` dikonversi ke format numerik menggunakan `pd.to_numeric(..., errors='coerce')` untuk menghindari error pada data yang mengandung karakter non-numerik.

2. **Penghapusan nilai nol dan negatif** — baris dengan `tm_ha ≤ 0` dihapus karena tidak dapat digunakan untuk menghitung produktivitas.

3. **Penghapusan *outlier* yield ekstrem** — desa dengan produktivitas di bawah **100 kg/ha** dihapus dari dataset. Ambang batas ini dipilih berdasarkan analisis distribusi data, di mana terdapat **37 rekaman** (3,4%) dengan yield sangat rendah (< 100 kg/ha) yang tidak mencerminkan kondisi kebun aktif, melainkan kemungkinan merupakan lahan terbengkalai, kesalahan pencatatan, atau desa yang baru membuka kebun. Setelah penghapusan *outlier*, rentang yield berada pada **210,5 – 1.166,7 kg/ha** dengan rata-rata **738,2 kg/ha**.

4. **Normalisasi nama desa** — nama desa diubah menjadi huruf kecil (*lowercase*) menggunakan `str.strip().str.lower()` untuk menghindari ketidakcocokan penggabungan data akibat perbedaan kapitalisasi (misalnya *"Wihni Durin"* vs *"wihni Durin"*).

### 3.2 Rekayasa Fitur Iklim (*Climate Feature Engineering*)

Karena data produksi bersifat tahunan sedangkan data iklim bersifat harian, diperlukan agregasi data iklim menjadi representasi tahunan. Tiga jenis fitur iklim dibangun:

**a) Rata-rata tahunan (8 fitur)**
Setiap variabel iklim dirata-rata sepanjang satu tahun kalender per desa, menghasilkan satu nilai ringkasan per variabel per desa per tahun.

**b) Rata-rata per kuartal — Q1 hingga Q4 (32 fitur)**
Setiap variabel iklim juga dirata-rata per kuartal (Q1: Jan–Mar, Q2: Apr–Jun, Q3: Jul–Sep, Q4: Okt–Des) per desa per tahun. Ini menghasilkan 4 × 8 = **32 fitur kuartalan**. Pendekatan ini penting karena kopi Arabika memiliki fase fenologi yang berbeda di setiap kuartal — misalnya fase pembungaan dan pembentukan buah yang sangat dipengaruhi kondisi iklim pada kuartal tertentu. Hasil analisis korelasi menunjukkan bahwa fitur kuartal Q3 (khususnya `soil_moisture_percent_Q3` dan `vpd_kpa_Q3`) memiliki korelasi tertinggi dengan yield.

**c) Interaksi antar variabel (10 fitur)**
Dibuat fitur perkalian (*interaction terms*) antara pasangan variabel iklim yang secara agronomis saling berinteraksi, yaitu: `temperature × vpd`, `temperature × soil_moisture`, `temperature × dtr`, `temperature × rainfall`, `vpd × soil_moisture`, `vpd × dtr`, `vpd × rainfall`, `soil_moisture × dtr`, `soil_moisture × rainfall`, dan `dtr × rainfall`. Fitur interaksi ini membantu model menangkap efek gabungan antar variabel iklim yang tidak dapat ditangkap oleh masing-masing variabel secara terpisah.

**Total fitur iklim: 8 + 32 + 10 = 50 fitur**

### 3.3 Struktur Lag Temporal

Penelitian ini menggunakan pendekatan **lag satu tahun**, di mana data iklim tahun *N* digunakan untuk memprediksi produksi kopi tahun *N+1*. Logika ini didasarkan pada siklus pertumbuhan kopi Arabika:

| Data Iklim | Memprediksi Produksi |
|---|---|
| Tahun 2020 | Tahun 2021 |
| Tahun 2021 | Tahun 2022 |
| Tahun 2022 | Tahun 2023 |
| Tahun 2023 | Tahun 2024 |
| Tahun 2024 | Tahun 2025 |

Pendekatan ini mencerminkan kenyataan biologis bahwa kondisi iklim selama satu musim (terutama fase pembungaan dan pengisian buah) berdampak pada hasil panen di musim berikutnya.

### 3.4 Fitur Yield Tahun Sebelumnya (*prev_yield_kg_ha*)

Salah satu fitur terpenting yang ditambahkan adalah **produktivitas desa pada tahun sebelumnya** (`prev_yield_kg_ha`). Fitur ini mencerminkan kondisi dasar kebun kopi di setiap desa — desa yang produktif cenderung tetap produktif dari tahun ke tahun karena dipengaruhi faktor struktural seperti umur tanaman, kesuburan tanah, dan praktik budidaya yang relatif stabil. Penambahan fitur ini terbukti meningkatkan R² model secara drastis dari sekitar **0,07 menjadi 0,87**, membuktikan bahwa produktivitas historis merupakan prediktor terkuat dalam model ini.

Untuk desa yang tidak memiliki data tahun sebelumnya (misalnya tahun pertama dalam dataset), nilai `prev_yield_kg_ha` diisi dengan **median yield desa tersebut** lintas tahun. Apabila masih terdapat nilai yang hilang setelah pengisian tersebut, digunakan median global dari seluruh dataset pelatihan.

### 3.5 Transformasi Target (*Log Transform*)

Distribusi variabel target (`yield_kg_ha`) bersifat miring kanan (*right-skewed*), yang dapat menyebabkan model lebih dominan mempelajari nilai-nilai tinggi dan mengabaikan variasi pada rentang rendah-menengah. Untuk mengatasi hal ini, diterapkan transformasi logaritma natural:

$$y_{train} = \ln(1 + \text{yield\_kg\_ha})$$

Model dilatih menggunakan target yang telah ditransformasi. Saat melakukan prediksi, hasil keluaran model dikonversi kembali ke satuan kg/ha menggunakan transformasi invers:

$$\hat{\text{yield\_kg\_ha}} = e^{\hat{y}} - 1$$

Transformasi `log1p` dipilih (bukan `log` biasa) karena aman terhadap nilai nol dan menghasilkan distribusi yang lebih mendekati normal, yang membantu algoritma berbasis pohon bekerja lebih efisien.

---

## 4. Pembagian Data (*Train-Validate Split*)

Data dibagi menjadi dua subset berdasarkan tahun produksi:

| Subset | Tahun Produksi | Iklim yang Digunakan | Jumlah Rekaman |
|---|---|---|---|
| **Pelatihan (*Training*)** | 2021 – 2024 | 2020 – 2023 | 846 |
| **Validasi (*Validation*)** | 2025 | 2024 | 220 |

Pembagian dilakukan secara temporal (berdasarkan waktu), bukan secara acak, untuk mensimulasikan skenario prediksi nyata: model dilatih pada data masa lalu dan diuji pada data yang belum pernah dilihat sebelumnya (tahun 2025).

---

## 5. Algoritma yang Digunakan

### 5.1 Random Forest (RF)

Random Forest adalah algoritma *ensemble* berbasis pohon keputusan (*decision tree*) yang bekerja dengan membangun sejumlah besar pohon secara paralel menggunakan teknik **bagging** (*bootstrap aggregating*). Setiap pohon dilatih pada subset data yang diambil secara acak dengan pengembalian (*bootstrap sample*), dan pada setiap pembelahan (*split*) node hanya sebagian fitur yang dipertimbangkan secara acak. Prediksi akhir merupakan rata-rata dari seluruh pohon dalam hutan.

Keunggulan RF dalam konteks penelitian ini:
- Tahan terhadap *overfitting* karena mekanisme bagging dan pemilihan fitur acak
- Mampu menangani fitur yang berkorelasi tinggi
- Memberikan estimasi *feature importance* yang dapat diinterpretasikan
- Tidak memerlukan normalisasi fitur

### 5.2 XGBoost (*Extreme Gradient Boosting*)

XGBoost adalah algoritma *ensemble* berbasis *gradient boosting* yang membangun pohon secara **sekuensial**, di mana setiap pohon baru berusaha memperbaiki kesalahan (*residual*) dari pohon sebelumnya. XGBoost menggunakan teknik regularisasi L1 (`reg_alpha`) dan L2 (`reg_lambda`) untuk mencegah *overfitting*, serta dioptimalkan untuk efisiensi komputasi tinggi.

Dalam penelitian ini, XGBoost dijalankan menggunakan akselerasi **GPU (NVIDIA RTX 3050)** melalui parameter `device='cuda'` dan `tree_method='hist'`, yang secara signifikan mempercepat proses pencarian hiperparameter (*hyperparameter tuning*).

Keunggulan XGBoost dalam konteks penelitian ini:
- Umumnya menghasilkan akurasi lebih tinggi dibandingkan RF untuk data tabular
- Mekanisme *early stopping* dan regularisasi yang kuat
- Efisien secara komputasi dengan dukungan GPU
- Fleksibel dalam penanganan nilai hilang

---

## 6. Optimasi Hiperparameter (*Hyperparameter Tuning*)

### 6.1 Metode: Randomized Search Cross-Validation

Optimasi hiperparameter dilakukan menggunakan **RandomizedSearchCV** dari library scikit-learn dengan konfigurasi:

- **Jumlah iterasi**: 100 kombinasi hiperparameter yang diuji secara acak
- **Validasi silang**: 5-fold *cross-validation* (K-Fold, data dikocok dengan `random_state=42`)
- **Metrik optimasi**: *Root Mean Square Error* (RMSE) dalam ruang logaritma
- **Parallelisasi**: `n_jobs=-1` (menggunakan semua inti CPU yang tersedia)

Penggunaan *randomized search* dipilih dibandingkan *grid search* karena lebih efisien secara komputasi untuk ruang hiperparameter yang besar, sambil tetap memberikan eksplorasi yang cukup luas.

### 6.2 Ruang Pencarian Hiperparameter Random Forest

| Hiperparameter | Rentang / Nilai yang Dicoba | Keterangan |
|---|---|---|
| `n_estimators` | 100 – 800 | Jumlah pohon |
| `max_depth` | None, 5, 8, 10, 15, 20, 30 | Kedalaman maksimum pohon |
| `min_samples_split` | 2 – 20 | Minimum sampel untuk pembelahan |
| `min_samples_leaf` | 1 – 15 | Minimum sampel pada daun |
| `max_features` | sqrt, log2, 0.3, 0.5, 0.7, 0.8 | Fraksi fitur per split |
| `bootstrap` | True, False | Penggunaan bootstrap |
| `max_leaf_nodes` | None, 50, 100, 200, 500 | Batas jumlah daun |
| `min_impurity_decrease` | 0.0 – 0.02 | Penurunan impuriti minimum |

**Hiperparameter terbaik RF:**

| Hiperparameter | Nilai Optimal |
|---|---|
| `n_estimators` | 747 |
| `max_depth` | 30 |
| `max_features` | 0.7 |
| `bootstrap` | True |
| `min_samples_leaf` | 9 |
| `min_samples_split` | 2 |
| `max_leaf_nodes` | None |
| `min_impurity_decrease` | 0.000313 |

### 6.3 Ruang Pencarian Hiperparameter XGBoost

| Hiperparameter | Rentang / Nilai yang Dicoba | Keterangan |
|---|---|---|
| `n_estimators` | 100 – 800 | Jumlah pohon |
| `learning_rate` | 0.001 – 0.3 (skala log) | Laju pembelajaran |
| `max_depth` | 2 – 10 | Kedalaman pohon |
| `min_child_weight` | 1 – 15 | Bobot minimum anak |
| `subsample` | 0.5 – 1.0 | Fraksi sampel per pohon |
| `colsample_bytree` | 0.5 – 1.0 | Fraksi fitur per pohon |
| `colsample_bylevel` | 0.5 – 1.0 | Fraksi fitur per level |
| `gamma` | 0.0 – 1.0 | Pengurangan loss minimum untuk split |
| `reg_alpha` | 0.0001 – 5.0 (skala log) | Regularisasi L1 |
| `reg_lambda` | 0.001 – 5.0 (skala log) | Regularisasi L2 |
| `max_bin` | 128, 256, 512 | Jumlah bin histogram |
| `grow_policy` | depthwise, lossguide | Strategi pertumbuhan pohon |
| `max_leaves` | 0 – 32 | Batas daun (untuk lossguide) |

**Hiperparameter terbaik XGBoost:**

| Hiperparameter | Nilai Optimal |
|---|---|
| `n_estimators` | 759 |
| `learning_rate` | 0.0524 |
| `max_depth` | 4 |
| `min_child_weight` | 5 |
| `subsample` | 0.792 |
| `colsample_bytree` | 0.886 |
| `colsample_bylevel` | 0.704 |
| `gamma` | 0.012 |
| `reg_alpha` | 0.000713 |
| `reg_lambda` | 2.293 |
| `grow_policy` | depthwise |
| `max_leaves` | 26 |

---

## 7. Metrik Evaluasi

Model dievaluasi menggunakan empat metrik kuantitatif yang umum digunakan dalam pemodelan regresi untuk prediksi hasil pertanian:

### 7.1 Root Mean Square Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}$$

RMSE mengukur rata-rata kesalahan prediksi dalam satuan yang sama dengan variabel target (kg/ha). Metrik ini memberikan bobot lebih besar pada kesalahan yang besar karena menggunakan kuadrat selisih. RMSE yang lebih rendah menunjukkan model yang lebih akurat.

### 7.2 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

MAE mengukur rata-rata nilai absolut kesalahan prediksi dalam satuan kg/ha. Berbeda dengan RMSE, MAE tidak memberikan penalti ekstra pada kesalahan besar, sehingga lebih mudah diinterpretasikan secara praktis sebagai "rata-rata selisih prediksi dari nilai sebenarnya".

### 7.3 Koefisien Determinasi (R²)

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{\sum_{i=1}^{n} (\bar{y} - y_i)^2}$$

R² mengukur proporsi variansi variabel target yang dapat dijelaskan oleh model. Nilainya berkisar antara 0 hingga 1, di mana nilai 1 berarti prediksi sempurna dan nilai 0 berarti model tidak lebih baik dari prediksi rata-rata. Nilai R² negatif menunjukkan model lebih buruk dari sekadar menggunakan nilai rata-rata.

### 7.4 Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

MAPE mengukur rata-rata persentase kesalahan prediksi relatif terhadap nilai aktual. Metrik ini mudah diinterpretasikan karena dinyatakan dalam persentase dan tidak tergantung pada skala data.

> **Catatan**: Dalam penelitian prediksi hasil pertanian, F1-score tidak digunakan karena merupakan metrik untuk masalah klasifikasi (kategori), bukan regresi (nilai kontinu). Keempat metrik di atas adalah padanan yang tepat untuk masalah regresi.

---

## 8. Hasil dan Pembahasan

### 8.1 Ringkasan Performa Model

| Model | Split | RMSE (kg/ha) | MAE (kg/ha) | R² | MAPE (%) |
|---|---|---|---|---|---|
| Random Forest | Training | 25,79 | 17,44 | 0,9427 | 2,40 |
| Random Forest | Validasi | 47,46 | 31,33 | 0,8301 | 4,67 |
| **XGBoost** | **Training** | **15,57** | **11,62** | **0,9791** | **1,59** |
| **XGBoost** | **Validasi** | **41,28** | **26,70** | **0,8715** | **4,11** |

### 8.2 Analisis Performa Random Forest

Pada data pelatihan, Random Forest menghasilkan RMSE sebesar **25,79 kg/ha** dengan R² sebesar **0,9427**, yang berarti model mampu menjelaskan 94,27% variasi produktivitas kopi pada data pelatihan. Pada data validasi (tahun 2025 yang belum digunakan dalam pelatihan), RMSE meningkat menjadi **47,46 kg/ha** dengan R² sebesar **0,8301** dan MAPE sebesar **4,67%**.

Penurunan performa dari data pelatihan ke validasi (selisih R² sebesar 0,1126) menunjukkan adanya *overfitting* yang moderat, namun tidak parah mengingat kompleksitas data pertanian.

### 8.3 Analisis Performa XGBoost

XGBoost menunjukkan performa yang lebih baik dibandingkan Random Forest pada kedua subset data. Pada data pelatihan, XGBoost menghasilkan RMSE **15,57 kg/ha** dengan R² **0,9791**. Pada data validasi, RMSE sebesar **41,28 kg/ha** dengan R² **0,8715** dan MAPE **4,11%**.

Selisih R² antara pelatihan dan validasi sebesar 0,1076 menunjukkan tingkat *overfitting* yang sedikit lebih rendah dibandingkan RF, dengan akurasi validasi yang lebih tinggi. Berdasarkan seluruh metrik evaluasi pada data validasi, **XGBoost dipilih sebagai model terbaik** dalam penelitian ini.

### 8.4 Interpretasi Nilai MAPE

Nilai MAPE sebesar **4,11%** pada data validasi XGBoost dapat diinterpretasikan sebagai berikut: secara rata-rata, prediksi model menyimpang sekitar **4,11%** dari nilai produksi aktual. Dalam konteks prediksi yield pertanian, nilai ini termasuk kategori **sangat baik** berdasarkan standar umum yang digunakan dalam literatur:

| Rentang MAPE | Interpretasi |
|---|---|
| < 5% | Sangat baik |
| 5% – 10% | Baik |
| 10% – 15% | Cukup |
| > 15% | Lemah |

### 8.5 Pengaruh Fitur *prev_yield_kg_ha*

Penambahan fitur `prev_yield_kg_ha` (produktivitas tahun sebelumnya) memberikan dampak yang sangat signifikan terhadap performa model. Perbandingan sebelum dan sesudah penambahan fitur ini adalah sebagai berikut:

| Kondisi | R² Validasi | MAPE Validasi |
|---|---|---|
| Tanpa `prev_yield_kg_ha` | 0,07 | ~11,8% |
| Dengan `prev_yield_kg_ha` | **0,87** | **4,11%** |

Peningkatan R² dari 0,07 menjadi 0,87 menunjukkan bahwa **produktivitas historis merupakan prediktor terkuat** dalam model ini. Hal ini sesuai dengan karakteristik tanaman kopi Arabika sebagai tanaman tahunan (*perennial*) yang tingkat produksinya sangat dipengaruhi oleh kondisi struktural kebun yang relatif stabil dari tahun ke tahun, seperti umur tanaman, kerapatan populasi, dan praktik budidaya petani.

### 8.6 Fitur Iklim Terpenting

Berdasarkan analisis *feature importance* dari kedua model, fitur-fitur iklim yang memiliki kontribusi terbesar (setelah `prev_yield_kg_ha`) adalah:

1. **`soil_moisture_percent_Q3`** — Kadar air tanah pada kuartal Juli–September, yang merupakan fase kritis pengisian buah kopi Arabika.
2. **`vpd_kpa_Q3`** — Defisit tekanan uap pada kuartal yang sama, yang mencerminkan tekanan kekeringan atmosfer pada fase pembentukan biji.
3. **`temperature_celsius`** — Suhu rata-rata tahunan, yang pada ketinggian Bener Meriah berkorelasi dengan ketinggian lokasi kebun.
4. **`dtr_celsius`** — *Diurnal Temperature Range* (selisih suhu siang-malam), yang merupakan faktor penentu kualitas biji kopi Arabika dataran tinggi.
5. **`vpd_kpa`** — Defisit tekanan uap rata-rata tahunan, yang mencerminkan kondisi stres air pada tanaman.

### 8.7 Uji Kelayakan Model untuk Prediksi

Berdasarkan analisis komprehensif di atas, model yang dibangun dinilai **layak** digunakan untuk keperluan prediksi produksi kopi dengan pertimbangan berikut:

| Aspek | Penilaian |
|---|---|
| Akurasi (MAPE 4,11%) | ✅ Sangat baik untuk prediksi yield pertanian |
| Generalisasi (R² validasi 0,87) | ✅ Model belajar pola nyata, bukan hafalan data |
| *Overfitting* (selisih R² 0,11) | ✅ Moderat dan dapat diterima |
| Data yang belum dilihat (2025) | ✅ Terbukti mampu memprediksi tahun baru |
| Estimasi ketidakpastian | ⚠️ Gunakan dengan toleransi ±41 kg/ha |

---

## 9. Inferensi Model (Penggunaan Model untuk Prediksi)

Setelah model dilatih dan disimpan, prediksi untuk desa baru dilakukan dengan langkah berikut:

1. **Siapkan input iklim** — Hitung rata-rata tahunan dan kuartalan dari data iklim ERA5-Land untuk tahun yang ingin diprediksi (tahun *N* untuk memprediksi produksi tahun *N+1*).

2. **Masukkan luas TM** — Pengguna memasukkan luas Tanaman Menghasilkan (ha) untuk desa yang bersangkutan.

3. **Masukkan yield tahun sebelumnya** — Produktivitas aktual desa tersebut pada tahun sebelumnya (`prev_yield_kg_ha`).

4. **Jalankan model** — Model menghasilkan prediksi dalam ruang logaritma yang kemudian dikonversi balik:

$$\hat{\text{yield\_kg\_ha}} = e^{\text{model.predict}(\mathbf{X})} - 1$$

5. **Hitung total produksi**:

$$\text{Total Produksi (kg)} = \hat{\text{yield\_kg\_ha}} \times \text{luas\_TM\_ha}$$

---

## 10. Keterbatasan Penelitian

1. **Periode data terbatas** — Data produksi hanya tersedia untuk 5 tahun (2021–2025), sehingga model belum dapat menangkap variabilitas iklim jangka panjang seperti fenomena El Niño/La Niña secara komprehensif.

2. **Faktor non-iklim tidak dimasukkan** — Variabel seperti umur tanaman, penggunaan pupuk, serangan hama/penyakit, dan praktik budidaya tidak tersedia dalam dataset, padahal faktor-faktor tersebut memiliki pengaruh signifikan terhadap yield.

3. **Resolusi spasial ERA5-Land** — Meskipun data iklim diekstrak per koordinat desa (~11 km resolusi), variasi mikroklimat dalam skala yang lebih halus (seperti perbedaan lereng, naungan pohon, dan drainase tanah) tidak dapat ditangkap.

4. **Dominasi fitur historis** — Tingginya kontribusi `prev_yield_kg_ha` (produktivitas tahun sebelumnya) terhadap akurasi model menunjukkan bahwa kontribusi murni fitur iklim masih perlu ditingkatkan melalui penambahan variabel non-iklim di masa mendatang.

---

## 11. Kesimpulan

Penelitian ini berhasil membangun model prediksi produktivitas kopi Arabika berbasis *machine learning* menggunakan kombinasi data iklim ERA5-Land dan data produksi historis di 218 desa di Kabupaten Bener Meriah. Model terbaik yang dihasilkan adalah **XGBoost** dengan nilai:

- **RMSE validasi: 41,28 kg/ha**
- **MAE validasi: 26,70 kg/ha**
- **R² validasi: 0,8715**
- **MAPE validasi: 4,11%**

Hasil ini menunjukkan bahwa model mampu memprediksi produktivitas kopi Arabika dengan tingkat kesalahan rata-rata sekitar **4%** pada data tahun 2025 yang tidak digunakan dalam pelatihan. Penggabungan fitur iklim multiskala (tahunan dan kuartalan) dengan fitur produktivitas historis (`prev_yield_kg_ha`) terbukti menjadi kombinasi yang efektif untuk meningkatkan akurasi prediksi secara signifikan.

Model yang dihasilkan dapat digunakan sebagai alat bantu perencanaan produksi kopi di tingkat kabupaten, dengan cara memasukkan data iklim tahun berjalan dan luas TM masing-masing desa untuk memperoleh estimasi produksi pada tahun berikutnya.

---
