# 🌿 Sistem Prediksi Produksi Kopi Arabika — Bener Meriah

Website prediksi produksi kopi Arabika berbasis Machine Learning (XGBoost + Random Forest)
menggunakan data iklim ERA5-Land per desa di Kabupaten Bener Meriah, Aceh.

---

## Struktur Folder

```
coffee_app/
├── app.py                          ← Backend FastAPI
├── requirements.txt
├── .env.example                    ← Salin ke .env dan isi project ID
├── .env                            ← (buat sendiri, jangan di-commit)
├── climate_cache.json              ← Cache data iklim (auto-generated)
│
├── static/
│   └── index.html                  ← Frontend website
│
├── result/                         ← Folder model hasil training
│   ├── best_model.pkl              ← Model terbaik (XGBoost)
│   ├── rf_model.pkl                ← Random Forest
│   ├── xgb_model.pkl              ← XGBoost
│   ├── features.csv               ← Daftar 51 fitur
│   └── train_feature_median.csv   ← Median fitur untuk imputation
│
├── climateData_BenerMeriah_2020-01-01_2024-12-31.csv  ← Data iklim pre-extracted
└── coffee_production_bener_meriah.csv                  ← Data produksi kopi
```

---

## Setup & Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Konfigurasi .env
```bash
cp .env.example .env
# Edit .env dan isi GEE_PROJECT_ID
```

### 3. Pastikan file model dan data sudah ada
- `result/best_model.pkl`
- `result/rf_model.pkl`
- `result/features.csv`
- `result/train_feature_median.csv`
- `climateData_BenerMeriah_2020-01-01_2024-12-31.csv`
- `coffee_production_bener_meriah.csv`

### 4. Jalankan server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Buka browser
```
http://localhost:8000
```

---

## Cara Kerja Data Iklim (Smart Caching)

```
User request (desa X, tahun panen Y)
          ↓
Sistem butuh data iklim tahun Y-1 untuk desa X
          ↓
1. Cek climate_cache.json  → Ada? Gunakan langsung ✓
          ↓ Tidak ada
2. Cek CSV pre-extracted   → Ada? Gunakan + simpan ke cache ✓
          ↓ Tidak ada
3. Fetch dari GEE live     → Ambil, simpan ke cache, gunakan ✓
```

Data yang sudah di-cache akan digunakan secara otomatis oleh pengguna berikutnya
yang meminta desa dan tahun yang sama → tidak perlu fetch ulang.

---

## API Endpoints

| Method | Path | Deskripsi |
|---|---|---|
| GET | `/` | Halaman website |
| GET | `/api/villages` | Daftar semua desa |
| POST | `/api/predict` | Prediksi produksi |
| GET | `/api/health` | Status sistem |

### Contoh Request Prediksi
```json
POST /api/predict
{
  "rows": [
    { "village": "Alam Jaya", "year": 2025, "luas_ha": 4.80 },
    { "village": "Kenine",    "year": 2025, "luas_ha": 93.54 }
  ]
}
```

---

## Catatan Teknis

- Model dilatih dengan target `log1p(yield_kg_ha)` → prediksi dibalik dengan `expm1()`
- Akurasi validasi: MAPE 4,11%, R² 0,8715 (XGBoost)
- Data iklim: ERA5-Land ~11km resolusi, per koordinat desa
- GEE fetch membutuhkan autentikasi Application Default Credentials (ADC)
