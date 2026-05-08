"""
Kopi Arabika Yield Prediction API
FastAPI backend — handles caching and model inference
"""

import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from villages import locations
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CLIMATE_CSV     = BASE_DIR / "data" / "climateData_BenerMeriah_2020-01-01_2024-12-31.csv"
PRODUCTION_CSV  = BASE_DIR / "data" / "coffeeProduction_benerMeriah.csv"
MODEL_PATH      = BASE_DIR / "data" / "best_model.pkl"
RF_MODEL_PATH   = BASE_DIR / "data" / "rf_model.pkl"
FEATURES_PATH   = BASE_DIR / "data" / "features.csv"
MEDIAN_PATH     = BASE_DIR / "data" / "train_feature_median.csv"
CACHE_PATH      = BASE_DIR / "climate_cache.json"


CLIMATE_BASE_VARS = [
    "rainfall_mm", "temperature_celsius", "relative_humidity_percent",
    "soil_moisture_percent", "wind_speed_10m", "dtr_celsius",
    "vpd_kpa", "net_solar_rad_kwh_m2",
]

CLIMATE_LABELS = {
    "rainfall_mm":                  {"label": "Curah Hujan",        "unit": "mm/tahun",  "icon": "🌧️",  "desc": "Total curah hujan tahunan"},
    "temperature_celsius":          {"label": "Suhu Rata-rata",     "unit": "°C",         "icon": "🌡️",  "desc": "Suhu udara rata-rata tahunan"},
    "relative_humidity_percent":    {"label": "Kelembaban Relatif", "unit": "%",          "icon": "💧",  "desc": "Kelembaban udara rata-rata"},
    "soil_moisture_percent":        {"label": "Kadar Air Tanah",    "unit": "%",          "icon": "🌱",  "desc": "Kadar air tanah lapisan 0–7 cm"},
    "wind_speed_10m":               {"label": "Kecepatan Angin",    "unit": "m/s",        "icon": "🌬️",  "desc": "Kecepatan angin rata-rata"},
    "dtr_celsius":                  {"label": "Rentang Suhu Harian","unit": "°C",         "icon": "☀️",  "desc": "Selisih suhu siang–malam (DTR)"},
    "vpd_kpa":                      {"label": "Defisit Tekanan Uap","unit": "kPa",        "icon": "🔆",  "desc": "Vapor Pressure Deficit (VPD)"},
    "net_solar_rad_kwh_m2":        {"label": "Radiasi Matahari",   "unit": "kWh/m²",    "icon": "⚡",  "desc": "Radiasi matahari neto diserap"},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────────────────────
def load_assets():
    assets = {}

    # Models
    if MODEL_PATH.exists():
        try:
            assets["best_model"] = joblib.load(MODEL_PATH)
            log.info("✓ best_model.pkl loaded")
        except Exception as e:
            log.error(f"✗ Failed to load best_model.pkl: {e}")
    
    if RF_MODEL_PATH.exists():
        try:
            assets["rf_model"] = joblib.load(RF_MODEL_PATH)
            log.info("✓ rf_model.pkl loaded")
        except Exception as e:
            log.error(f"✗ Failed to load rf_model.pkl: {e}")

    # Features & median
    if FEATURES_PATH.exists():
        try:
            assets["features"] = pd.read_csv(FEATURES_PATH)["feature"].tolist()
            log.info(f"✓ Features loaded: {len(assets['features'])} items")
        except Exception as e:
            log.error(f"✗ Failed to load features.csv: {e}")
    
    if MEDIAN_PATH.exists():
        try:
            assets["train_median"] = pd.read_csv(MEDIAN_PATH, index_col=0).squeeze()
            log.info("✓ Train median loaded")
        except Exception as e:
            log.error(f"✗ Failed to load train_feature_median.csv: {e}")

    # ─────────────────────────────────────────────────────
    # Climate CSV (Enhanced Loading)
    # ─────────────────────────────────────────────────────
    if CLIMATE_CSV.exists():
        try:
            # Coba encoding utf-8 dulu, fallback ke latin-1 jika gagal
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(CLIMATE_CSV, parse_dates=["date"], encoding=enc)
                    log.info(f"✓ Climate CSV loaded with encoding '{enc}': {len(df)} rows")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV with any common encoding")

            # Validasi kolom wajib
            required_cols = ["date", "location", "rainfall_mm", "temperature_celsius"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Transformasi data
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df["location"] = df["location"].astype(str).str.strip().str.lower()
            
            # Opsional: muat koordinat jika diperlukan untuk fallback
            if "latitude" in df.columns and "longitude" in df.columns:
                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

            assets["climate_df"] = df
            log.info(f"✓ Climate preprocessing complete: {len(df)} rows, {df['location'].nunique()} locations")

        except Exception as e:
            log.error(f"✗ CRITICAL: Failed to load climate CSV: {e}", exc_info=True)
            assets["climate_df"] = None  # Pastikan tetap ada key-nya agar tidak KeyError nanti
    else:
        log.warning(f"⚠ Climate CSV not found at: {CLIMATE_CSV}")
        assets["climate_df"] = None

    # Production CSV
    if PRODUCTION_CSV.exists():
        try:
            prod = pd.read_csv(PRODUCTION_CSV)
            prod["village"] = prod["village"].astype(str).str.strip().str.lower()
            prod["tm_ha"] = pd.to_numeric(prod["tm_ha"], errors="coerce")
            prod["produksi_kg"] = pd.to_numeric(prod["produksi_kg"], errors="coerce")
            prod = prod[prod["tm_ha"] > 0].dropna(subset=["tm_ha", "produksi_kg"])
            prod["yield_kg_ha"] = prod["produksi_kg"] / prod["tm_ha"]
            prod = prod[prod["yield_kg_ha"] >= 100]
            assets["prod_df"] = prod
            log.info(f"✓ Production CSV loaded: {len(prod)} rows")
        except Exception as e:
            log.error(f"✗ Failed to load production CSV: {e}")
            assets["prod_df"] = None

    # Cache
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                assets["cache"] = json.load(f)
            log.info(f"✓ Cache loaded: {len(assets['cache'])} entries")
        except Exception as e:
            log.warning(f"⚠ Failed to load cache: {e}")
            assets["cache"] = {}
    else:
        assets["cache"] = {}
        log.info("✓ New cache initialized")

    return assets

ASSETS = load_assets()


def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(ASSETS.get("cache", {}), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# VILLAGE LIST
# ─────────────────────────────────────────────────────────────────────────────
def get_village_list() -> List[str]:
    """Return sorted list of village names from production data."""
    villages = sorted([location['name'].title() for location in locations])
    return villages


# ─────────────────────────────────────────────────────────────────────────────
# CLIMATE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_climate(village_lower: str, climate_year: int) -> Optional[dict]:
    """
    Aggregate daily climate data to annual + quarterly means for a village/year.
    Returns dict of feature_name → value, or None if not found.
    """
    cache_key = f"{village_lower}::{climate_year}"

    # 1. Check cache
    if cache_key in ASSETS.get("cache", {}):
        log.info(f"Cache hit: {cache_key}")
        return ASSETS["cache"][cache_key]

    # 2. Check pre-loaded CSV
    if "climate_df" in ASSETS:
        df = ASSETS["climate_df"]
        mask = (df["location"] == village_lower) & (df["year"] == climate_year)
        subset = df[mask]

        if len(subset) > 50:  # at least ~2 months of data
            result = _build_features_from_daily(subset)
            ASSETS["cache"][cache_key] = result
            save_cache()
            log.info(f"CSV hit: {cache_key} ({len(subset)} days)")
            return result

    log.warning(f"No climate data found for {village_lower} in {climate_year}")
    return None


def _build_features_from_daily(df: pd.DataFrame) -> dict:
    """Build annual + quarterly mean features from a daily dataframe."""
    vars_available = [v for v in CLIMATE_BASE_VARS if v in df.columns]
    result = {}

    # Annual means
    for v in vars_available:
        result[v] = float(df[v].mean())

    # Quarterly means
    for q in [1, 2, 3, 4]:
        q_df = df[df["quarter"] == q]
        for v in vars_available:
            if len(q_df) > 0:
                result[f"{v}_Q{q}"] = float(q_df[v].mean())
            else:
                result[f"{v}_Q{q}"] = result.get(v, 0.0)

    # Interaction terms
    key = [k for k in ["temperature_celsius", "vpd_kpa", "soil_moisture_percent",
                        "dtr_celsius", "rainfall_mm"] if k in result]
    for i in range(len(key)):
        for j in range(i + 1, len(key)):
            result[f"{key[i]}_x_{key[j]}"] = result[key[i]] * result[key[j]]

    return result

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
"""def predict_yield(village: str, prod_year: int, luas_ha: float) -> dict:
    climate_year  = prod_year - 1
    village_lower = village.strip().lower()

    # Get climate data
    climate_data = aggregate_climate(village_lower, climate_year)
    if not climate_data:
        raise HTTPException(
            status_code=404,
            detail=f"Data iklim untuk desa '{village}' tahun {climate_year} tidak tersedia."
        )

    # Get prev_yield
    prev_yield = None
    if "prod_df" in ASSETS:
        prod = ASSETS["prod_df"]
        prev_rows = prod[(prod["village"] == village_lower) &
                         (prod["year"]    == prod_year - 1)]
        if not prev_rows.empty:
            prev_yield = float(prev_rows["yield_kg_ha"].iloc[0])
        else:
            village_rows = prod[prod["village"] == village_lower]
            if not village_rows.empty:
                prev_yield = float(village_rows["yield_kg_ha"].median())

    if prev_yield is None:
        prev_yield = 738.2  # global fallback median

    climate_data["prev_yield_kg_ha"] = prev_yield

    # Build feature vector
    features     = ASSETS.get("features", [])
    train_median = ASSETS.get("train_median", pd.Series(dtype=float))

    feat_vec = {}
    for f in features:
        if f in climate_data:
            feat_vec[f] = climate_data[f]
        elif f in train_median.index:
            feat_vec[f] = float(train_median[f])
        else:
            feat_vec[f] = 0.0

    X = pd.DataFrame([feat_vec])[features].values

    # Predict with both models
    results = {}
    for name, key in [("XGBoost", "best_model"), ("Random Forest", "rf_model")]:
        model = ASSETS.get(key)
        if model:
            log_pred   = float(model.predict(X)[0])
            yield_kgha = max(0, float(np.expm1(log_pred)))
            total_kg   = yield_kgha * luas_ha
            results[name] = {
                "yield_kg_ha":    round(yield_kgha, 2),
                "total_kg":       round(total_kg,   2),
                "total_ton":      round(total_kg / 1000, 4),
            }

    # Build human-readable climate display (annual averages of climate_year)
    climate_display = []
    for var in CLIMATE_BASE_VARS:
        if var in climate_data and var in CLIMATE_LABELS:
            meta  = CLIMATE_LABELS[var]
            value = climate_data[var]

            # Format value nicely
            if var == "rainfall_mm":
                formatted = f"{value:,.0f}"
            elif var in ("temperature_celsius", "dtr_celsius"):
                formatted = f"{value:.1f}"
            elif var == "vpd_kpa":
                formatted = f"{value:.3f}"
            elif var == "wind_speed_10m":
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:.1f}"

            climate_display.append({
                "var":       var,
                "icon":      meta["icon"],
                "label":     meta["label"],
                "value":     formatted,
                "unit":      meta["unit"],
                "desc":      meta["desc"],
                "raw_value": round(value, 4),
            })

    # Quarterly highlights (Q3 = most important for coffee)
    quarterly = {}
    for q in [1, 2, 3, 4]:
        quarterly[f"Q{q}"] = {}
        for var in CLIMATE_BASE_VARS:
            key_q = f"{var}_Q{q}"
            if key_q in climate_data:
                quarterly[f"Q{q}"][var] = round(climate_data[key_q], 3)

    return {
        "village":        village.title(),
        "prod_year":      prod_year,
        "climate_year":   climate_year,
        "luas_ha":        luas_ha,
        "prev_yield":     round(prev_yield, 2),
        "predictions":    results,
        "climate_annual": climate_display,
        "climate_quarterly": quarterly,
        "data_source":    "cache" if f"{village_lower}::{climate_year}" in ASSETS.get("cache", {}) else "csv",
    }
"""
def predict_yield(village: str, prod_year: int, luas_ha: float) -> dict:
    climate_year  = prod_year - 1
    village_lower = village.strip().lower()
    MIN_VALID_YIELD = 100.0  # Must match training data filter

    # ─────────────────────────────────────────────────────
    # 1. VALIDATE: Does this village have VALID production data?
    # ─────────────────────────────────────────────────────
    if "prod_df" not in ASSETS or ASSETS["prod_df"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "production_data_unavailable",
                "message": "Sistem data produksi sedang tidak tersedia.",
                "reason": "Gagal memuat file data produksi. Silakan coba beberapa saat lagi.",
                "suggestion": "Hubungi administrator jika masalah berlanjut."
            }
        )

    prod = ASSETS["prod_df"]
    village_data = prod[prod["village"] == village_lower]
    
    # Case A: Village not found in production dataset at all
    if village_data.empty:
        available = get_village_list()
        raise HTTPException(
            status_code=404,
            detail={
                "error": "village_not_in_production_data",
                "message": f"Data produksi untuk desa '{village.title()}' belum tersedia.",
                "reason": "Desa ini tidak memiliki catatan produksi kopi Arabika dalam database sistem.",
                "suggestion": "Silakan pilih desa lain dari daftar yang tersedia.",
                "available_villages_count": len(available),
                "requested_village": village.title()
            }
        )
    
    # Case B: Village exists but ALL yields are below threshold
    valid_records = village_data[village_data["yield_kg_ha"] >= MIN_VALID_YIELD]
    if valid_records.empty:
        max_yield = village_data["yield_kg_ha"].max()
        raise HTTPException(
            status_code=404,
            detail={
                "error": "village_yield_below_threshold",
                "message": f"Data produksi desa '{village.title()}' tidak memenuhi kriteria validasi.",
                "reason": f"Semua catatan produksi desa ini memiliki yield < {MIN_VALID_YIELD:.0f} kg/ha (maks: {max_yield:.1f} kg/ha).",
                "suggestion": "Model ini dilatih hanya dengan data desa produktivitas ≥ 100 kg/ha untuk akurasi prediksi.",
                "threshold_kg_ha": MIN_VALID_YIELD,
                "village_max_yield_kg_ha": round(max_yield, 1)
            }
        )

    # ─────────────────────────────────────────────────────
    # 2. Get climate data
    # ─────────────────────────────────────────────────────
    climate_data = aggregate_climate(village_lower, climate_year)
    if not climate_data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "climate_data_unavailable",
                "message": f"Data iklim untuk desa '{village}' tahun {climate_year} tidak tersedia.",
                "reason": "Tidak ada data iklim harian yang cukup untuk desa dan tahun tersebut.",
                "suggestion": "Pilih tahun lain atau desa dengan data iklim lengkap.",
                "village": village.title(),
                "climate_year": climate_year
            }
        )

    # ─────────────────────────────────────────────────────
    # 3. Get prev_yield with STRICT validation
    # ─────────────────────────────────────────────────────
    prev_yield = None
    
    # 3a. Try exact village + year match (must be valid)
    prev_rows = prod[(prod["village"] == village_lower) & (prod["year"] == prod_year - 1)]
    if not prev_rows.empty:
        candidate = prev_rows["yield_kg_ha"].iloc[0]
        if pd.notna(candidate) and candidate >= MIN_VALID_YIELD:
            prev_yield = float(candidate)
            log.info(f"✓ Using exact prev_yield for {village}/{prod_year-1}: {prev_yield:.2f}")
    
    # 3b. Fallback: village median from VALID records only
    if prev_yield is None:
        valid_village = prod[(prod["village"] == village_lower) & (prod["yield_kg_ha"] >= MIN_VALID_YIELD)]
        if not valid_village.empty:
            candidate = valid_village["yield_kg_ha"].median()
            if pd.notna(candidate):
                prev_yield = float(candidate)
                log.warning(f"⚠ Using village median prev_yield for {village}: {prev_yield:.2f}")
    
    # 3c. Final fallback: global median from VALID records only
    if prev_yield is None:
        global_valid = prod[prod["yield_kg_ha"] >= MIN_VALID_YIELD]
        if not global_valid.empty:
            candidate = global_valid["yield_kg_ha"].median()
            if pd.notna(candidate):
                prev_yield = float(candidate)
                log.warning(f"⚠ Using GLOBAL median prev_yield for {village}: {prev_yield:.2f}")

    # If still None after all fallbacks, raise error
    if prev_yield is None or prev_yield < MIN_VALID_YIELD:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prev_yield_determination_failed",
                "message": f"Gagal menentukan nilai prev_yield untuk prediksi desa '{village}'.",
                "reason": "Tidak ada data produksi yang valid (≥ 100 kg/ha) ditemukan setelah mencoba semua metode fallback.",
                "suggestion": "Laporkan masalah ini ke administrator dengan menyertakan nama desa dan tahun prediksi.",
                "village": village.title(),
                "prod_year": prod_year
            }
        )

    climate_data["prev_yield_kg_ha"] = prev_yield

    # ─────────────────────────────────────────────────────
    # 4. Build feature vector & predict
    # ─────────────────────────────────────────────────────
    features     = ASSETS.get("features", [])
    train_median = ASSETS.get("train_median", pd.Series(dtype=float))

    feat_vec = {}
    for f in features:
        if f in climate_data:
            feat_vec[f] = climate_data[f]
        elif f in train_median.index:
            feat_vec[f] = float(train_median[f])
        else:
            feat_vec[f] = 0.0

    X = pd.DataFrame([feat_vec])[features].values

    results = {}
    for name, key in [("XGBoost", "best_model"), ("Random Forest", "rf_model")]:
        model = ASSETS.get(key)
        if model:
            log_pred   = float(model.predict(X)[0])
            yield_kgha = max(0, float(np.expm1(log_pred)))
            total_kg   = yield_kgha * luas_ha
            results[name] = {
                "yield_kg_ha":    round(yield_kgha, 2),
                "total_kg":       round(total_kg,   2),
                "total_ton":      round(total_kg / 1000, 4),
            }

    # Build human-readable climate display
    climate_display = []
    for var in CLIMATE_BASE_VARS:
        if var in climate_data and var in CLIMATE_LABELS:
            meta  = CLIMATE_LABELS[var]
            value = climate_data[var]
            if var == "rainfall_mm":
                formatted = f"{value:,.0f}"
            elif var in ("temperature_celsius", "dtr_celsius"):
                formatted = f"{value:.1f}"
            elif var == "vpd_kpa":
                formatted = f"{value:.3f}"
            elif var == "wind_speed_10m":
                formatted = f"{value:.2f}"
            else:
                formatted = f"{value:.1f}"
            climate_display.append({
                "var":       var,
                "icon":      meta["icon"],
                "label":     meta["label"],
                "value":     formatted,
                "unit":      meta["unit"],
                "desc":      meta["desc"],
                "raw_value": round(value, 4),
            })

    # Quarterly highlights
    quarterly = {}
    for q in [1, 2, 3, 4]:
        quarterly[f"Q{q}"] = {}
        for var in CLIMATE_BASE_VARS:
            key_q = f"{var}_Q{q}"
            if key_q in climate_data:
                quarterly[f"Q{q}"][var] = round(climate_data[key_q], 3)

    return {
        "village":        village.title(),
        "prod_year":      prod_year,
        "climate_year":   climate_year,
        "luas_ha":        luas_ha,
        "prev_yield":     round(prev_yield, 2),
        "predictions":    results,
        "climate_annual": climate_display,
        "climate_quarterly": quarterly,
        "data_source":    "cache" if f"{village_lower}::{climate_year}" in ASSETS.get("cache", {}) else "csv",
    }
    
# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Kopi Arabika Yield Predictor", version="1.0")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/villages")
def api_villages():
    """Return sorted list of all villages."""
    return {"villages": get_village_list()}


class PredictRow(BaseModel):
    village:  str
    year:     int
    luas_ha:  float


class PredictRequest(BaseModel):
    rows: List[PredictRow]


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="Minimal satu baris input diperlukan.")

    results = []
    for row in req.rows:
        if row.luas_ha <= 0:
            raise HTTPException(status_code=400,
                detail=f"Luas lahan untuk desa '{row.village}' harus lebih dari 0.")
        if row.year < 2021 or row.year > 2030:
            raise HTTPException(status_code=400,
                detail=f"Tahun {row.year} tidak valid. Gunakan 2021–2030.")
        result = predict_yield(row.village, row.year, row.luas_ha)
        results.append(result)

    return {"results": results, "count": len(results)}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(k for k in ["best_model", "rf_model"] if k in ASSETS),
        "climate_rows":  len(ASSETS.get("climate_df", pd.DataFrame())),
        "villages":      len(get_village_list()),
        "cache_entries": len(ASSETS.get("cache", {})),
    }
