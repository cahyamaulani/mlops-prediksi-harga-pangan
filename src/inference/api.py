import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Metrics
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total jumlah request",
    ["endpoint", "komoditas", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latensi request dalam detik",
    ["endpoint", "komoditas"]
)

PREDICTION_VALUE = Histogram(
    "api_prediction_value",
    "Nilai prediksi harga",
    ["komoditas", "target"],
    buckets=[10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000]
)

# ==============================
# SETUP
# ==============================

app = FastAPI(
    title="API Prediksi Harga Pangan Jawa Timur",
    description="Early Warning dan Prediksi Harga Pangan berbasis ML",
    version="1.0.0"
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan.mlflow"
)
MLFLOW_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "trend", "day_of_week", "month", "year",
    "is_ramadan", "is_end_of_month", "is_start_of_month"
]

ALERT_THRESHOLD = 0.05  # 5% kenaikan = warning

# Cache model supaya tidak load ulang tiap request
model_cache = {}


# ==============================
# LOAD MODEL
# ==============================

def load_model(komoditas, suffix):
    key = f"{komoditas}_{suffix}"
    if key not in model_cache:
        model_name = f"harga-pangan-{komoditas}-{suffix}"
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
        except Exception:
            # Fallback ke staging kalau production tidak ada
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        model_cache[key] = model
    return model_cache[key]


# ==============================
# HELPER
# ==============================

def get_latest_features(komoditas):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    path = os.path.join(base_dir, "data", "features", f"features_{komoditas}.csv")

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Data fitur untuk {komoditas} tidak ditemukan"
        )

    df = pd.read_csv(path)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal")
    return df


def detect_alert(harga_sekarang, prediksi_7d):
    perubahan = (prediksi_7d - harga_sekarang) / harga_sekarang
    if perubahan > ALERT_THRESHOLD:
        return True, round(perubahan * 100, 2)
    return False, round(perubahan * 100, 2)


# ==============================
# ENDPOINTS
# ==============================

@app.get("/")
def root():
    return {
        "service": "API Prediksi Harga Pangan Jawa Timur",
        "version": "1.0.0",
        "endpoints": [
            "/predict/{komoditas}",
            "/predict/all",
            "/health",
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predict/{komoditas}")
def predict(komoditas: str):
    start_time = time.time()

    if komoditas not in KOMODITAS_LIST:
        REQUEST_COUNT.labels(
            endpoint="predict",
            komoditas=komoditas,
            status="error"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail=f"Komoditas tidak valid. Pilih dari: {KOMODITAS_LIST}"
        )

    df = get_latest_features(komoditas)
    sample = df[FEATURES].tail(1)
    harga_sekarang = float(df["harga"].iloc[-1])
    tanggal_terakhir = str(df["tanggal"].iloc[-1].date())

    model_1d = load_model(komoditas, "1d")
    prediksi_besok = float(model_1d.predict(sample)[0])

    model_7d = load_model(komoditas, "7d")
    prediksi_7d = float(model_7d.predict(sample)[0])

    alert, perubahan_pct = detect_alert(harga_sekarang, prediksi_7d)

    # Log metrics ke Prometheus
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="predict", komoditas=komoditas).observe(latency)
    REQUEST_COUNT.labels(endpoint="predict", komoditas=komoditas, status="success").inc()
    PREDICTION_VALUE.labels(komoditas=komoditas, target="1d").observe(prediksi_besok)
    PREDICTION_VALUE.labels(komoditas=komoditas, target="7d").observe(prediksi_7d)

    return {
        "komoditas": komoditas,
        "tanggal_data": tanggal_terakhir,
        "harga_sekarang": harga_sekarang,
        "prediksi": {
            "besok": round(prediksi_besok, 0),
            "7_hari": round(prediksi_7d, 0),
        },
        "early_warning": {
            "alert": alert,
            "perubahan_pct": perubahan_pct,
            "status": "⚠️ WASPADA - Potensi lonjakan harga!" if alert else "✅ NORMAL",
        }
    }


@app.get("/predict/all/summary")
def predict_all():
    results = {}
    for komoditas in KOMODITAS_LIST:
        try:
            results[komoditas] = predict(komoditas)
        except Exception as e:
            results[komoditas] = {"error": str(e)}
    return results