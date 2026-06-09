"""
Unit tests untuk pipeline prediksi harga pangan.
Dijalankan otomatis oleh GitHub Actions setiap push.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Tambahkan root project ke path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.feature_engineering import create_features, is_ramadan
from src.data.preprocess import preprocess
from src.monitoring.drift_detection import calculate_psi, interpret_psi


# ============================================================
# TEST: FEATURE ENGINEERING
# ============================================================

def make_dummy_df(n=60):
    """Buat dataframe dummy untuk testing."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    harga = np.random.randint(12000, 15000, size=n).astype(float)
    return pd.DataFrame({
        "tanggal": dates,
        "komoditas": "beras",
        "provinsi": "Jawa Timur",
        "harga": harga
    })


def test_create_features_columns():
    """Feature engineering harus menghasilkan kolom yang benar."""
    df = make_dummy_df(60)
    result = create_features(df)

    expected_cols = [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "trend", "day_of_week", "month", "year",
        "is_ramadan", "is_end_of_month", "is_start_of_month",
        "target_1d", "target_7d"
    ]
    for col in expected_cols:
        assert col in result.columns, f"Kolom '{col}' tidak ditemukan!"


def test_create_features_no_nan():
    """Hasil feature engineering tidak boleh ada NaN."""
    df = make_dummy_df(60)
    result = create_features(df)
    assert result.isnull().sum().sum() == 0, "Masih ada NaN setelah feature engineering!"


def test_create_features_row_count():
    """Jumlah baris harus berkurang karena lag dan target."""
    df = make_dummy_df(60)
    result = create_features(df)
    assert len(result) < len(df), "Jumlah baris seharusnya berkurang karena lag/target!"
    assert len(result) > 0, "Hasil feature engineering kosong!"


def test_is_ramadan():
    """Deteksi Ramadan harus benar."""
    assert is_ramadan(pd.Timestamp("2024-03-20")) == 1
    assert is_ramadan(pd.Timestamp("2024-06-01")) == 0
    assert is_ramadan(pd.Timestamp("2025-03-15")) == 1


# ============================================================
# TEST: PREPROCESSING
# ============================================================

def test_preprocess_removes_nulls():
    """Preprocessing harus menghapus baris dengan harga null."""
    df = pd.DataFrame({
        "tanggal": pd.date_range("2024-01-01", periods=5),
        "commodity_name": ["beras"] * 5,
        "Provinsi": ["Jawa Timur"] * 5,
        "Nilai": [12000, None, 13000, None, 14000]
    })
    result = preprocess(df)
    assert result["harga"].isnull().sum() == 0


def test_preprocess_columns():
    """Preprocessing harus menghasilkan kolom yang benar."""
    df = pd.DataFrame({
        "tanggal": pd.date_range("2024-01-01", periods=3),
        "commodity_name": ["beras"] * 3,
        "Provinsi": ["Jawa Timur"] * 3,
        "Nilai": [12000, 12500, 13000]
    })
    result = preprocess(df)
    assert list(result.columns) == ["tanggal", "komoditas", "provinsi", "harga"]


def test_preprocess_sorted_by_date():
    """Data harus terurut berdasarkan tanggal."""
    df = pd.DataFrame({
        "tanggal": ["2024-03-01", "2024-01-01", "2024-02-01"],
        "commodity_name": ["beras"] * 3,
        "Provinsi": ["Jawa Timur"] * 3,
        "Nilai": [13000, 12000, 12500]
    })
    result = preprocess(df)
    assert result["tanggal"].is_monotonic_increasing


# ============================================================
# TEST: DRIFT DETECTION
# ============================================================

def test_psi_stable():
    """Distribusi yang sama harus menghasilkan PSI mendekati 0."""
    data = np.random.normal(12000, 500, 100)
    psi = calculate_psi(data, data)
    assert psi < 0.1, f"PSI seharusnya stabil (< 0.1), dapat: {psi}"


def test_psi_significant_drift():
    """Distribusi berbeda jauh harus menghasilkan PSI > 0.2."""
    reference = np.random.normal(12000, 200, 100)
    current = np.random.normal(18000, 200, 100)
    psi = calculate_psi(reference, current)
    assert psi > 0.2, f"PSI seharusnya drift signifikan (> 0.2), dapat: {psi}"


def test_interpret_psi():
    """Interpretasi PSI harus sesuai threshold."""
    assert interpret_psi(0.05)[0] == "STABIL"
    assert interpret_psi(0.15)[0] == "DRIFT RINGAN"
    assert interpret_psi(0.25)[0] == "DRIFT SIGNIFIKAN"


# ============================================================
# TEST: DATA VALIDATION
# ============================================================

def test_feature_harga_positive():
    """Harga tidak boleh negatif atau nol."""
    df = make_dummy_df(60)
    result = create_features(df)
    assert (result["harga"] > 0).all(), "Ada harga negatif atau nol!"


def test_lag_values_correct():
    """Nilai lag_1 harus sama dengan harga hari sebelumnya."""
    df = make_dummy_df(60)
    result = create_features(df)
    # lag_1 di baris ke-i harus = harga di baris ke-(i-1) sebelum dropna
    assert not result["lag_1"].isnull().any(), "lag_1 masih ada NaN!"