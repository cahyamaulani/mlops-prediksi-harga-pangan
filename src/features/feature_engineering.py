import os
import pandas as pd
import numpy as np
from datetime import datetime

# ==============================
# RAMADAN DATES (2024-2026)
# ==============================

RAMADAN_PERIODS = [
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]

# ==============================
# FEATURE ENGINEERING
# ==============================

def is_ramadan(date):
    for start, end in RAMADAN_PERIODS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return 1
    return 0


def create_features(df):
    """
    Membuat fitur time-series dari data harga harian.
    Input: df dengan kolom [tanggal, komoditas, provinsi, harga]
    Output: df dengan tambahan fitur untuk training
    """
    df = df.copy()
    df = df.sort_values("tanggal").reset_index(drop=True)

    # ==============================
    # LAG FEATURES
    # Harga di hari-hari sebelumnya
    # ==============================
    df["lag_1"] = df["harga"].shift(1)    # harga kemarin
    df["lag_7"] = df["harga"].shift(7)    # harga 7 hari lalu
    df["lag_14"] = df["harga"].shift(14)  # harga 14 hari lalu

    # ==============================
    # ROLLING FEATURES
    # Rata-rata dan volatilitas harga
    # ==============================
    df["rolling_mean_7"] = df["harga"].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df["harga"].shift(1).rolling(window=14).mean()
    df["rolling_std_7"] = df["harga"].shift(1).rolling(window=7).std()

    # ==============================
    # TREND
    # Selisih harga hari ini vs kemarin
    # ==============================
    df["trend"] = df["harga"].shift(1).diff()

    # ==============================
    # CALENDAR FEATURES
    # Pola musiman berdasarkan waktu
    # ==============================
    df["day_of_week"] = df["tanggal"].dt.dayofweek   # 0=Senin, 6=Minggu
    df["month"] = df["tanggal"].dt.month              # 1-12
    df["year"] = df["tanggal"].dt.year

    # ==============================
    # DOMAIN FEATURES
    # Pengetahuan spesifik harga pangan
    # ==============================
    df["is_ramadan"] = df["tanggal"].apply(is_ramadan)

    # Akhir bulan (biasanya harga naik karena belanja bulanan)
    df["is_end_of_month"] = (df["tanggal"].dt.day >= 25).astype(int)

    # Awal bulan
    df["is_start_of_month"] = (df["tanggal"].dt.day <= 5).astype(int)

    # ==============================
    # TARGET
    # Yang mau diprediksi: harga besok (1 hari) dan 7 hari kedepan
    # ==============================
    df["target_1d"] = df["harga"].shift(-1)   # harga besok
    df["target_7d"] = df["harga"].shift(-7)   # harga 7 hari kedepan

    # Hapus baris dengan NaN (dari lag & target)
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


# ==============================
# PROCESS PER KOMODITAS
# ==============================

def process_all_commodities():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    processed_dir = os.path.join(base_dir, "data", "processed")
    features_dir = os.path.join(base_dir, "data", "features")
    os.makedirs(features_dir, exist_ok=True)

    komoditas_files = {
        "beras": "harga_beras.csv",
        "telur_ayam": "harga_telur_ayam.csv",
        "daging_ayam": "harga_daging_ayam.csv",
    }

    summary = {}

    for nama, filename in komoditas_files.items():
        path = os.path.join(processed_dir, filename)

        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            continue

        print(f"\nProcessing: {nama}")

        df = pd.read_csv(path)
        df["tanggal"] = pd.to_datetime(df["tanggal"])

        df_features = create_features(df)

        # Simpan hasil
        out_path = os.path.join(features_dir, f"features_{nama}.csv")
        df_features.to_csv(out_path, index=False)

        summary[nama] = {
            "rows": len(df_features),
            "features": len(df_features.columns),
            "date_range": f"{df_features['tanggal'].min().date()} s/d {df_features['tanggal'].max().date()}"
        }

        print(f"Saved: {out_path}")
        print(f"Rows: {len(df_features)} | Features: {len(df_features.columns)}")
        print(f"Kolom: {list(df_features.columns)}")

    return summary


# ==============================
# MAIN
# ==============================

if __name__ == "__main__" :
    print("Memulai feature engineering...\n")
    summary = process_all_commodities()

    print("\n=== SUMMARY ===")
    for nama, info in summary.items():
        print(f"{nama:15} | {info['rows']} rows | {info['features']} features | {info['date_range']}")