import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ==============================
# PSI - Population Stability Index
# ==============================

def calculate_psi(expected, actual, bins=10):
    """
    Hitung PSI antara distribusi lama (expected) vs baru (actual).
    PSI < 0.1  → Tidak ada drift
    PSI 0.1-0.2 → Drift ringan
    PSI > 0.2  → Drift signifikan, retraining diperlukan
    """
    expected = np.array(expected)
    actual = np.array(actual)

    # Buat bin dari data expected
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Hindari division by zero
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def interpret_psi(psi_value):
    if psi_value < 0.1:
        return "STABIL", "green"
    elif psi_value < 0.2:
        return "DRIFT RINGAN", "orange"
    else:
        return "DRIFT SIGNIFIKAN", "red"


# ==============================
# DETEKSI DRIFT PER KOMODITAS
# ==============================

def detect_drift_all(window_days=30):
    """
    Deteksi drift untuk semua komoditas.
    Membandingkan 30 hari terakhir vs 30 hari sebelumnya.
    
    Returns:
        dict: hasil PSI per komoditas
    """
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    processed_dir = os.path.join(base_dir, "data", "processed")

    komoditas_list = {
        "beras": "harga_beras.csv",
        "telur ayam": "harga_telur_ayam.csv",
        "daging_ayam": "harga_daging_ayam.csv",
    }

    results = {}

    for nama, filename in komoditas_list.items():
        path = os.path.join(processed_dir, filename)

        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            continue

        df = pd.read_csv(path)
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df = df.sort_values("tanggal")

        if len(df) < window_days * 2:
            print(f"{nama}: data tidak cukup untuk deteksi drift")
            continue

        # Split: reference (lama) vs current (baru)
        reference = df["harga"].iloc[-(window_days * 2):-window_days].values
        current = df["harga"].iloc[-window_days:].values

        psi = calculate_psi(reference, current)
        status, color = interpret_psi(psi)

        results[nama] = {
            "psi": round(psi, 4),
            "status": status,
            "color": color,
            "reference_mean": round(float(np.mean(reference)), 2),
            "current_mean": round(float(np.mean(current)), 2),
            "perubahan_pct": round(
                (np.mean(current) - np.mean(reference)) / np.mean(reference) * 100, 2
            ),
        }

        print(f"{nama}: PSI={psi:.4f} → {status}")

    return results


# ==============================
# SAVE DRIFT REPORT
# ==============================

def save_drift_report(results):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    report_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(report_dir, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "window_days": 30,
        "results": results
    }

    path = os.path.join(report_dir, "drift_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDrift report saved: {path}")
    return path


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("Menjalankan drift detection...\n")
    results = detect_drift_all(window_days=30)
    save_drift_report(results)

    print("\n=== SUMMARY ===")
    for komoditas, info in results.items():
        print(
            f"{komoditas:15} | PSI: {info['psi']:.4f} | "
            f"Status: {info['status']:20} | "
            f"Perubahan: {info['perubahan_pct']:+.1f}%"
        )