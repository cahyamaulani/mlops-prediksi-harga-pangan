"""
evaluate_and_validate.py
Membandingkan metrik model baru dengan threshold dari LK-01.
Pipeline akan GAGAL (exit code 1) jika model tidak lolos validasi.
"""

import os
import sys
import json

# ============================================================
# THRESHOLD dari LK-01
# ============================================================
MAPE_THRESHOLD = 10.0

MAX_RMSE = {
    "beras":      500.0,
    "telur_ayam": 2000.0,
    "daging_ayam": 2000.0,
}

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = ["1d", "7d"]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model_metrics(komoditas, suffix):
    path = os.path.join(MODELS_DIR, f"best_model_{komoditas}_{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with open(path) as f:
        return json.load(f)


def validate_model(komoditas, info):
    metrics = info["metrics"]
    mape_pct = metrics["mape_pct"]
    rmse = metrics["rmse"]

    passed = True
    issues = []

    if mape_pct >= MAPE_THRESHOLD:
        passed = False
        issues.append(f"MAPE {mape_pct:.2f}% >= threshold {MAPE_THRESHOLD}%")

    if rmse > MAX_RMSE.get(komoditas, float("inf")):
        passed = False
        issues.append(f"RMSE {rmse:.2f} > threshold {MAX_RMSE[komoditas]}")

    return passed, mape_pct, rmse, issues


def main():
    print("=" * 60)
    print("MODEL EVALUATION & VALIDATION")
    print(f"Threshold MAPE : < {MAPE_THRESHOLD}%")
    print("=" * 60)

    all_passed = True
    results = {}

    for komoditas in KOMODITAS_LIST:
        print(f"\n[ {komoditas.upper()} ]")
        results[komoditas] = {}

        for suffix in TARGETS:
            label = "Prediksi Besok" if suffix == "1d" else "Deteksi Lonjakan 7 Hari"
            print(f"  --- {label} ---")

            try:
                info = load_model_metrics(komoditas, suffix)
                passed, mape_pct, rmse, issues = validate_model(komoditas, info)

                status = "✅ LULUS" if passed else "❌ GAGAL"
                print(f"  Model  : {info['best_model']}")
                print(f"  MAPE   : {mape_pct:.4f}%")
                print(f"  RMSE   : {rmse:.2f}")
                print(f"  Status : {status}")

                if issues:
                    for issue in issues:
                        print(f"  ⚠️  {issue}")

                results[komoditas][suffix] = {
                    "passed": passed,
                    "mape_pct": mape_pct,
                    "rmse": rmse,
                    "model": info["best_model"]
                }

                if not passed:
                    all_passed = False

            except FileNotFoundError as e:
                print(f"  ❌ ERROR: {e}")
                all_passed = False

    # Simpan hasil validasi
    report_path = os.path.join(MODELS_DIR, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "all_passed": all_passed,
            "threshold_mape_pct": MAPE_THRESHOLD,
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ SEMUA MODEL LULUS VALIDASI → lanjut ke registry")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ ADA MODEL YANG GAGAL VALIDASI → pipeline dihentikan")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()