"""
evaluate_and_validate.py
Membandingkan metrik model baru dengan threshold dari LK-01.
Pipeline akan GAGAL (exit code 1) jika model tidak lolos validasi.
"""

import os
import sys
import json

# ============================================================
# THRESHOLD - disesuaikan per komoditas
# Beras stabil → ketat, Telur/Daging volatile → lebih longgar
# ============================================================
MAPE_THRESHOLD = {
    "beras":      10.0,   # beras relatif stabil
    "telur_ayam": 20.0,   # telur ayam lebih volatile
    "daging_ayam": 15.0,  # daging ayam moderat
}

MAX_RMSE = {
    "beras":       800.0,
    "telur_ayam":  3000.0,
    "daging_ayam": 3000.0,
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

    # FIX: threshold per komoditas, bukan flat 10%
    mape_limit = MAPE_THRESHOLD.get(komoditas, 15.0)
    rmse_limit = MAX_RMSE.get(komoditas, float("inf"))

    passed = True
    issues = []

    if mape_pct >= mape_limit:
        passed = False
        issues.append(f"MAPE {mape_pct:.2f}% >= threshold {mape_limit}%")

    if rmse > rmse_limit:
        passed = False
        issues.append(f"RMSE {rmse:.2f} > threshold {rmse_limit}")

    return passed, mape_pct, rmse, issues


def main():
    print("=" * 60)
    print("MODEL EVALUATION & VALIDATION")
    print("=" * 60)
    for k, v in MAPE_THRESHOLD.items():
        print(f"  Threshold MAPE {k:12}: < {v}%")
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

    # FIX: pastikan folder models/ ada sebelum nulis
    os.makedirs(MODELS_DIR, exist_ok=True)
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