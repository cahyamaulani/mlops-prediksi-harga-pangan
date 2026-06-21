"""
Check trigger untuk Continuous Training.
Skenario A: Performance-based (MAPE > threshold)
Skenario B: Data Drift (PSI > threshold)
Skenario C: Schedule-based (setiap Minggu)
"""

import os
import json
import sys
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# FIX: threshold per komoditas (sinkron dengan evaluate_and_validate.py)
MAPE_THRESHOLD = {
    "beras":       10.0,
    "telur_ayam":  20.0,
    "daging_ayam": 15.0,
}
PSI_THRESHOLD = 0.2
KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = ["1d", "7d"]


# ==============================
# SKENARIO A: Performance-based
# ==============================

def check_performance_trigger():
    print("\n=== Skenario A: Performance Check ===")
    triggered = False

    for komoditas in KOMODITAS_LIST:
        # FIX: ambil threshold per komoditas
        mape_limit = MAPE_THRESHOLD.get(komoditas, 15.0)

        for suffix in TARGETS:
            path = os.path.join(MODELS_DIR, f"best_model_{komoditas}_{suffix}.json")
            if not os.path.exists(path):
                print(f"  {komoditas}_{suffix}: file tidak ditemukan, skip")
                continue

            with open(path) as f:
                info = json.load(f)

            mape = info["metrics"]["mape_pct"]
            print(f"  {komoditas}_{suffix}: MAPE={mape:.4f}% (limit={mape_limit}%)", end=" ")

            if mape > mape_limit:
                print(f"→ ⚠️ TRIGGER!")
                triggered = True
            else:
                print(f"→ ✅ OK")

    return triggered


# ==============================
# SKENARIO B: Data Drift
# ==============================

def check_drift_trigger():
    print("\n=== Skenario B: Data Drift Check ===")
    triggered = False

    drift_path = os.path.join(PROCESSED_DIR, "drift_report.json")
    if not os.path.exists(drift_path):
        print("  Drift report tidak ditemukan, skip")
        return False

    with open(drift_path) as f:
        report = json.load(f)

    results = report.get("results", {})
    for komoditas, info in results.items():
        psi = info["psi"]
        status = info["status"]
        print(f"  {komoditas}: PSI={psi:.4f} → {status}", end=" ")

        if psi > PSI_THRESHOLD:
            print(f"→ ⚠️ TRIGGER!")
            triggered = True
        else:
            print(f"→ ✅ OK")

    return triggered


# ==============================
# SKENARIO C: Schedule-based
# ==============================

def check_schedule_trigger():
    print("\n=== Skenario C: Schedule Check ===")

    today = datetime.now().weekday()  # 0=Senin, 6=Minggu
    is_sunday = today == 6

    if is_sunday:
        print("  Hari ini Minggu → ⚠️ TRIGGER scheduled retraining!")
    else:
        day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        print(f"  Hari ini {day_names[today]} → ✅ Belum jadwal retraining")

    return is_sunday


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("CONTINUOUS TRAINING TRIGGER CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    trigger_a = check_performance_trigger()
    trigger_b = check_drift_trigger()
    trigger_c = check_schedule_trigger()

    should_retrain = trigger_a or trigger_b or trigger_c

    print("\n" + "=" * 60)
    print("HASIL TRIGGER CHECK:")
    print(f"  Skenario A (Performance): {'⚠️ TRIGGERED' if trigger_a else '✅ OK'}")
    print(f"  Skenario B (Drift):       {'⚠️ TRIGGERED' if trigger_b else '✅ OK'}")
    print(f"  Skenario C (Schedule):    {'⚠️ TRIGGERED' if trigger_c else '✅ OK'}")
    print(f"\nKeputusan: {'🔄 RETRAINING DIPERLUKAN' if should_retrain else '✅ TIDAK PERLU RETRAIN'}")
    print("=" * 60)

    # FIX: output ke GITHUB_OUTPUT bukan hanya exit code
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"should_retrain={'true' if should_retrain else 'false'}\n")

    sys.exit(1 if should_retrain else 0)