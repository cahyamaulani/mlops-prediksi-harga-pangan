import os
import dagshub
import mlflow
import pandas as pd

dagshub.init(
    repo_owner="cahyamaulani",
    repo_name="mlops-prediksi-harga-pangan",
    mlflow=True
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = {
    "1d": "Prediksi Harga Besok",
    "7d": "Deteksi Lonjakan 7 Hari",
}

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "trend", "day_of_week", "month", "year",
    "is_ramadan", "is_end_of_month", "is_start_of_month"
]

ALERT_THRESHOLD_PCT = 10.0


def verify_all():
    print("=" * 60)
    print("VERIFIKASI MODEL PRODUCTION")
    print("=" * 60)

    for komoditas in KOMODITAS_LIST:
        print(f"\n[ {komoditas.upper()} ]")

        csv_path = os.path.join(BASE_DIR, "data", "features", f"features_{komoditas}.csv")
        if not os.path.exists(csv_path):
            print(f"  ❌ File tidak ditemukan: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        sample = df[FEATURES].tail(1)
        harga_terkini = df["harga"].iloc[-1]

        for suffix, label in TARGETS.items():
            model_name = f"harga-pangan-{komoditas}-{suffix}"
            print(f"  --- {label} ---")

            try:
                model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
                prediksi = model.predict(sample)[0]

                print(f"  Model    : {model_name}@Production")
                print(f"  Harga kini: Rp {harga_terkini:,.0f}")
                print(f"  Prediksi : Rp {prediksi:,.0f}")

                if suffix == "7d":
                    selisih_pct = (prediksi - harga_terkini) / harga_terkini * 100
                    if selisih_pct >= ALERT_THRESHOLD_PCT:
                        print(f"  ⚠️  ALERT: Potensi lonjakan {selisih_pct:.1f}% dalam 7 hari!")
                    else:
                        print(f"  Perubahan : {selisih_pct:+.1f}% (aman)")

                print(f"  Status   : OK ✅")

            except Exception as e:
                print(f"  ❌ ERROR: {e}")


if __name__ == "__main__":
    verify_all()