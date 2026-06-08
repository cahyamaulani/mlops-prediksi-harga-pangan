import os
import mlflow
import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

MODELS = {
    "beras": "harga-pangan-beras",
    "telur_ayam": "harga-pangan-telur_ayam",
    "daging_ayam": "harga-pangan-daging_ayam",
}

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "trend", "day_of_week", "month", "year",
    "is_ramadan", "is_end_of_month", "is_start_of_month"
]


def verify_all():
    print("Verifikasi model production...\n")

    for komoditas, model_name in MODELS.items():
        print(f"[ {komoditas} ]")

        # Load model production
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")

        # Ambil data sample terbaru
        df = pd.read_csv(
            os.path.join(BASE_DIR, "data", "features", f"features_{komoditas}.csv")
        )
        sample = df[FEATURES].tail(1)

        # Prediksi
        prediksi = model.predict(sample)

        print(f"  Model: {model_name}@production")
        print(f"  Prediksi harga besok: Rp {prediksi[0]:,.0f}")
        print(f"  Status: OK ✅\n")


if __name__ == "__main__":
    verify_all()