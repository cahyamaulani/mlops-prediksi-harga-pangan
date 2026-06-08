import os
import mlflow
from mlflow.tracking import MlflowClient

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

EXPERIMENT_NAME = "Prediksi Harga Pangan"

BEST_MODELS = {
    "beras": "RandomForest_v1_beras",
    "telur_ayam": "RandomForest_v1_telur_ayam",
    "daging_ayam": "RandomForest_v1_daging_ayam",
}

# ==============================
# REGISTER MODEL
# ==============================

def register_best_models():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise Exception(f"Experiment '{EXPERIMENT_NAME}' tidak ditemukan!")

    for komoditas, run_name in BEST_MODELS.items():
        print(f"\nRegistering model untuk: {komoditas}")

        # Cari run berdasarkan nama
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1
        )

        if not runs:
            print(f"  Run '{run_name}' tidak ditemukan, skip.")
            continue

        run = runs[0]
        run_id = run.info.run_id
        mape = run.data.metrics.get("mape_pct", 0)
        rmse = run.data.metrics.get("rmse", 0)

        # Register ke Model Registry
        model_name = f"harga-pangan-{komoditas}"
        model_uri = f"runs:/{run_id}/model"

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Tambah deskripsi
        client.update_registered_model(
            name=model_name,
            description=f"Model RandomForest untuk prediksi harga {komoditas} di Jawa Timur"
        )

        # Tag versi model
        client.set_model_version_tag(
            name=model_name,
            version=registered.version,
            key="mape_pct",
            value=str(round(mape, 4))
        )
        client.set_model_version_tag(
            name=model_name,
            version=registered.version,
            key="rmse",
            value=str(round(rmse, 2))
        )

        print(f"  ✅ Registered: {model_name} v{registered.version}")
        print(f"  MAPE: {mape:.2f}% | RMSE: {rmse:.2f}")

    print("\nSemua model berhasil diregistrasi!")
    print("Buka MLflow UI untuk melihat Model Registry:")
    print("mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    register_best_models()