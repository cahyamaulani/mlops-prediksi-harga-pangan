import os
import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

# ==============================
# SETUP MLFLOW → DAGSHUB
# ==============================

dagshub.init(
    repo_owner="cahyamaulani",
    repo_name="mlops-prediksi-harga-pangan",
    mlflow=True
)

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = {
    "1d": "Prediksi Harga Pangan - target_1d",
    "7d": "Prediksi Harga Pangan - target_7d",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_best_model_info(komoditas, suffix):
    path = os.path.join(MODELS_DIR, f"best_model_{komoditas}_{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with open(path) as f:
        return json.load(f)


def register_best_models():
    client = MlflowClient()

    for komoditas in KOMODITAS_LIST:
        print(f"\n[ {komoditas.upper()} ]")

        for suffix, experiment_name in TARGETS.items():
            label = "Prediksi Besok" if suffix == "1d" else "Deteksi Lonjakan 7 Hari"
            print(f"  --- {label} ---")

            try:
                info = load_best_model_info(komoditas, suffix)
            except FileNotFoundError as e:
                print(f"  ❌ {e}")
                continue

            model_type = info["best_model"]
            target = info["target"]

            # Cari experiment
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"  ❌ Experiment '{experiment_name}' tidak ditemukan, skip.")
                continue

            # Cari run terbaik
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    f"params.model_type = '{model_type}' "
                    f"AND params.komoditas = '{komoditas}' "
                    f"AND params.target = '{target}'"
                ),
                order_by=["metrics.mape ASC"],
                max_results=1
            )

            if not runs:
                print(f"  ❌ Run tidak ditemukan, skip.")
                continue

            run = runs[0]
            run_id = run.info.run_id
            mape = run.data.metrics.get("mape_pct", 0)
            rmse = run.data.metrics.get("rmse", 0)

            # Nama model: harga-pangan-beras-1d / harga-pangan-beras-7d
            model_name = f"harga-pangan-{komoditas}-{suffix}"
            model_uri = f"runs:/{run_id}/model"

            registered = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            # Set ke Staging
            client.transition_model_version_stage(
                name=model_name,
                version=registered.version,
                stage="Staging",
                archive_existing_versions=True
            )

            # Deskripsi dan tag
            client.update_registered_model(
                name=model_name,
                description=f"Model {model_type} untuk {label} harga {komoditas} di Jawa Timur"
            )
            client.set_model_version_tag(name=model_name, version=registered.version,
                                         key="mape_pct", value=str(round(mape, 4)))
            client.set_model_version_tag(name=model_name, version=registered.version,
                                         key="rmse", value=str(round(rmse, 2)))
            client.set_model_version_tag(name=model_name, version=registered.version,
                                         key="auto_registered", value="true")

            print(f"  ✅ Registered: {model_name} v{registered.version} → Staging")
            print(f"  Model : {model_type} | MAPE: {mape:.4f}% | RMSE: {rmse:.2f}")

    print("\n" + "="*60)
    print("Model registry selesai!")
    print("Lihat di: https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan/models")
    print("="*60)


if __name__ == "__main__":
    register_best_models()