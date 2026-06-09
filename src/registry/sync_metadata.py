import os
import json
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

dagshub.init(
    repo_owner="cahyamaulani",
    repo_name="mlops-prediksi-harga-pangan",
    mlflow=True
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = ["1d", "7d"]


def sync_model_metadata():
    client = MlflowClient()
    models_metadata = {}

    for komoditas in KOMODITAS_LIST:
        models_metadata[komoditas] = {}
        for suffix in TARGETS:
            model_name = f"harga-pangan-{komoditas}-{suffix}"
            try:
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if not versions:
                    print(f"  ⚠️  {model_name} belum Production, skip.")
                    continue

                v = versions[0]
                run = client.get_run(v.run_id)

                models_metadata[komoditas][suffix] = {
                    "model_name": model_name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "model_type": run.data.params.get("model_type", "-"),
                    "mape_pct": run.data.metrics.get("mape_pct", 0),
                    "rmse": run.data.metrics.get("rmse", 0),
                    "sliding_window_days": run.data.params.get("sliding_window_days", 365),
                    "target": run.data.params.get("target", suffix),
                    "run_id": v.run_id,
                }
                print(f"  ✅ {model_name} v{v.version} ({v.current_stage})")

            except Exception as e:
                print(f"  ❌ {model_name}: {e}")

    metadata = {
        "synced_at": datetime.now().isoformat(),
        "mlflow_tracking_uri": "https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan.mlflow",
        "models": models_metadata,
        "data_lineage": {
            "raw_data": "data/raw/harga_pangan.csv",
            "processed_data": "data/processed/",
            "features_data": "data/features/",
            "dvc_tracked": True,
            "dvc_remote": "dagshub"
        }
    }

    out_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Sinkronisasi metadata dari DagsHub MLflow...\n")
    path = sync_model_metadata()
    print("\nJalankan DVC untuk track metadata:")
    print("dvc add models/model_metadata.json && dvc push --remote dagshub")