import os
import json
from datetime import datetime

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

def sync_model_metadata():
    metadata = {
        "registered_at": datetime.now().isoformat(),
        "experiment": "Prediksi Harga Pangan",
        "models": {
            "harga-pangan-beras": {
                "version": 1,
                "alias": "production",
                "model_type": "RandomForest",
                "mape_pct": 0.16,
                "rmse": 34.41,
                "sliding_window_days": 365,
                "features": 13
            },
            "harga-pangan-telur_ayam": {
                "version": 1,
                "alias": "production",
                "model_type": "RandomForest",
                "mape_pct": 1.64,
                "rmse": 583.53,
                "sliding_window_days": 365,
                "features": 13
            },
            "harga-pangan-daging_ayam": {
                "version": 1,
                "alias": "production",
                "model_type": "RandomForest",
                "mape_pct": 1.65,
                "rmse": 734.58,
                "sliding_window_days": 365,
                "features": 13
            },
        },
        "data_lineage": {
            "raw_data": "data/raw/harga_pangan.csv",
            "processed_data": "data/processed/harga_all.csv",
            "features_data": "data/features/",
            "dvc_tracked": True
        }
    }

    out_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {out_path}")
    return out_path


if __name__ == "__main__":
    path = sync_model_metadata()
    print("\nJalankan DVC untuk track metadata:")
    print("dvc add models/model_metadata.json && dvc push")