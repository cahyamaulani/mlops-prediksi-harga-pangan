"""
Bandingkan model baru vs model lama di registry.
Promote ke Production hanya kalau model baru lebih baik.
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")
mlflow.set_tracking_uri("https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan.mlflow")

KOMODITAS_LIST = ["beras", "telur_ayam", "daging_ayam"]
TARGETS = ["1d", "7d"]
MAPE_THRESHOLD = 10.0


def get_production_mape(client, model_name):
    try:
        version = client.get_model_version_by_alias(model_name, "production")
        tags = version.tags
        mape = float(tags.get("mape_pct", 999))
        return mape, version.version
    except Exception:
        return 999, None


def compare_and_promote():
    client = MlflowClient()

    print("=" * 60)
    print("EVALUASI KOMPARATIF: MODEL BARU vs MODEL LAMA")
    print("=" * 60)

    results = {}
    promoted = 0

    for komoditas in KOMODITAS_LIST:
        for suffix in TARGETS:
            model_name = f"harga-pangan-{komoditas}-{suffix}"
            label = f"{komoditas}_{suffix}"

            print(f"\n[ {label} ]")

            # Load metrics model baru dari file
            path = os.path.join(MODELS_DIR, f"best_model_{komoditas}_{suffix}.json")
            if not os.path.exists(path):
                print(f"  File tidak ditemukan, skip")
                continue

            with open(path) as f:
                new_info = json.load(f)

            new_mape = new_info["metrics"]["mape_pct"]
            new_model_type = new_info["best_model"]

            # Get MAPE model lama di Production
            old_mape, old_version = get_production_mape(client, model_name)

            print(f"  Model lama (Production v{old_version}): MAPE={old_mape:.4f}%")
            print(f"  Model baru ({new_model_type}):           MAPE={new_mape:.4f}%")

            # Bandingkan
            if new_mape < old_mape and new_mape < MAPE_THRESHOLD:
                print(f"  ✅ Model baru LEBIH BAIK → Register & Promote ke Production")

                # Cari run terbaru
                experiment = client.get_experiment_by_name(
                    f"Prediksi Harga Pangan - target_{suffix}"
                )
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"params.model_type = '{new_model_type}' AND params.komoditas = '{komoditas}'",
                        order_by=["metrics.mape ASC"],
                        max_results=1
                    )

                    if runs:
                        run_id = runs[0].info.run_id
                        registered = mlflow.register_model(
                            model_uri=f"runs:/{run_id}/model",
                            name=model_name
                        )
                        # Set alias production ke versi baru
                        client.set_registered_model_alias(
                            model_name, "production", registered.version
                        )
                        client.set_model_version_tag(
                            model_name, registered.version,
                            "mape_pct", str(round(new_mape, 4))
                        )
                        client.set_model_version_tag(
                            model_name, registered.version,
                            "auto_promoted", "true"
                        )
                        print(f"  🚀 Promoted v{registered.version} ke Production!")
                        promoted += 1

            else:
                print(f"  ❌ Model baru TIDAK lebih baik → Production tetap v{old_version}")

            results[label] = {
                "old_mape": old_mape,
                "new_mape": new_mape,
                "promoted": new_mape < old_mape
            }

    print("\n" + "=" * 60)
    print(f"SELESAI: {promoted} model dipromosikan ke Production")
    print("=" * 60)

    # Simpan hasil
    report_path = os.path.join(MODELS_DIR, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "compared_at": datetime.now().isoformat(),
            "promoted": promoted,
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    compare_and_promote()