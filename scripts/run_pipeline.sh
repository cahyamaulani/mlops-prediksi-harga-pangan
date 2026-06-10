#!/bin/bash
set -e

echo "======================================"
echo "MLOps Pipeline - Prediksi Harga Pangan"
echo "======================================"

# Load env variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
fi

echo ""
echo "=== Stage 1: Data Ingestion ==="
python src/data/ingest_data.py incremental  
echo "✅ Ingest selesai"

echo ""
echo "=== Stage 2: Preprocessing ==="
python src/data/preprocess.py
echo "✅ Preprocessing selesai"

echo ""
echo "=== Stage 3: Feature Engineering ==="
python src/features/feature_engineering.py
echo "✅ Feature engineering selesai"

echo ""
echo "=== Stage 4: Drift Detection ==="
python src/monitoring/drift_detection.py
echo "✅ Drift detection selesai"

echo ""
echo "=== Stage 5: DVC Push ke DagsHub ==="
dvc add data/raw/harga_pangan.csv   
dvc push --remote dagshub
echo "✅ DVC push selesai"

echo ""
echo "=== Stage 6: Commit ke GitHub ==="
git add data/raw/harga_pangan.csv.dvc
git add data/processed/drift_report.json || true
git diff --staged --quiet || git commit -m "data: update dataset $(date +%Y-%m-%d) [skip ci]"
# [skip ci] → supaya commit ini TIDAK memicu GitHub Actions lagi (infinite loop!)
git push origin main
echo "✅ Git push selesai"

echo ""
echo "======================================"
echo "Pipeline selesai!"
echo "GitHub Actions akan otomatis trigger"
echo "training & registry saat push kode"
echo "(bukan saat push data [skip ci])"
echo "======================================"