# mlops-prediksi-harga-pangan
Sistem Early Warning dan Prediksi Lonjakan Harga Pangan di Jawa Timur

# 🌾 Sistem Early Warning & Prediksi Harga Pangan Jawa Timur

> Sistem MLOps end-to-end untuk memantau dan memprediksi harga pangan strategis (Beras, Telur Ayam, Daging Ayam) di Jawa Timur secara otomatis.


---

## 📋 Deskripsi Proyek

Sistem ini mengintegrasikan pipeline MLOps lengkap mulai dari pengambilan data otomatis, pelatihan model, evaluasi, registrasi model, hingga serving prediksi secara real-time. Data harga pangan diambil otomatis dari **BI PIHPS (Bank Indonesia – Pusat Informasi Harga Pangan Strategis)** setiap hari, diproses, dan digunakan untuk menghasilkan prediksi harga 1 hari dan 7 hari ke depan.

**Komoditas yang diprediksi:**
- 🌾 Beras
- 🥚 Telur Ayam
- 🍗 Daging Ayam

---

## 🏗️ Arsitektur Sistem

```
BI PIHPS (Sumber Data)
        │
        ▼ scraping otomatis (setiap hari 05.00 WIB)
ingest_data.py (incremental)
        │
        ▼
DVC + DagsHub (versioning data)
        │
        ▼
preprocess.py → feature_engineering.py
        │
        ▼
drift_detection.py (PSI monitoring)
        │
        ▼
train.py (LinearRegression | RandomForest | XGBoost)
        │
        ▼
MLflow Registry (DagsHub) → Staging → Production
        │
        ▼
FastAPI (model serving + Prometheus metrics)
        │
        ▼
Streamlit Dashboard (visualisasi & early warning)
        │
Prometheus + Grafana (monitoring runtime)
```

---

## 🗂️ Struktur Direktori

```
mlops-prediksi-harga-pangan/
├── .github/
│   └── workflows/
│       ├── mlops-automation.yaml       # CI/CD pipeline (trigger: push)
│       └── continous_training.yaml     # Retraining mingguan otomatis
│
├── data/
│   ├── raw/
│   │   └── harga_pangan.csv.dvc        # Pointer DVC ke data mentah
│   ├── processed/
│   │   ├── harga_beras.csv             # Data bersih per komoditas
│   │   ├── harga_telur_ayam.csv
│   │   ├── harga_daging_ayam.csv
│   │   └── drift_report.json           # Hasil deteksi drift (PSI)
│   └── features/
│       ├── features_beras.csv          # Fitur siap training per komoditas
│       ├── features_telur_ayam.csv
│       └── features_daging_ayam.csv
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py              # Scraping data dari BI PIHPS
│   │   └── preprocess.py              # Cleaning & transformasi data
│   ├── features/
│   │   └── feature_engineering.py     # Pembuatan 13 fitur time-series
│   ├── models/
│   │   └── train.py                   # Training semua model + log MLflow
│   ├── inference/
│   │   ├── api.py                     # FastAPI endpoint prediksi
│   │   └── verify_inference.py        # Verifikasi model serving
│   ├── monitoring/
│   │   └── drift_detection.py         # Deteksi data drift (PSI)
│   └── registry/
│       ├── register_model.py          # Register model terbaik ke MLflow
│       └── sync_metadata.py           # Sinkronisasi metadata model
│
├── scripts/
│   ├── run_pipeline.sh                # Script manual jalankan pipeline
│   ├── evaluate_and_validate.py       # Evaluasi & validasi model
│   ├── compare_models.py              # Bandingkan & promote model
│   ├── check_triggers.py              # Cek apakah perlu retraining
│   └── simulate_shifted_data.py       # Simulasi data drift untuk testing
│
├── models/
│   ├── best_model_beras_1d.json       # Info model terbaik per komoditas
│   ├── best_model_beras_7d.json
│   ├── best_model_telur_ayam_1d.json
│   ├── best_model_telur_ayam_7d.json
│   ├── best_model_daging_ayam_1d.json
│   └── best_model_daging_ayam_7d.json
│
├── dashboard/
│   └── app.py                         # Dashboard Streamlit
│
├── monitoring/
│   └── prometheus.yml                 # Konfigurasi scraping Prometheus
│
├── tests/
│   └── test_pipeline.py               # Unit test pipeline
│
├── Dockerfile                         # Image Docker untuk FastAPI
├── docker-compose.yaml                # Orkestrasi semua service
├── requirements.txt                   # Dependencies training & dashboard
├── requirements-api.txt               # Dependencies FastAPI
└── models.dvc                         # Pointer DVC untuk model artifacts
```

---

## 📊 Data & Fitur

### Sumber Data
Data diambil dari **BI PIHPS** (Bank Indonesia – Pusat Informasi Harga Pangan Strategis) secara otomatis setiap hari menggunakan mode **incremental** — hanya data baru yang belum tersimpan yang diambil.

### Struktur Data Raw (`harga_pangan.csv`)
| Kolom | Tipe | Keterangan |
|---|---|---|
| `ingested_at` | datetime | Waktu data diambil |
| `province_id` | float | ID provinsi (16 = Jawa Timur) |
| `province_name` | string | Nama provinsi |
| `tanggal_str` | string | Tanggal format string |
| `komoditas` | string | Nama komoditas |
| `harga` | float | Harga dalam Rupiah |
| `perubahan_str` | string | Perubahan harga (string) |
| `harga_cleaned` | float | Harga setelah cleaning |
| `komoditas_id` | float | ID komoditas |
| `harga_referensi` | float | Harga referensi nasional |
| `pct_change` | float | Persentase perubahan |
| `is_jawa_timur` | bool | Flag Jawa Timur |
| `tanggal` | date | Tanggal (datetime) |
| `komoditas_clean` | string | Nama komoditas terstandarisasi |

### 13 Fitur Time-Series (Input Model)
| Fitur | Keterangan |
|---|---|
| `lag_1` | Harga 1 hari lalu |
| `lag_7` | Harga 7 hari lalu |
| `lag_14` | Harga 14 hari lalu |
| `rolling_mean_7` | Rata-rata harga 7 hari |
| `rolling_mean_14` | Rata-rata harga 14 hari |
| `rolling_std_7` | Standar deviasi harga 7 hari |
| `trend` | Tren linear harga |
| `day_of_week` | Hari dalam seminggu (0–6) |
| `month` | Bulan (1–12) |
| `year` | Tahun |
| `is_ramadan` | Flag periode Ramadan |
| `is_end_of_month` | Flag akhir bulan |
| `is_start_of_month` | Flag awal bulan |

### Target Prediksi
- **`target_1d`** — harga besok (1 hari ke depan)
- **`target_7d`** — harga 7 hari ke depan (untuk deteksi lonjakan)

---

## 🤖 Model Machine Learning

Training menggunakan **sliding window 365 hari** terakhir dengan split 80% train / 20% test.

### Model yang Dilatih
| Model | Varian | Keterangan |
|---|---|---|
| LinearRegression | 1 | Baseline model |
| RandomForest | 3 varian | n_estimators: 100/200/300, max_depth: 5/8/10 |
| XGBoost | 3 varian | n_estimators: 100/250/400, learning_rate: 0.1/0.08/0.05 |

**Total: 7 model per komoditas × 2 target × 3 komoditas = 42 model per training run**

### Pemilihan Model Terbaik
Model terbaik dipilih berdasarkan **MAPE (Mean Absolute Percentage Error) terkecil** pada data test. Info model terbaik disimpan di `models/best_model_{komoditas}_{suffix}.json` dan digunakan oleh `register_model.py` untuk registrasi ke MLflow.

### Early Warning
Alert dipicu jika prediksi 7 hari ke depan menunjukkan kenaikan **≥ 5%** dari harga sekarang.

---

## 🔄 CI/CD Pipeline

### MLOps Automation Pipeline (`mlops-automation.yaml`)
Trigger: **setiap push/PR ke branch `main`** + **scheduled setiap hari 05.00 WIB**

```
Push ke GitHub
      │
      ├── [Scheduled] Daily Ingest
      │       ingest → preprocess → feature engineering
      │       → drift detection → retrain jika PSI ≥ 0.2
      │       → DVC push → git commit [skip ci]
      │
      └── [Push/PR] CI/CD Pipeline
              Stage 1: Automated Testing (pytest)
              Stage 2: Training (7 model per komoditas)
              Stage 3: Evaluation & Validation
              Stage 4: Auto Registry Update (MLflow)
```

### Continuous Training Pipeline (`continous_training.yaml`)
Trigger: **setiap Minggu 00.00 WIB** + manual via `workflow_dispatch`

```
Setiap Minggu
      │
      ├── check-triggers: cek drift & kondisi retraining
      │
      ├── retrain: training ulang jika perlu
      │
      └── evaluate-and-promote: compare_models.py
                → promote ke Production jika lebih baik
```

---

## 🐳 Docker Services

Semua service dijalankan via Docker Compose:

| Service | Port | Keterangan |
|---|---|---|
| **FastAPI** | 8000 | REST API prediksi (3 replika) |
| **Streamlit** | 8501 | Dashboard visualisasi |
| **MLflow** | 5000 | Experiment tracking (lokal) |
| **Prometheus** | 9090 | Metrics scraping dari FastAPI |
| **Grafana** | 3000 | Dashboard monitoring runtime |

```bash
# Jalankan semua service
docker compose up -d

# Cek status
docker compose ps

# Restart API setelah model update
docker compose restart api
```

---

## 🚀 Cara Menjalankan

### Prasyarat
- Python 3.10+
- Docker & Docker Compose
- DVC
- Akun DagsHub
- GitHub Secrets: `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`

### Setup Awal
```bash
# Clone repo
git clone https://github.com/cahyamaulani/mlops-prediksi-harga-pangan.git
cd mlops-prediksi-harga-pangan

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env dengan kredensial DagsHub kamu

# Pull data dari DagsHub
dvc pull --remote dagshub

# Jalankan semua service
docker compose up -d
```

### Menjalankan Pipeline Manual
```bash
# Jalankan full pipeline (ingest → preprocess → feature engineering → drift → push)
bash scripts/run_pipeline.sh

# Atau per tahap:
python src/data/ingest_data.py incremental
python src/data/preprocess.py
python src/features/feature_engineering.py
python src/monitoring/drift_detection.py
python src/models/train.py
python src/registry/register_model.py
```

### Akses Layanan
| Layanan | URL |
|---|---|
| FastAPI Docs | http://localhost:8000/docs |
| Prediksi Beras | http://localhost:8000/predict/beras |
| Prediksi Semua | http://localhost:8000/predict/all/summary |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## 📈 Monitoring

### Data Drift (PSI)
Drift dideteksi menggunakan **Population Stability Index (PSI)**:
| PSI | Status | Aksi |
|---|---|---|
| < 0.1 | 🟢 Stabil | Tidak ada aksi |
| 0.1 – 0.2 | 🟡 Monitor | Pantau lebih sering |
| ≥ 0.2 | 🔴 Drift | Retrain otomatis |

### Metrics Prometheus
- `api_request_total` — total request per endpoint & komoditas
- `api_request_latency_seconds` — latensi request
- `api_prediction_value` — distribusi nilai prediksi harga

---

## 🔢 Versioning

| Komponen | Tool | Penyimpanan |
|---|---|---|
| **Kode** | Git | GitHub |
| **Data** | DVC | DagsHub Storage |
| **Model & Eksperimen** | MLflow | DagsHub MLflow |

```bash
# Cek versi data
git log --oneline -- data/raw/harga_pangan.csv.dvc

# Rollback ke versi data tertentu
git checkout <commit-hash>
dvc pull --remote dagshub
```

---

## 🔗 Links

- **DagsHub Repository:** https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan
- **MLflow Tracking:** https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan.mlflow
- **GitHub Actions:** https://github.com/cahyamaulani/mlops-prediksi-harga-pangan/actions

---

## 👩‍💻 Author

**Cahya Maulani**  
Universitas Brawijaya — Sistem Early Warning Harga Pangan Jawa Timur
