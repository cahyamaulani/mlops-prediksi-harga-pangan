import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

# ==============================
# CONFIG
# ==============================

SLIDING_WINDOW_DAYS = 365

FEATURES = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "trend",
    "day_of_week", "month", "year",
    "is_ramadan", "is_end_of_month", "is_start_of_month"
]

KOMODITAS_LIST = {
    "beras": "features_beras.csv",
    "telur_ayam": "features_telur_ayam.csv",
    "daging_ayam": "features_daging_ayam.csv",
}

# 2 target: prediksi besok + deteksi lonjakan 7 hari
TARGETS = {
    "target_1d": "Prediksi Harga Besok",
    "target_7d": "Deteksi Lonjakan 7 Hari",
}

# Threshold alert lonjakan (%)
ALERT_THRESHOLD_PCT = 10.0

# ==============================
# SETUP MLFLOW → DAGSHUB
# ==============================

def setup_mlflow():
    dagshub.init(
        repo_owner="cahyamaulani",
        repo_name="mlops-prediksi-harga-pangan",
        mlflow=True
    )


# ==============================
# LOAD DATA
# ==============================

def load_data(filename, target):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(base_dir, "data", "features", filename)
    df = pd.read_csv(path)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)

    cutoff = df["tanggal"].max() - pd.Timedelta(days=SLIDING_WINDOW_DAYS)
    df = df[df["tanggal"] >= cutoff]

    X = df[FEATURES]
    y = df[target]

    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ==============================
# EVALUASI MODEL
# ==============================

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))
    return {
        "mape": round(float(mape), 6),
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "mape_pct": round(float(mape) * 100, 4)
    }


# ==============================
# TRAINING PER MODEL
# ==============================

def train_all_models(X_train, X_test, y_train, y_test, komoditas, target):
    results = {}

    # Linear Regression
    with mlflow.start_run(run_name=f"LinearRegression_{komoditas}_{target}"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params({"model_type": "LinearRegression", "komoditas": komoditas,
                           "target": target, "sliding_window_days": SLIDING_WINDOW_DAYS})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        results["LinearRegression"] = (model, metrics)
        print(f"  LinearRegression    | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")

    # Random Forest
    rf_variants = [
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 10, "min_samples_split": 3},
    ]
    best_rf, best_rf_metrics = None, {"mape": float("inf")}
    for i, params in enumerate(rf_variants):
        with mlflow.start_run(run_name=f"RandomForest_v{i+1}_{komoditas}_{target}"):
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_params({"model_type": "RandomForest", "komoditas": komoditas,
                               "target": target, "sliding_window_days": SLIDING_WINDOW_DAYS, **params})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"  RandomForest v{i+1}    | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")
            if metrics["mape"] < best_rf_metrics["mape"]:
                best_rf, best_rf_metrics = model, metrics
    results["RandomForest"] = (best_rf, best_rf_metrics)

    # XGBoost
    xgb_variants = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.08,
         "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
         "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 7},
    ]
    best_xgb, best_xgb_metrics = None, {"mape": float("inf")}
    for i, params in enumerate(xgb_variants):
        with mlflow.start_run(run_name=f"XGBoost_v{i+1}_{komoditas}_{target}"):
            model = XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_params({"model_type": "XGBoost", "komoditas": komoditas,
                               "target": target, "sliding_window_days": SLIDING_WINDOW_DAYS, **params})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"  XGBoost v{i+1}         | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")
            if metrics["mape"] < best_xgb_metrics["mape"]:
                best_xgb, best_xgb_metrics = model, metrics
    results["XGBoost"] = (best_xgb, best_xgb_metrics)

    return results


# ==============================
# SAVE BEST MODEL INFO
# ==============================

def save_best_model_info(komoditas, target, model_type, metrics):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    info = {
        "komoditas": komoditas,
        "target": target,
        "best_model": model_type,
        "metrics": metrics,
        "sliding_window_days": SLIDING_WINDOW_DAYS,
        "alert_threshold_pct": ALERT_THRESHOLD_PCT if target == "target_7d" else None,
        "trained_at": pd.Timestamp.now().isoformat()
    }

    # Nama file: best_model_beras_1d.json / best_model_beras_7d.json
    suffix = "1d" if target == "target_1d" else "7d"
    path = os.path.join(models_dir, f"best_model_{komoditas}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {path}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    setup_mlflow()

    print("=" * 60)
    print("TRAINING SEMUA MODEL - PREDIKSI HARGA PANGAN JAWA TIMUR")
    print("=" * 60)

    for komoditas, filename in KOMODITAS_LIST.items():
        print(f"\n{'='*60}")
        print(f"KOMODITAS: {komoditas.upper()}")
        print(f"{'='*60}")

        for target, deskripsi in TARGETS.items():
            print(f"\n--- {deskripsi} ({target}) ---")
            mlflow.set_experiment(f"Prediksi Harga Pangan - {target}")

            X_train, X_test, y_train, y_test = load_data(filename, target)
            print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

            results = train_all_models(X_train, X_test, y_train, y_test, komoditas, target)

            # Pilih best model
            best_name = min(results, key=lambda x: results[x][1]["mape"])
            best_metrics = results[best_name][1]

            print(f"\n  ✅ Best model [{target}]: {best_name}")
            print(f"     MAPE: {best_metrics['mape_pct']:.2f}% | RMSE: {best_metrics['rmse']}")

            save_best_model_info(komoditas, target, best_name, best_metrics)

    print("\n" + "="*60)
    print("TRAINING SELESAI!")
    print("Lihat eksperimen di: https://dagshub.com/cahyamaulani/mlops-prediksi-harga-pangan.mlflow")
    print("="*60)