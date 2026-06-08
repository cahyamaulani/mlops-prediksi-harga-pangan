import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

# ==============================
# CONFIG
# ==============================

SLIDING_WINDOW_DAYS = 365  # 12 bulan terakhir

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

# Target yang diprediksi: ganti ke "target_7d" untuk prediksi 7 hari
TARGET = "target_1d"

# ==============================
# LOAD DATA
# ==============================

def load_data(filename, target):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    path = os.path.join(base_dir, "data", "features", filename)
    df = pd.read_csv(path)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)

    # Terapkan sliding window
    cutoff = df["tanggal"].max() - pd.Timedelta(days=SLIDING_WINDOW_DAYS)
    df = df[df["tanggal"] >= cutoff]

    X = df[FEATURES]
    y = df[target]

    # Split 80% train, 20% test (time-based, tidak random)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    return X_train, X_test, y_train, y_test


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

def train_linear_regression(X_train, X_test, y_train, y_test, komoditas):
    params = {"fit_intercept": True}

    with mlflow.start_run(run_name=f"LinearRegression_{komoditas}"):
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # Log ke MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("komoditas", komoditas)
        mlflow.log_param("sliding_window_days", SLIDING_WINDOW_DAYS)
        mlflow.log_param("target", TARGET)
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model")

        print(f"  LinearRegression | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")
        return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, komoditas):
    # 3 variasi hyperparameter
    variants = [
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5},
        {"n_estimators": 300, "max_depth": 10, "min_samples_split": 3},
    ]

    best_model, best_metrics = None, {"mape": float("inf")}

    for i, params in enumerate(variants):
        with mlflow.start_run(run_name=f"RandomForest_v{i+1}_{komoditas}"):
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("komoditas", komoditas)
            mlflow.log_param("sliding_window_days", SLIDING_WINDOW_DAYS)
            mlflow.log_param("target", TARGET)
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(model, "model")

            print(f"  RandomForest v{i+1} | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")

            if metrics["mape"] < best_metrics["mape"]:
                best_model, best_metrics = model, metrics

    return best_model, best_metrics


def train_xgboost(X_train, X_test, y_train, y_test, komoditas):
    # 3 variasi hyperparameter
    variants = [
        {
            "n_estimators": 100, "max_depth": 3,
            "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3
        },
        {
            "n_estimators": 250, "max_depth": 4,
            "learning_rate": 0.08, "subsample": 0.7,
            "colsample_bytree": 0.7, "min_child_weight": 5
        },
        {
            "n_estimators": 400, "max_depth": 5,
            "learning_rate": 0.05, "subsample": 0.6,
            "colsample_bytree": 0.6, "min_child_weight": 7
        },
    ]

    best_model, best_metrics = None, {"mape": float("inf")}

    for i, params in enumerate(variants):
        with mlflow.start_run(run_name=f"XGBoost_v{i+1}_{komoditas}"):
            model = XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("komoditas", komoditas)
            mlflow.log_param("sliding_window_days", SLIDING_WINDOW_DAYS)
            mlflow.log_param("target", TARGET)
            for k, v in params.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(model, "model")

            print(f"  XGBoost v{i+1}     | MAPE: {metrics['mape_pct']:.2f}% | RMSE: {metrics['rmse']}")

            if metrics["mape"] < best_metrics["mape"]:
                best_model, best_metrics = model, metrics

    return best_model, best_metrics


# ==============================
# SAVE BEST MODEL PER KOMODITAS
# ==============================

def save_best_model_info(komoditas, model_type, metrics):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    info = {
        "komoditas": komoditas,
        "best_model": model_type,
        "metrics": metrics,
        "sliding_window_days": SLIDING_WINDOW_DAYS,
        "target": TARGET,
        "trained_at": pd.Timestamp.now().isoformat()
    }

    path = os.path.join(models_dir, f"best_model_{komoditas}.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Best model info saved: {path}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    # Setup MLflow
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")
    mlflow.set_experiment("Prediksi Harga Pangan")

    print("=" * 60)
    print("TRAINING SEMUA MODEL - PREDIKSI HARGA PANGAN JAWA TIMUR")
    print("=" * 60)

    for komoditas, filename in KOMODITAS_LIST.items():
        print(f"\n{'='*60}")
        print(f"KOMODITAS: {komoditas.upper()}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = load_data(filename, TARGET)
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

        # Training semua model
        print("[ Linear Regression ]")
        lr_model, lr_metrics = train_linear_regression(
            X_train, X_test, y_train, y_test, komoditas
        )

        print("[ Random Forest ]")
        rf_model, rf_metrics = train_random_forest(
            X_train, X_test, y_train, y_test, komoditas
        )

        print("[ XGBoost ]")
        xgb_model, xgb_metrics = train_xgboost(
            X_train, X_test, y_train, y_test, komoditas
        )

        # Tentukan model terbaik per komoditas
        all_results = {
            "LinearRegression": lr_metrics,
            "RandomForest": rf_metrics,
            "XGBoost": xgb_metrics,
        }
        best_model_name = min(all_results, key=lambda x: all_results[x]["mape"])
        best_metrics = all_results[best_model_name]

        print(f"\nBest model untuk {komoditas}: {best_model_name}")
        print(f"   MAPE: {best_metrics['mape_pct']:.2f}% | RMSE: {best_metrics['rmse']}")

        save_best_model_info(komoditas, best_model_name, best_metrics)

    print("\n" + "="*60)
    print("TRAINING SELESAI!")
    print("Buka MLflow UI dengan:")
    print("mlflow ui --host 0.0.0.0 --port 5000")
    print("="*60)