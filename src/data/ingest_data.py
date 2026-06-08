import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import sys

# ==============================
# CONFIG
# ==============================

BASE_URL = "https://www.bi.go.id/hargapangan/WebSite/Home/GetGridData1"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.bi.go.id/hargapangan"
}

# Hanya 3 komoditas yang dipakai
# ID didapat dari API BI PIHPS:
# 1 = Beras Kualitas Medium I
# 4 = Cabai Merah Besar  
# 7 = Daging Ayam Ras Segar
COMMODITY_MAP = {
    1: "Beras",
    4: "Telur Ayam",
    2: "Daging Ayam"
}

PROVINCE_TARGET = "Jawa Timur"


# ==============================
# FETCH DATA PER TANGGAL
# ==============================

def fetch_data_by_date(date, commodity_id):
    formatted_date = date.strftime("%m/%d/%Y")

    params = {
        "tanggal": formatted_date,
        "commodity": commodity_id,
        "priceType": 1,
        "isPasokan": 1,
        "jenis": 1,
        "periode": 1,
        "provId": 16,
    }

    response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json()


# ==============================
# FETCH DATA 2 TAHUN (INITIAL LOAD)
# ==============================

def fetch_two_years():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 2)

    all_data = []
    total_days = (end_date - start_date).days

    print(f"Total hari: {total_days}")
    print(f"Komoditas: {list(COMMODITY_MAP.values())}\n")

    for cid, nama in COMMODITY_MAP.items():
        print(f"\n===== {nama} (ID: {cid}) =====")

        current = start_date
        day_count = 0

        while current <= end_date:
            day_count += 1
            print(f"[{nama}] {day_count}/{total_days} → {current.date()}")

            try:
                response_data = fetch_data_by_date(current, cid)
                data = response_data.get("data", [])

                if data:
                    for row in data:
                        # Filter hanya Jawa Timur
                        if row.get("Provinsi") == PROVINCE_TARGET:
                            row["tanggal"] = str(current.date())
                            row["commodity_id"] = cid
                            row["commodity_name"] = nama
                            all_data.append(row)

            except Exception as e:
                print(f"Error {nama} - {current.date()}: {e}")

            time.sleep(0.3)
            current += timedelta(days=1)

    return pd.DataFrame(all_data)


# ==============================
# FETCH DATA HARIAN (INCREMENTAL)
# ==============================

def fetch_recent_data(days=1):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    all_data = []

    for cid, nama in COMMODITY_MAP.items():
        current = start_date

        while current <= end_date:
            print(f"[INCREMENTAL] {nama} → {current.date()}")

            try:
                response_data = fetch_data_by_date(current, cid)
                data = response_data.get("data", [])

                if data:
                    for row in data:
                        if row.get("Provinsi") == PROVINCE_TARGET:
                            row["tanggal"] = str(current.date())
                            row["commodity_id"] = cid
                            row["commodity_name"] = nama
                            all_data.append(row)

            except Exception as e:
                print(f"Error {nama} - {current.date()}: {e}")

            time.sleep(0.3)
            current += timedelta(days=1)

    return pd.DataFrame(all_data)


# ==============================
# SAVE DATA
# ==============================

def save_data(df_new):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "harga_pangan.csv")

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["tanggal", "commodity_id", "Provinsi"]
        )
    else:
        df = df_new

    df["ingest_time"] = datetime.now().isoformat()
    df.to_csv(file_path, index=False)

    print(f"\nDONE! Saved: {file_path}")
    print(f"Total rows: {len(df)}")
    return df


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    mode = "incremental"

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "full":
        print("Full ingestion (2 tahun, 3 komoditas)")
        df = fetch_two_years()
    else:
        print("Incremental ingestion (harian)")
        df = fetch_recent_data(days=1)

    save_data(df)