import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def simulate_price_shift():
    """
    Simulasi lonjakan harga mendadak +20% 
    untuk trigger drift detection dan retraining.
    """
    raw_path = os.path.join(BASE_DIR, "data", "raw", "harga_pangan.csv")
    df = pd.read_csv(raw_path)
    df["tanggal"] = pd.to_datetime(df["tanggal"], format='mixed')

    last_date = df["tanggal"].max()
    new_rows = []

    print("Simulasi shifted data (+20% harga) untuk 7 hari ke depan...")

    for i in range(1, 8):
        new_date = last_date + timedelta(days=i)
        for _, group in df[df["tanggal"] == last_date].iterrows():
            new_rows.append({
                "ProvID": group["ProvID"],
                "Provinsi": group["Provinsi"],
                "tanggal": str(new_date.date()),
                "commodity_id": group["commodity_id"],
                "commodity_name": group["commodity_name"],
                "Nilai": group["Nilai"] * 1.20,  # +20% shift
                "ingest_time": datetime.now().isoformat()
            })

    df_new = pd.concat([df, pd.DataFrame(new_rows)]).drop_duplicates()
    df_new.to_csv(raw_path, index=False)

    print(f"Total rows setelah shift: {len(df_new)}")
    df_new["tanggal"] = pd.to_datetime(df_new["tanggal"])
    print(f"Tanggal terakhir: {df_new['tanggal'].max()}")
    print("Shifted data berhasil ditambahkan!")

if __name__ == "__main__":
    simulate_price_shift()