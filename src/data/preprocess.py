import os
import pandas as pd

COMMODITY_MAP = {
    1: "Beras",
    4: "Telur Ayam",
    2: "Daging Ayam"
}


def load_raw():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    path = os.path.join(base_dir, "data", "raw", "harga_pangan.csv")
    df = pd.read_csv(path)
    return df


def preprocess(df):
    # Pastikan kolom tanggal datetime
    df["tanggal"] = pd.to_datetime(df["tanggal"], format='mixed')

    # Pilih kolom yang relevan
    df = df[["tanggal", "commodity_name", "Provinsi", "Nilai"]].copy()
    df.columns = ["tanggal", "komoditas", "provinsi", "harga"]

    # Hapus nilai null & duplikat
    df = df.dropna(subset=["harga"])
    df = df.drop_duplicates(subset=["tanggal", "komoditas"])

    # Sort by tanggal
    df = df.sort_values("tanggal").reset_index(drop=True)

    return df


def save_processed(df):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    out_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    # Simpan per komoditas
    for komoditas in df["komoditas"].unique():
        df_k = df[df["komoditas"] == komoditas].copy()
        nama_file = komoditas.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"harga_{nama_file}.csv")
        df_k.to_csv(path, index=False)
        print(f"Saved: {path} ({len(df_k)} rows)")

    # Simpan juga gabungan
    path_all = os.path.join(out_dir, "harga_all.csv")
    df.to_csv(path_all, index=False)
    print(f"Saved: {path_all} ({len(df)} rows)")


if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = preprocess(df_raw)
    save_processed(df_clean)
    print("\nPreprocessing selesai!")
    print(df_clean.groupby("komoditas").agg(
        jumlah=("harga", "count"),
        harga_min=("harga", "min"),
        harga_max=("harga", "max"),
    ))