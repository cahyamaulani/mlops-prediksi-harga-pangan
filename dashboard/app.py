import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

KOMODITAS_LIST = {
    "beras": "🌾 Beras",
    "telur_ayam": "🥚 Telur Ayam",
    "daging_ayam": "🍗 Daging Ayam"
}

KOMODITAS_COLORS = {
    "beras": "#F59E0B",
    "telur_ayam": "#EF4444",
    "daging_ayam": "#8B5CF6"
}

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Sistem Early Warning Harga Pangan Jawa Timur",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CSS
# ==============================

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d5986);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .alert-box {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    .normal-box {
        background: linear-gradient(135deg, #14532d, #15803d);
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    .title-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# HELPER FUNCTIONS
# ==============================

@st.cache_data(ttl=300)  # Cache 5 menit
def fetch_prediction(komoditas):
    try:
        response = requests.get(f"{API_BASE_URL}/predict/{komoditas}", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def load_historical_data(komoditas):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    path = os.path.join(base_dir, "data", "processed", f"harga_{komoditas}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    return df.sort_values("tanggal")


def load_drift_report():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    path = os.path.join(base_dir, "data", "processed", "drift_report.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def format_rupiah(value):
    return f"Rp {value:,.0f}"


# ==============================
# MAIN DASHBOARD
# ==============================

def main():
    # Header
    st.markdown("""
    <div class="title-header">
        <h1>🌾 Sistem Early Warning & Prediksi Harga Pangan</h1>
        <p>Jawa Timur | Beras • Telur Ayam • Daging Ayam</p>
    </div>
    """, unsafe_allow_html=True)

    # Timestamp
    st.caption(f"🕐 Data diperbarui: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}")

    # ==============================
    # SIDEBAR
    # ==============================
    with st.sidebar:
        st.title("⚙️ Pengaturan")
        selected_komoditas = st.selectbox(
            "Pilih Komoditas",
            options=list(KOMODITAS_LIST.keys()),
            format_func=lambda x: KOMODITAS_LIST[x]
        )
        show_all = st.checkbox("Tampilkan semua komoditas", value=True)
        historical_days = st.slider("Hari historis ditampilkan", 30, 365, 90)

        st.divider()
        st.markdown("### 📊 Info Model")
        st.markdown("""
        - **Model:** RandomForest & XGBoost
        - **Target:** Prediksi 1 & 7 hari
        - **Fitur:** 13 fitur time-series
        - **Sliding Window:** 365 hari
        """)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ==============================
    # SECTION 1: RINGKASAN SEMUA KOMODITAS
    # ==============================
    st.subheader("📊 Ringkasan Harga & Prediksi Semua Komoditas")

    cols = st.columns(3)
    all_predictions = {}

    for i, (k, label) in enumerate(KOMODITAS_LIST.items()):
        with cols[i]:
            with st.spinner(f"Memuat prediksi {label}..."):
                pred = fetch_prediction(k)
                all_predictions[k] = pred

            if pred:
                alert = pred["early_warning"]["alert"]
                perubahan = pred["early_warning"]["perubahan_pct"]

                # Status color
                delta_color = "inverse" if alert else "normal"

                st.metric(
                    label=f"{label}",
                    value=format_rupiah(pred["harga_sekarang"]),
                    delta=f"{perubahan:+.2f}% (7 hari)",
                    delta_color=delta_color
                )

                if alert:
                    st.error(f"⚠️ WASPADA LONJAKAN!")
                else:
                    st.success(f"✅ Normal")

                st.caption(f"Besok: {format_rupiah(pred['prediksi']['besok'])}")
                st.caption(f"7 Hari: {format_rupiah(pred['prediksi']['7_hari'])}")
            else:
                st.error(f"❌ Gagal memuat data {label}")

    st.divider()

    # ==============================
    # SECTION 2: DETAIL KOMODITAS TERPILIH
    # ==============================
    st.subheader(f"📈 Detail: {KOMODITAS_LIST[selected_komoditas]}")

    pred = all_predictions.get(selected_komoditas)
    df_hist = load_historical_data(selected_komoditas)

    if pred and df_hist is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Grafik historis + prediksi
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            # Data historis
            df_plot = df_hist.tail(historical_days)
            color = KOMODITAS_COLORS[selected_komoditas]

            ax.plot(
                df_plot["tanggal"], df_plot["harga"],
                color=color, linewidth=2, label="Harga Aktual"
            )
            ax.fill_between(
                df_plot["tanggal"], df_plot["harga"],
                alpha=0.1, color=color
            )

            # Titik prediksi
            last_date = df_plot["tanggal"].iloc[-1]
            last_price = df_plot["harga"].iloc[-1]

            pred_dates = [
                last_date + timedelta(days=1),
                last_date + timedelta(days=7)
            ]
            pred_prices = [
                pred["prediksi"]["besok"],
                pred["prediksi"]["7_hari"]
            ]

            ax.plot(
                [last_date] + pred_dates,
                [last_price] + pred_prices,
                color="#60a5fa", linewidth=2,
                linestyle="--", label="Prediksi", marker="o"
            )

            # Styling
            ax.set_xlabel("Tanggal", color="white")
            ax.set_ylabel("Harga (Rp)", color="white")
            ax.tick_params(colors="white")
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(facecolor='#1e1e2e', labelcolor='white')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 🎯 Prediksi")
            st.markdown(f"""
            | Horizon | Harga |
            |---|---|
            | Harga Sekarang | **{format_rupiah(pred['harga_sekarang'])}** |
            | Besok (1 hari) | **{format_rupiah(pred['prediksi']['besok'])}** |
            | 7 Hari ke Depan | **{format_rupiah(pred['prediksi']['7_hari'])}** |
            """)

            st.divider()
            st.markdown("### 🚨 Early Warning")

            alert = pred["early_warning"]["alert"]
            perubahan = pred["early_warning"]["perubahan_pct"]

            if alert:
                st.markdown(f"""
                <div class="alert-box">
                    <h3>⚠️ WASPADA!</h3>
                    <p>Potensi lonjakan harga dalam 7 hari ke depan</p>
                    <h2>{perubahan:+.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-box">
                    <h3>✅ NORMAL</h3>
                    <p>Harga diprediksi stabil</p>
                    <h2>{perubahan:+.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            st.caption(f"Data per: {pred['tanggal_data']}")

    st.divider()

    # ==============================
    # SECTION 3: DATA DRIFT MONITORING
    # ==============================
    st.subheader("🔍 Data Drift Monitoring (PSI)")

    drift_report = load_drift_report()

    if drift_report:
        st.caption(f"Laporan dibuat: {drift_report.get('generated_at', 'N/A')}")

        drift_cols = st.columns(3)
        komoditas_keys = list(drift_report["results"].keys())

        for i, k in enumerate(komoditas_keys):
            info = drift_report["results"][k]
            with drift_cols[i % 3]:
                psi = info["psi"]
                status = info["status"]
                color = info["color"]
                perubahan = info["perubahan_pct"]

                if color == "red":
                    st.error(f"**{k.replace('_', ' ').title()}**\nPSI: {psi:.4f}\n{status}")
                elif color == "orange":
                    st.warning(f"**{k.replace('_', ' ').title()}**\nPSI: {psi:.4f}\n{status}")
                else:
                    st.success(f"**{k.replace('_', ' ').title()}**\nPSI: {psi:.4f}\n{status}")

                st.caption(f"Perubahan rata-rata: {perubahan:+.1f}%")
    else:
        st.info("Drift report belum tersedia. Jalankan drift_detection.py terlebih dahulu.")

    st.divider()

    # ==============================
    # SECTION 4: TABEL DATA HISTORIS
    # ==============================

    with st.expander("📋 Lihat Data Historis"):
        if df_hist is not None:
            df_show = df_hist.tail(30).copy()
            df_show["harga"] = df_show["harga"].apply(format_rupiah)
            df_show["tanggal"] = df_show["tanggal"].dt.strftime("%d %b %Y")
            df_show = df_show.rename(columns={
                "tanggal": "Tanggal",
                "komoditas": "Komoditas",
                "harga": "Harga"
            })
            st.dataframe(df_show[["Tanggal", "Harga"]], use_container_width=True)


if __name__ == "__main__":
    main()