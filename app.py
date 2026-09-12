# APP.PY
import streamlit as st
import pandas as pd
import joblib
from model_pipeline import StuntingAggregator
import plotly.express as px

st.set_page_config(
    page_title="Composite Stunting Lens",
    layout="wide",
    page_icon="🔍"
)

st.title("🔍 Composite Stunting Lens: Dashboard Analisis Wilayah")
st.caption("Simulasi Prediksi Jumlah Balita Stunting Berbasis Indikator Komposit Rinci (Breakdown)")

@st.cache_resource
def load_aggregator():
    # Load model joblib asli yang sudah ada
    base_model = joblib.load("stunting_pipeline.joblib")
    return StuntingAggregator(base_model)

aggregator = load_aggregator()

with st.sidebar:
    st.header("⚙️ Parameter Masukan Rinci (Breakdown)")
    
    with st.expander("1. Fasilitas Kesehatan", expanded=True):
        val_posyandu = st.number_input("Aktivitas Rutin Posyandu", 0.0, 500.0, 20.0)
        val_ibu_hamil = st.number_input("Jumlah Ibu Hamil Periksa", 0.0, 500.0, 15.0)
        val_tumbuh = st.number_input("Balita Terpantau Tumbuh Kembang", 0.0, 500.0, 50.0)
        val_paud = st.number_input("Anak Mengikuti BKB/PAUD", 0.0, 500.0, 30.0)
        val_imunisasi = st.number_input("Anak Imunisasi Dasar Lengkap", 0.0, 500.0, 40.0)

    with st.expander("2. Sanitasi & Air Bersih", expanded=False):
        val_air = st.number_input("Keluarga Akses Air Bersih/Minum", 0.0, 500.0, 60.0)
        val_jamban = st.number_input("Keluarga Akses Jamban Sehat", 0.0, 500.0, 55.0)
        val_limbah = st.number_input("Keluarga Akses Sanitasi Limbah", 0.0, 500.0, 50.0)

    with st.expander("3. Layanan Gizi", expanded=False):
        val_anemia = st.number_input("Remaja Putri Periksa Anemia (Hb)", 0.0, 500.0, 25.0)
        val_ttd = st.number_input("Remaja Putri Dapat Tablet Tambah Darah", 0.0, 500.0, 20.0)
        val_ttd_bumil = st.number_input("Ibu Hamil Konsumsi TTD (90 Tablet)", 0.0, 500.0, 35.0)
        val_gizi = st.number_input("Anak Gizi Kurang/Buruk Dapat Asupan", 0.0, 500.0, 10.0)

    with st.expander("4. Fasilitas Desa", expanded=False):
        val_rembuk = st.number_input("Desa Rembuk Stunting (0/1)", 0.0, 1.0, 1.0)
        val_apb = st.number_input("Penetapan Anggaran APB Desa (0/1)", 0.0, 1.0, 1.0)
        val_alokasi = st.number_input("Jumlah Alokasi Anggaran (Rp)", 0.0, 100000000.0, 10000000.0)
        val_rpp = st.number_input("Pembentukan RDS/TPPS (0/1)", 0.0, 1.0, 1.0)
        val_kader = st.number_input("Peningkatan Kapasitas Kader/TPK (0/1)", 0.0, 1.0, 1.0)

    btn_predict = st.button("🚀 Jalankan Prediksi Stunting", type="primary", use_container_width=True)

# Susun dictionary input persis dengan nama kolom asli dataset
input_dict = {
    'aktivitas_rutin_Penyelenggaraan_posyandu_': val_posyandu,
    'jumlah_Ibu_Hamil_Periksa_Kehamilan/Nifas': val_ibu_hamil,
    'jumlah_Anak_usia_0-59_Bulan_terpantau_tumbuh_kembang_(datang_ke_posyandu/layanan_kesehatan_lainnya)': val_tumbuh,
    'jumlah_anak_usia_0-59_bulan_mengikuti_kegiatan_BKB/PAUD': val_paud,
    'jumlah_Anak_usia_0-59_bulan_mendapatkan_imunisasi_dasar_lengkap': val_imunisasi,
    'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_ke_sumber_air_bersih/minum': val_air,
    'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_ke_jamban_sehat.': val_jamban,
    'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_sanitasi/pembuangan_limbah_layak': val_limbah,
    'jumlah_Remaja_Putri_Ikut_Pemeriksaan_Anemia_(Hb)': val_anemia,
    'jumlah_Remaja_Putri_Mendapat_Tablet_Tambah_Darah_(TTB)': val_ttd,
    'jumlah_Ibu_Hamil_mengkonsumsi_tablet_tambah_darah_(TTD)_(minimal_90_tablet_selama_masa_kehamilan)': val_ttd_bumil,
    'jumlah_Anak_usia_0-59_Bulan_mengalami_gizi_kurang/buruk/stunting_mendapatkan_tambahan_asupan_gizi_dan_konseling_gizi': val_gizi,
    'desa_Melaksanakan_rembuk_stunting_desa_dengan_melibatkan_UPTD': val_rembuk,
    'terdapan_Penetapan_anggaran__kegiatan_stunting_dalam_APB_Desa': val_apb,
    'jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting': val_alokasi,
    'terdapat_Pembentukan_RDS/TPPS': val_rpp,
    'terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas': val_kader
}

if btn_predict:
    predicted_stunting, composite_scores = aggregator.predict_from_breakdown(input_dict)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="📊 Estimasi Jumlah Balita Stunting", value=f"{predicted_stunting:.1f} Anak")
    with col2:
        st.info("Nilai prediksi diperoleh dari kalkulasi otomatis agregasi sub-faktor wilayah ke dalam model SVR.")

    # Grafik distribusi skor komposit hasil rata-rata breakdown
    df_comp = pd.DataFrame({
        "Kategori Komposit": ["Fasilitas Kesehatan", "Sanitasi & Air", "Layanan Gizi", "Fasilitas Desa"],
        "Skor Rata-Rata": composite_scores
    })
    
    fig_bar = px.bar(df_comp, x="Kategori Komposit", y="Skor Rata-Rata", color="Kategori Komposit", title="Nilai Komposit Hasil Agregasi Breakdown")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("👈 Silakan masukkan parameter rinci pada panel sebelah kiri, lalu klik tombol **'Jalankan Prediksi Stunting'**.")