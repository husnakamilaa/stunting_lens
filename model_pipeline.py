# MODEL_PIPELINE.PY
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class StuntingAggregator(BaseEstimator, RegressorMixin):
    def __init__(self, trained_pipeline):
        self.trained_pipeline = trained_pipeline
        
        # Kelompok kolom breakdown sesuai dataset aslimu
        self.f_kes_cols = [
            'aktivitas_rutin_Penyelenggaraan_posyandu_',
            'jumlah_Ibu_Hamil_Periksa_Kehamilan/Nifas',
            'jumlah_Anak_usia_0-59_Bulan_terpantau_tumbuh_kembang_(datang_ke_posyandu/layanan_kesehatan_lainnya)',
            'jumlah_anak_usia_0-59_bulan_mengikuti_kegiatan_BKB/PAUD',
            'jumlah_Anak_usia_0-59_bulan_mendapatkan_imunisasi_dasar_lengkap'
        ]
        self.sanitasi_cols = [
            'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_ke_sumber_air_bersih/minum',
            'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_ke_jamban_sehat.',
            'jumlah_Keluarga_beresiko_stunting_dan_keluarga_rentan_memiliki_akses_sanitasi/pembuangan_limbah_layak'
        ]
        self.gizi_cols = [
            'jumlah_Remaja_Putri_Ikut_Pemeriksaan_Anemia_(Hb)',
            'jumlah_Remaja_Putri_Mendapat_Tablet_Tambah_Darah_(TTB)',
            'jumlah_Ibu_Hamil_mengkonsumsi_tablet_tambah_darah_(TTD)_(minimal_90_tablet_selama_masa_kehamilan)',
            'jumlah_Anak_usia_0-59_Bulan_mengalami_gizi_kurang/buruk/stunting_mendapatkan_tambahan_asupan_gizi_dan_konseling_gizi'
        ]
        self.desa_cols = [
            'desa_Melaksanakan_rembuk_stunting_desa_dengan_melibatkan_UPTD',
            'terdapan_Penetapan_anggaran__kegiatan_stunting_dalam_APB_Desa',
            'jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting',
            'terdapat_Pembentukan_RDS/TPPS',
            'terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas'
        ]

    def predict_from_breakdown(self, raw_input_dict):
        """Menerima dictionary input breakdown, merubahnya jadi 4 fitur komposit, lalu diprediksi oleh model joblib"""
        df_input = pd.DataFrame([raw_input_dict])
        
        # Hitung rata-rata komposit persis seperti saat melatih model di skripsimu
        comp_fk = df_input[self.f_kes_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
        comp_sa = df_input[self.sanitasi_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
        comp_lg = df_input[self.gizi_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
        comp_fd = df_input[self.desa_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).values[0]
        
        # Susun dataframe berisi 4 fitur komposit sesuai urutan asli model
        X_composite = pd.DataFrame([[comp_fk, comp_sa, comp_lg, comp_fd]], 
                                   columns=['Fasilitas_Kesehatan', 'Sanitasi_Air_Bersih', 'Layanan_Gizi', 'Fasilitas_Desa'])
        
        # Prediksi menggunakan model SVR asli yang sudah di-load
        prediction = self.trained_pipeline.predict(X_composite)[0]
        return max(0.0, prediction), [comp_fk, comp_sa, comp_lg, comp_fd]
    