# 🫀 CardioCare AI

Sistem prediksi risiko penyakit kardiovaskular berbasis *deep learning* (TensorFlow) yang dikembangkan sebagai Capstone Project Coding Camp 2026.

## 📋 Deskripsi Proyek
CardioCare AI adalah alat bantu skrining awal yang menggunakan *deep learning* untuk memprediksi risiko penyakit kardiovaskular. Dengan mengutamakan **Sensitivitas (Recall)** sebesar **85.94%** menggunakan *threshold medis* 0.40, sistem ini dioptimalkan untuk meminimalkan risiko *False Negative* (pasien berisiko yang tidak terdeteksi oleh sistem).

## 📂 Struktur Repositori
Struktur ini diatur agar alur data antara model, *preprocessing*, dan aplikasi API tetap terisolasi dan mudah dipelihara:

```text
cardio-care-project/
├── .gitignore             # Pengabaian file sampah & data sensitif (.env)
├── README.md              # Dokumentasi proyek
├── requirements.txt       # Daftar dependencies library
├── app.py                 # Main API (FastAPI) — pintu masuk aplikasi
└── model/                 # Folder khusus model
    ├── cardiocare_model_v7.keras  # File model TensorFlow utama
    ├── scaler_v7.pkl              # Scaler untuk normalisasi input
    ├── model_metadata.pkl         # Konfigurasi & versi model
    └── inference_v7.py            # Fungsi predict_cardio (API integrasi)
```

## ⚙️ Cara Penggunaan
1. Persiapan
Pastikan Python 3.9+ telah terpasang di sistem Anda.

2. Instalasi
Masuk ke direktori proyek dan instal dependencies yang diperlukan:

Bash 
```text
pip install -r requirements.txt
```
3. Menjalankan API
Jalankan server FastAPI dengan perintah berikut:

Bash
```text
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
4. API Endpoints
POST /predict   : Mengirimkan 11 data klinis pasien untuk mendapatkan hasil prediksi.

GET /health    : Mengecek status kesiapan server dan model.

GET /threshold : Melihat informasi threshold medis yang digunakan oleh sistem.
Dokumentasi API: Akses ```text http://localhost:8000/docs ``` untuk mencoba endpoint secara interaktif.

## Panduan Integrasi (Untuk Tim FS)Divisi Frontend/Full-Stack cukup mengirimkan 11 field data mentah ke fungsi predict_cardio() tanpa perlu melakukan scaling atau preprocessing manual.  Format JSON Input:JSON
```
{
  "age": 55, "gender": 1, "height": 165, "weight": 82,
  "ap_hi": 145, "ap_lo": 92, "cholesterol": 2, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 1
}
```
Contoh Response:JSON
```text
{
  "label": 1,
  "probability": 0.9036,
  "category": "BERISIKO TINGGI",
  "recommendation": "Segera konsultasikan ke dokter.",
  "top_risk_factors": ["Hipertensi", "Obesitas"]
}
```

## Tentang Model AI (CardioCare V7 Final)

Sistem prediksi risiko kardiovaskular berbasis *deep learning* yang dioptimalkan untuk performa klinis.

### Spesifikasi Teknis
| Komponen | Deskripsi |
| :--- | :--- |
| **Model** | `CardioCare_Final_V7` |
| **Arsitektur** | *Functional Keras Model* dengan *Medical Feature Attention* |
| **Input** | 24 fitur (demografi, tekanan darah, kolesterol, glukosa, & fitur komposit) |
| **Target** | *Recall* >78% (target penasihat medis) |
| **Loss Function** | *Clinical Asymmetric Loss* (bobot 1.5) |
| **Scaling** | `StandardScaler` (mencegah *data leakage*) |
| **Training** | *Early Stopping* (patience=15)|
 
## 🧠 Analisis Teknis
Proyek ini mengimplementasikan Custom Layer MedicalFeatureAttention untuk memberikan bobot lebih pada fitur klinis yang krusial bagi prediksi kardiovaskular. Sistem dirancang agar tim Frontend dapat bekerja secara efisien; cukup mengirimkan data mentah, sementara proses preprocessing dan inferensi dilakukan secara otomatis oleh inference_v7.py.

Catatan Integrasi: Tim pengembang wajib merujuk ke dokumen CardioCare_Dokumentasi_Integrasi_v7.docx untuk pemetaan 24 variabel input. Seluruh aset pendukung (model, metadata, scaler) tersedia di Folder Google Drive Proyek.

## ⚖️ Disclaimer Medis
Model ini adalah alat bantu skrining awal dan bukan pengganti diagnosis medis profesional. Setiap hasil prediksi 'BERISIKO TINGGI' wajib dikonfirmasi oleh tenaga medis melalui pemeriksaan klinis lebih lanjut.

## 📝 Catatan Tambahan
Keamanan: Pastikan file .env (berisi API Key Gemini jika digunakan) tidak pernah di-push ke repositori publik.

Integrasi: inference_v7.py telah menangani preprocessing otomatis. Divisi Frontend hanya perlu mengirimkan dictionary berisi 11 data klinis yang tepat sesuai dengan panduan dokumentasi integrasi.

> **Catatan Integrasi:** Tim pengembang wajib merujuk ke dokumen `CardioCare_Dokumentasi_Integrasi_v7.docx` untuk pemetaan 24 variabel input. Akses aset pendukung (model, metadata, scaler) tersedia di [Folder Google Drive Proyek](https://drive.google.com/drive/folders/1kcyELgWmln9VN8U5bDi17zxtGw1zk8vx).
