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

## 🧠 Analisis Teknis
Proyek ini mengimplementasikan Custom Layer MedicalFeatureAttention untuk memberikan bobot lebih pada fitur klinis yang krusial bagi prediksi kardiovaskular.

Proses integrasi ini dirancang agar tim Frontend tidak perlu memahami kompleksitas deep learning di balik layar; cukup mengirimkan data mentah, dan model akan mengembalikan hasil prediksi yang sudah tervalidasi secara otomatis.

## ⚖️ Disclaimer Medis
Model ini adalah alat bantu skrining awal dan bukan pengganti diagnosis medis profesional. Setiap hasil prediksi 'BERISIKO TINGGI' wajib dikonfirmasi oleh tenaga medis melalui pemeriksaan klinis lebih lanjut.

## 📝 Catatan Tambahan
Keamanan: Pastikan file .env (berisi API Key Gemini jika digunakan) tidak pernah di-push ke repositori publik.

Integrasi: inference_v7.py telah menangani preprocessing otomatis. Divisi Frontend hanya perlu mengirimkan dictionary berisi 11 data klinis yang tepat sesuai dengan panduan dokumentasi integrasi.
